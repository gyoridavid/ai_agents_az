from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.13",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --compile-bytecode --index-strategy unsafe-best-match Pillow~=11.2.1 safetensors~=0.5.3 transformers~=4.53.0 sentencepiece~=0.2.0 torch==2.7.1 torchvision torchaudio optimum-quanto==0.2.7 fastapi[standard]==0.115.4 python-multipart==0.0.12 https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0/nunchaku-1.0.0+torch2.7-cp313-cp313-linux_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu128"
    )
)

MODEL_NAME = "black-forest-labs/FLUX.1-schnell"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("flux-app-secrets", required_keys=["HF_TOKEN"])]

image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)

app = modal.App("nunchaku-flux-schnell-fastapi")

with image.imports():
    import torch
    import os
    from diffusers import FluxPipeline
    from nunchaku import NunchakuFluxTransformer2DModelV2
    from nunchaku.utils import get_precision
    from PIL import Image
    from fastapi import FastAPI, Form, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel


@app.cls(
    image=image,
    gpu="L40s",
    volumes=volumes,
    secrets=secrets,
    scaledown_window=120,
    timeout=10 * 60,  # 10 minutes
)
class NunchakuFluxSchnellModel:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} and Nunchaku transformer if necessary...")

        self.dtype = torch.bfloat16
        self.device = "cuda"

        # Auto-detect precision (int4 or fp4) based on GPU
        self.precision = get_precision()
        print(f"Detected precision: {self.precision}")

        # Load the quantized transformer
        self.transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-schnell/svdq-{self.precision}_r32-flux.1-schnell.safetensors",
            cache_dir=CACHE_DIR,
        )

        # Load the full pipeline with the quantized transformer
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_NAME,
            transformer=self.transformer,
            torch_dtype=self.dtype,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        ).to(self.device)

        print("Model loaded successfully!")

    @modal.method()
    def inference(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> bytes:
        # Use provided seed or generate a random one
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)

        print(f"Generating image with prompt: {prompt}")
        print(f"Using precision: {self.precision}")

        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pil",
            generator=generator,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


# Create a separate FastAPI app to handle the web interface
@app.function(image=image, volumes=volumes, secrets=secrets, cpu="0.5", memory="2GiB")
@modal.asgi_app()
def fastapi_app():
    from fastapi import Depends, FastAPI, Form, HTTPException, status
    from fastapi.responses import Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel

    class GenerateImageRequest(BaseModel):
        prompt: str
        width: int = 1024
        height: int = 1024
        steps: int = 4
        guidance_scale: float = 0.0
        seed: Optional[int] = None

    web_app = FastAPI(
        title="Nunchaku Flux Schnell Generator",
        description="Generate images using Nunchaku-quantized Flux Schnell model",
        version="1.0.0",
    )

    @web_app.post("/generate")
    async def generate_image(
        request: GenerateImageRequest,
        token: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        """
        Generate an image using Nunchaku-quantized Flux Schnell model.

        - **prompt**: Text description of the image you want to generate
        - **width**: Image width in pixels (default: 1024)
        - **height**: Image height in pixels (default: 1024)
        - **steps**: Number of denoising steps (default: 4, optimal for Schnell)
        - **guidance_scale**: Guidance scale (default: 0.0, optimal for Schnell)
        - **seed**: Optional seed for reproducible results
        """

        if os.environ.get("BEARER_TOKEN", False):
            if not token or token.credentials != os.environ["BEARER_TOKEN"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        try:
            # Validate dimensions (must be divisible by 8 for FLUX)
            if request.width % 8 != 0 or request.height % 8 != 0:
                raise HTTPException(
                    status_code=400,
                    detail="Width and height must be divisible by 8.",
                )

            if request.width * request.height > 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum image size is 1 megapixel (e.g., 1024x1024 or 2048x512).",
                )

            if request.width < 256 or request.height < 256:
                raise HTTPException(
                    status_code=400,
                    detail="Minimum image size is 256x256 pixels.",
                )

            model = NunchakuFluxSchnellModel()
            result_bytes = model.inference.remote(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )

            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={
                    "Content-Disposition": f"inline; filename=nunchaku_flux_schnell_{request.seed or 'random'}.png"
                },
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating image: {str(e)}"
            )

    return web_app
