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
        f"uv pip install --system --compile-bytecode --index-strategy unsafe-best-match hf_transfer Pillow~=11.2.1 safetensors~=0.5.3 transformers~=4.53.0 sentencepiece~=0.2.0 torch==2.7.1 torchvision torchaudio optimum-quanto==0.2.7 fastapi[standard]==0.115.4 python-multipart==0.0.12 diffusers https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.0/nunchaku-1.0.0+torch2.7-cp313-cp313-linux_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu128"
    )
)

MODEL_NAME = "Qwen/Qwen-Image"

# Qwen model configuration
num_inference_steps = 4  # you can also use the 8-step model to improve the quality
rank = 32  # you can also use the rank=128 model to improve the quality

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

app = modal.App("nunchaku-qwen-image-fastapi")

with image.imports():
    import torch
    import os
    import math
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
    from nunchaku.models.transformers.transformer_qwenimage import (
        NunchakuQwenImageTransformer2DModel,
    )
    from nunchaku.utils import get_precision, get_gpu_memory
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
class NunchakuQwenImageModel:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} and Nunchaku transformer if necessary...")

        self.dtype = torch.bfloat16
        self.device = "cuda"

        # Auto-detect precision (int4 or fp4) based on GPU
        self.precision = get_precision()
        print(f"Detected precision: {self.precision}")

        # Scheduler configuration for Qwen
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # Model paths for different step configurations
        model_paths = {
            4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{self.precision}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
            8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{self.precision}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
        }

        # Load the quantized transformer
        self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            model_paths[num_inference_steps],
            cache_dir=CACHE_DIR,
        )

        # Load the full pipeline with the quantized transformer
        self.pipe = QwenImagePipeline.from_pretrained(
            MODEL_NAME,
            transformer=self.transformer,
            scheduler=self.scheduler,
            torch_dtype=self.dtype,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        ).to(self.device)

        # Memory optimization based on GPU memory
        if get_gpu_memory() > 18:
            self.pipe.enable_model_cpu_offload()
        else:
            # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
            self.transformer.set_offload(True)
            self.pipe._exclude_from_cpu_offload.append("transformer")
            self.pipe.enable_sequential_cpu_offload()

        print("Model loaded successfully!")

    @modal.method()
    def inference(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        true_cfg_scale: float = 1.0,
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
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
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
        negative_prompt: str = ""
        width: int = 1024
        height: int = 1024
        true_cfg_scale: float = 1.0
        seed: Optional[int] = None

    web_app = FastAPI(
        title="Nunchaku Qwen Image Generator",
        description="Generate images using Nunchaku-quantized Qwen Image model",
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
        Generate an image using Nunchaku-quantized Qwen Image model.

        - **prompt**: Text description of the image you want to generate
        - **negative_prompt**: Text description of what you don't want in the image
        - **width**: Image width in pixels (default: 1024)
        - **height**: Image height in pixels (default: 1024)
        - **true_cfg_scale**: True CFG scale (default: 1.0, optimal for Qwen)
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
            # Validate reasonable size limits
            if request.width > 2048 or request.height > 2048:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum image size is 2048x2048 pixels.",
                )

            if request.width < 256 or request.height < 256:
                raise HTTPException(
                    status_code=400,
                    detail="Minimum image size is 256x256 pixels.",
                )

            model = NunchakuQwenImageModel()
            result_bytes = model.inference.remote(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                true_cfg_scale=request.true_cfg_scale,
                seed=request.seed,
            )

            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={
                    "Content-Disposition": f"inline; filename=nunchaku_qwen_image_{request.seed or 'random'}.png"
                },
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating image: {str(e)}"
            )

    return web_app
