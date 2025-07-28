from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

# Define the container image with all necessary dependencies
diffusers_commit_sha = "00f95b9755718aabb65456e791b8408526ae6e76"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --compile-bytecode --index-strategy unsafe-best-match accelerate~=1.8.1 git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha} huggingface-hub[hf-transfer]~=0.33.1 Pillow~=11.2.1 safetensors~=0.5.3 transformers~=4.53.0 sentencepiece~=0.2.0 torch==2.7.1 optimum-quanto==0.2.7 fastapi[standard]==0.115.4 python-multipart==0.0.12 --extra-index-url https://download.pytorch.org/whl/cu128"
    )
)

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
MODEL_REVISION = "f9fdd1a95e0dfd7653cb0966cda2486745122695"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])]

image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)

app = modal.App("flux-kontext-fastapi")

with image.imports():
    import torch
    import os
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import Response

@app.cls(
    image=image,
    cpu="0.5",
    memory="2GiB",
    gpu="L40s",
    volumes=volumes,
    secrets=secrets,
    scaledown_window=120,
    timeout=10 * 60, # 10 minutes
)
class FluxKontextModel:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")

        dtype = torch.bfloat16
        self.device = "cuda"

        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        ).to(self.device)

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
    ) -> bytes:
        # Use provided seed or generate a random one
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)

        init_image = load_image(Image.open(BytesIO(image_bytes)))

        image = self.pipe(
            width=width,
            height=height,
            image=init_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
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
    from fastapi import Depends, FastAPI, File, UploadFile, Form, HTTPException, status
    from fastapi.responses import Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from PIL import Image
    from io import BytesIO

    web_app = FastAPI(
        title="Flux Kontext Image Editor",
        description="Edit images using Flux Kontext Dev model",
        version="1.0.0",
    )

    @web_app.post("/edit_image")
    async def edit_image(
        image: UploadFile = File(..., description="Input image file"),
        prompt: str = Form(..., description="Text prompt describing the desired edit"),
        token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        guidance_scale: float = Form(
            3.5, description="Guidance scale (higher = more prompt adherence)"
        ),
        num_inference_steps: int = Form(20, description="Number of inference steps"),
        seed: int = Form(None, description="Random seed for reproducible results"),
    ):
        """
        Edit an image using Flux Kontext Dev model.

        - **image**: Upload an image file (PNG, JPG, JPEG)
        - **prompt**: Text description of how you want to edit the image
        - **guidance_scale**: Controls how closely the model follows the prompt (default: 3.5)
        - **num_inference_steps**: Number of denoising steps (default: 20)
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
            if image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Please upload a PNG or JPEG image.",
                )

            image_bytes = await image.read()

            # Validate image size
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(
                    status_code=400,
                    detail="Image file too large. Please upload an image smaller than 10MB.",
                )

            pil_image = Image.open(BytesIO(image_bytes))
            width, height = pil_image.size
            
            aspect_ratio_float = width / height
            # https://docs.bfl.ai/kontext/kontext_text_to_image#flux-1-kontext-text-to-image-parameters
            min_ratio = 3 / 7
            max_ratio = 7 / 3
            
            if aspect_ratio_float < min_ratio or aspect_ratio_float > max_ratio:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid aspect ratio. Image aspect ratio ({width}:{height}) must be between 3:7 and 7:3. Current ratio: {aspect_ratio_float:.2f}",
                )

            model = FluxKontextModel()
            result_bytes = model.inference.remote(
                width=width,
                height=height,
                image_bytes=image_bytes,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )

            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={"Content-Disposition": "inline; filename=edited_image.png"},
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing image: {str(e)}"
            )

    return web_app


# ## CLI Interface
# You can test the model locally using the CLI interface.
# Run with: modal run flux_fastapi.py --prompt "your prompt here" --image-path "path/to/image.jpg" --output-path "path/to/output.jpg"
@app.local_entrypoint()
def main(
    image_path: str = "demo_images/dog.png",
    output_path: str = "/tmp/edited_image.png",
    prompt: str = "A cute dog wizard inspired by Gandalf from Lord of the Rings, featuring detailed fantasy elements in Studio Ghibli style",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 20,
    seed: Optional[int] = None,
):
    """
    Test the Flux Kontext model locally via CLI.
    """
    print(f"🎨 Reading input image from {image_path}")

    try:
        input_image_bytes = Path(image_path).read_bytes()
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        return

    pil_image = Image.open(BytesIO(input_image_bytes))
    width, height = pil_image.size

    aspect_ratio_float = width / height
    # https://docs.bfl.ai/kontext/kontext_text_to_image#flux-1-kontext-text-to-image-parameters
    min_ratio = 3 / 7
    max_ratio = 7 / 3

    if aspect_ratio_float < min_ratio or aspect_ratio_float > max_ratio:
        print(f"❌ Error: Invalid aspect ratio. Image aspect ratio ({width}:{height}) must be between 3:7 and 7:3. Current ratio: {aspect_ratio_float:.2f}")
        return

    print(f"🎨 Editing image with prompt: {prompt}")

    model = FluxKontextModel()
    output_image_bytes = model.inference.remote(
        image_bytes=input_image_bytes,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    print(f"🎨 Saving output image to {output_path}")
    output_path.write_bytes(output_image_bytes)

    print("✅ Image editing completed successfully!")
