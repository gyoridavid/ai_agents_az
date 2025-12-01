from io import BytesIO
from pathlib import Path
from typing import Optional
import base64

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode flask torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "uv pip install --system --compile-bytecode git+https://github.com/huggingface/diffusers git+https://github.com/Disty0/sdnq"
    )
)

MODEL_NAME = "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

image = image.env(
    {
        "HF_HOME": str(CACHE_DIR),
    }
)

app = modal.App("z-image-turbo-fastapi")

with image.imports():
    import torch
    import os
    import diffusers
    from sdnq import SDNQConfig
    from sdnq.loader import apply_sdnq_options_to_model
    from flask import Flask, request, jsonify


@app.cls(
    image=image,
    gpu="L40s",
    volumes=volumes,
    scaledown_window=120,
    timeout=10 * 60,
)
class ZImageTurboModel:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} and applying SDNQ optimizations...")

        # Enable TensorFloat32 for better performance on modern GPUs
        torch.set_float32_matmul_precision("high")

        self.device = "cuda"

        # Load the Z-Image pipeline with SDNQ quantization
        self.pipe = diffusers.ZImagePipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cuda",
            cache_dir=CACHE_DIR,
        )

        # Apply SDNQ optimizations
        self.pipe.transformer = apply_sdnq_options_to_model(
            self.pipe.transformer, use_quantized_matmul=True
        )
        self.pipe.text_encoder = apply_sdnq_options_to_model(
            self.pipe.text_encoder, use_quantized_matmul=True
        )

        print("Model loaded successfully!")

    @modal.method()
    def inference(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> bytes:
        import random

        def adjust_dimensions_preserve_aspect(w, h, divisor=16, min_size=256):
            """Adjust dimensions to be divisible by divisor while preserving aspect ratio."""
            aspect_ratio = w / h

            # Start from width, go up and down to find first valid pair
            for offset in range(0, max(w, 1000), divisor):
                for test_w in [w + offset, w - offset]:
                    if test_w < min_size:
                        continue
                    if test_w % divisor != 0:
                        continue

                    # Calculate height from aspect ratio
                    test_h = round(test_w / aspect_ratio)
                    if test_h < min_size:
                        continue
                    if test_h % divisor == 0:
                        return int(test_w), int(test_h)

            # Fallback: just round both
            return (w // divisor) * divisor, (h // divisor) * divisor

        # Adjust dimensions while preserving aspect ratio
        width, height = adjust_dimensions_preserve_aspect(width, height)

        # Use provided seed or generate random one
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.manual_seed(seed)

        print(f"Generating image with prompt: {prompt}")
        print(f"Dimensions: {width}x{height}")

        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


@app.function(image=image, volumes=volumes, cpu="0.5", memory="2GiB")
@modal.wsgi_app()
def flask_app():
    web_app = Flask(__name__)

    @web_app.route("/")
    def health_check():
        return jsonify({"status": "alive"})

    @web_app.route("/generate", methods=["POST"])
    def generate_image():
        """
        Generate an image using Z-Image Turbo SDNQ model.

        - **prompt**: Text description of the image you want to generate
        - **width**: Desired image width in pixels (default: 1024)
        - **height**: Desired image height in pixels (default: 1024)
        - **steps**: Number of inference steps (default: 9)
        - **guidance_scale**: Guidance scale (default: 0.0)
        - **seed**: Seed for reproducible results (default: 42)
        """
        data = request.get_json(force=True)
        prompt = data.get("prompt")
        width = data.get("width", 1024)
        height = data.get("height", 1024)
        steps = data.get("steps", 9)
        guidance_scale = data.get("guidance_scale", 0.0)
        seed = data.get("seed")  # None if not provided, will use random

        if not prompt:
            return jsonify({"error": "'prompt' parameter is required"}), 400

        def adjust_dimensions_preserve_aspect(w, h, divisor=16, min_size=256):
            """Adjust dimensions to be divisible by divisor while preserving aspect ratio."""
            aspect_ratio = w / h

            # Start from width, go up and down to find first valid pair
            for offset in range(0, max(w, 1000), divisor):
                for test_w in [w + offset, w - offset]:
                    if test_w < min_size:
                        continue
                    if test_w % divisor != 0:
                        continue

                    # Calculate height from aspect ratio
                    test_h = round(test_w / aspect_ratio)
                    if test_h < min_size:
                        continue
                    if test_h % divisor == 0:
                        return int(test_w), int(test_h)

            # Fallback: just round both
            return (w // divisor) * divisor, (h // divisor) * divisor

        try:
            # Adjust dimensions while preserving aspect ratio
            adjusted_width, adjusted_height = adjust_dimensions_preserve_aspect(
                width, height
            )

            model = ZImageTurboModel()
            result_bytes = model.inference.remote(
                prompt=prompt,
                width=adjusted_width,
                height=adjusted_height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

            # Encode as base64
            img_base64 = base64.b64encode(result_bytes).decode("utf-8")

            return jsonify(
                {
                    "image": img_base64,
                    "format": "png",
                }
            )

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": str(e)}), 500

    return web_app
