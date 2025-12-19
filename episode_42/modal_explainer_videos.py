from io import BytesIO
from pathlib import Path
from typing import Optional, List
import base64
import tempfile
import os

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("git", "ffmpeg", "espeak-ng")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode flask torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "uv pip install --system --compile-bytecode git+https://github.com/huggingface/diffusers git+https://github.com/Disty0/sdnq"
    )
    .run_commands(
        "uv pip install --system --compile-bytecode 'kokoro>=0.9.4' soundfile ffmpeg-python Pillow"
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

app = modal.App("explainer-videos")

with image.imports():
    import torch
    import os
    import diffusers
    from sdnq import SDNQConfig
    from sdnq.loader import apply_sdnq_options_to_model
    from flask import Flask, request, jsonify, send_file
    import soundfile as sf
    import ffmpeg
    from PIL import Image as PILImage


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


@app.cls(
    image=image,
    gpu="L40s",
    volumes=volumes,
    scaledown_window=120,
    timeout=15 * 60,
)
class KokoroTTSModel:
    @modal.enter()
    def enter(self):
        from kokoro import KPipeline

        print("Loading Kokoro TTS pipeline...")
        self.pipeline = KPipeline(lang_code="a")
        print("Kokoro TTS loaded successfully!")

    @modal.method()
    def generate_audio(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> bytes:
        """Generate audio from text using Kokoro TTS."""
        import numpy as np

        print(f"Generating audio for text: {text[:100]}...")

        # Generate audio segments
        audio_segments = []
        generator = self.pipeline(text, voice=voice, speed=speed)

        for i, (gs, ps, audio) in enumerate(generator):
            audio_segments.append(audio)

        # Concatenate all audio segments
        if audio_segments:
            full_audio = np.concatenate(audio_segments)
        else:
            raise ValueError("No audio generated")

        # Save to bytes
        byte_stream = BytesIO()
        sf.write(byte_stream, full_audio, 24000, format="WAV")
        byte_stream.seek(0)

        return byte_stream.getvalue()


@app.cls(
    image=image,
    volumes=volumes,
    scaledown_window=120,
    timeout=30 * 60,
    memory=4096,
    cpu=2,
)
class ExplainerVideoGenerator:
    """Generates explainer videos by combining images with narration audio."""

    @modal.enter()
    def enter(self):
        print("ExplainerVideoGenerator ready!")

    @modal.method()
    def generate_video(
        self,
        scenes: List[dict],
        voice: str = "af_heart",
        tts_speed: float = 1.0,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> bytes:
        """
        Generate a complete explainer video from scenes.

        Each scene should have:
        - narration: text to convert to speech
        - image_prompt: prompt for image generation
        """
        import subprocess
        import numpy as np

        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clip_files = []

            # Get model references
            image_model = ZImageTurboModel()
            tts_model = KokoroTTSModel()

            for idx, scene in enumerate(scenes):
                print(f"\n{'='*50}")
                print(f"Processing scene {idx + 1}/{len(scenes)}")
                print(f"{'='*50}")

                narration = scene.get("narration", "")
                image_prompt = scene.get("image_prompt", "")

                if not narration or not image_prompt:
                    print(
                        f"Skipping scene {idx + 1}: missing narration or image_prompt"
                    )
                    continue

                # Generate audio from narration
                print(f"Generating audio for scene {idx + 1}...")
                audio_bytes = tts_model.generate_audio.remote(
                    text=narration,
                    voice=voice,
                    speed=tts_speed,
                )

                audio_path = temp_path / f"scene_{idx}_audio.wav"
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"Audio saved to {audio_path}")

                # Generate image from prompt
                print(f"Generating image for scene {idx + 1}...")

                # Adjust dimensions to be divisible by 16
                def adjust_dimensions(w, h, divisor=16):
                    return (w // divisor) * divisor, (h // divisor) * divisor

                adj_width, adj_height = adjust_dimensions(image_width, image_height)

                image_bytes = image_model.inference.remote(
                    prompt=image_prompt,
                    width=adj_width,
                    height=adj_height,
                    num_inference_steps=9,
                    guidance_scale=0.0,
                )

                image_path = temp_path / f"scene_{idx}_image.png"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Resize image to exact target dimensions if needed
                if adj_width != image_width or adj_height != image_height:
                    img = PILImage.open(image_path)
                    img = img.resize(
                        (image_width, image_height), PILImage.Resampling.LANCZOS
                    )
                    img.save(image_path)

                print(f"Image saved to {image_path}")

                # Get audio duration using ffprobe
                probe = ffmpeg.probe(str(audio_path))
                audio_duration = float(probe["streams"][0]["duration"])
                print(f"Audio duration: {audio_duration:.2f}s")

                # Create video clip from image + audio using ffmpeg-python
                clip_path = temp_path / f"scene_{idx}_clip.mp4"

                try:
                    # Create video from image that loops for the duration of audio
                    # Then add audio track
                    (
                        ffmpeg.input(
                            str(image_path), loop=1, t=audio_duration, framerate=30
                        )
                        .output(
                            ffmpeg.input(str(audio_path)),
                            str(clip_path),
                            vcodec="libx264",
                            acodec="aac",
                            pix_fmt="yuv420p",
                            shortest=None,
                            **{"b:a": "192k"},
                        )
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    print(f"Video clip created: {clip_path}")
                    clip_files.append(str(clip_path))
                except ffmpeg.Error as e:
                    print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
                    raise

            if not clip_files:
                raise ValueError("No video clips were generated")

            # Merge all clips together
            print(f"\n{'='*50}")
            print(f"Merging {len(clip_files)} clips into final video...")
            print(f"{'='*50}")

            final_video_path = temp_path / "final_video.mp4"

            if len(clip_files) == 1:
                # Just copy the single clip
                import shutil

                shutil.copy(clip_files[0], final_video_path)
            else:
                # Create a concat file for ffmpeg
                concat_file = temp_path / "concat.txt"
                with open(concat_file, "w") as f:
                    for clip_file in clip_files:
                        f.write(f"file '{clip_file}'\n")

                # Concatenate all clips
                try:
                    (
                        ffmpeg.input(str(concat_file), format="concat", safe=0)
                        .output(
                            str(final_video_path),
                            vcodec="libx264",
                            acodec="aac",
                            pix_fmt="yuv420p",
                        )
                        .overwrite_output()
                        .run(quiet=True)
                    )
                except ffmpeg.Error as e:
                    print(
                        f"FFmpeg concat error: {e.stderr.decode() if e.stderr else str(e)}"
                    )
                    raise

            print(f"Final video created: {final_video_path}")

            # Read and return the final video
            with open(final_video_path, "rb") as f:
                video_bytes = f.read()

            print(f"Video size: {len(video_bytes) / (1024*1024):.2f} MB")
            return video_bytes


@app.function(image=image, volumes=volumes, cpu="0.5", memory="2GiB", timeout=30 * 60)
@modal.wsgi_app()
def flask_app():
    web_app = Flask(__name__)

    @web_app.route("/")
    def health_check():
        return jsonify({"status": "alive"})

    @web_app.route("/generate-video", methods=["POST"])
    def generate_video():
        """
        Generate an explainer video from scenes with narration and images.

        Request body should be a JSON array with the following structure:
        [
            {
                "scenes": [
                    {
                        "narration": "Text to convert to speech",
                        "image_prompt": "Prompt for image generation"
                    },
                    ...
                ]
            }
        ]

        Optional parameters (can be included in the first object):
        - **voice**: Kokoro TTS voice to use (default: "af_heart")
        - **tts_speed**: Speed of TTS (default: 1.0)
        - **image_width**: Width of generated images (default: 1920)
        - **image_height**: Height of generated images (default: 1080)
        - **return_base64**: If true, returns base64 encoded video instead of binary (default: false)
        """
        try:
            data = request.get_json(force=True)

            # Handle array input format
            if isinstance(data, list):
                if len(data) == 0:
                    return jsonify({"error": "Empty input array"}), 400
                first_item = data[0]
                scenes = first_item.get("scenes", [])
                voice = first_item.get("voice", "af_heart")
                tts_speed = first_item.get("tts_speed", 1.0)
                image_width = first_item.get("image_width", 1920)
                image_height = first_item.get("image_height", 1080)
                return_base64 = first_item.get("return_base64", False)
            else:
                scenes = data.get("scenes", [])
                voice = data.get("voice", "af_heart")
                tts_speed = data.get("tts_speed", 1.0)
                image_width = data.get("image_width", 1920)
                image_height = data.get("image_height", 1080)
                return_base64 = data.get("return_base64", False)

            if not scenes:
                return jsonify({"error": "'scenes' array is required"}), 400

            # Validate scenes
            for idx, scene in enumerate(scenes):
                if not scene.get("narration"):
                    return (
                        jsonify({"error": f"Scene {idx + 1} is missing 'narration'"}),
                        400,
                    )
                if not scene.get("image_prompt"):
                    return (
                        jsonify(
                            {"error": f"Scene {idx + 1} is missing 'image_prompt'"}
                        ),
                        400,
                    )

            print(f"Generating video with {len(scenes)} scenes...")
            print(f"Voice: {voice}, TTS Speed: {tts_speed}")
            print(f"Image dimensions: {image_width}x{image_height}")

            # Generate the video
            video_generator = ExplainerVideoGenerator()
            video_bytes = video_generator.generate_video.remote(
                scenes=scenes,
                voice=voice,
                tts_speed=tts_speed,
                image_width=image_width,
                image_height=image_height,
            )

            if return_base64:
                # Return as base64 encoded string
                video_base64 = base64.b64encode(video_bytes).decode("utf-8")
                return jsonify(
                    {
                        "video": video_base64,
                        "format": "mp4",
                        "size_bytes": len(video_bytes),
                    }
                )
            else:
                # Return as binary file
                return send_file(
                    BytesIO(video_bytes),
                    mimetype="video/mp4",
                    as_attachment=True,
                    download_name="explainer_video.mp4",
                )

        except Exception as e:
            import traceback

            print(f"Error generating video: {e}")
            print(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    return web_app
