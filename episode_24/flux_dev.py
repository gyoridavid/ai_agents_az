# ---
# output-directory: "/tmp/flux"
# args: ["--no-compile"]
# ---

# # Run Flux fast on H100s with `torch.compile`

# _Update: To speed up inference by another >2x, check out the additional optimization
# techniques we tried in [this blog post](https://modal.com/blog/flux-3x-faster)!_

# In this guide, we'll run Flux as fast as possible on Modal using open source tools.
# We'll use `torch.compile` and NVIDIA H100 GPUs.

# ## Setting up the image and dependencies

import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

# We'll make use of the full [CUDA toolkit](https://modal.com/docs/guide/cuda)
# in this example, so we'll build our container image off of the `nvidia/cuda` base.

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])

# Now we install most of our dependencies with `apt` and `pip`.
# For Hugging Face's [Diffusers](https://github.com/huggingface/diffusers) library
# we install from GitHub source and so pin to a specific commit.

# PyTorch added faster attention kernels for Hopper GPUs in version 2.5.

diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
        "fastapi[standard]==0.115.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)

# Later, we'll also use `torch.compile` to increase the speed further.
# Torch compilation needs to be re-executed when each new container starts,
# so we turn on some extra caching to reduce compile times for later containers.

flux_image = flux_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/cache/.inductor_cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "CUDA_CACHE_PATH": "/cache/.nv_cache",
        "TRITON_CACHE_DIR": "/cache/.triton_cache",
    }
)

# Finally, we construct our Modal [App](https://modal.com/docs/reference/modal.App),
# set its default image to the one we just constructed,
# and import `FluxPipeline` for downloading and running Flux.1.

app = modal.App("example-flux", image=flux_image)

with flux_image.imports():
    import torch
    from diffusers import FluxPipeline
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    from typing import Optional
    import os

    # Fix for transformers cache migration issue
    try:
        from transformers.utils.hub import move_cache

        move_cache()
    except Exception:
        # Ignore if cache migration fails or is not needed
        pass

# ## Defining a parameterized `Model` inference class

# Next, we map the model's setup and inference code onto Modal.

# 1. We run the model setup in the method decorated with `@modal.enter()`. This includes loading the
# weights and moving them to the GPU, along with an optional `torch.compile` step (see details below).
# The `@modal.enter()` decorator ensures that this method runs only once, when a new container starts,
# instead of in the path of every call.

# 2. We run the actual inference in methods decorated with `@modal.method()`.

# *Note: Access to the Flux.1-schnell model on Hugging Face is
# [gated by a license agreement](https://huggingface.co/docs/hub/en/models-gated)
# which you must agree to
# [here](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
# After you have accepted the license,
# [create a Modal Secret](https://modal.com/secrets)
# with the name `huggingface-secret` following the instructions in the template.*

MINUTES = 60  # seconds
VARIANT = "schnell"  # schnell or dev
NUM_INFERENCE_STEPS = 4  # use ~50 for [dev], (1-4) for [schnell]

CACHE_VOLUME = modal.Volume.from_name("cache_volume", create_if_missing=True)


@app.cls(
    gpu="L40S",  # fast GPU with strong software support
    scaledown_window=10 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    enable_memory_snapshot=True,
    volumes={  # add Volumes to store serializable compilation artifacts
        "/cache": CACHE_VOLUME,
    },
    secrets=[modal.Secret.from_name("flux-app-secrets")],
)
class Model:
    compile: bool = modal.parameter(  # see section on torch.compile below for details
        default=False
    )

    @modal.enter(snap=True)
    def load(self):
        """Load model to CPU for memory snapshot."""
        pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{VARIANT}",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cpu")
        self.pipe = pipe

    @modal.enter(snap=False)
    def setup(self):
        """Move model to GPU and apply optimizations."""
        self.pipe.to("cuda")
        self.pipe = optimize(self.pipe, compile=self.compile)

        # Apply TeaCache for 2x speedup (approximate caching technique)
        self.setup_teacache()
        print("🚀 TeaCache setup completed for 2x speedup")

    def setup_teacache(self):
        """Setup TeaCache for approximate caching speedup."""
        # Enable TeaCache on the transformer
        self.pipe.transformer.__class__.enable_teacache = True
        self.pipe.transformer.__class__.cnt = 0
        self.pipe.transformer.__class__.num_steps = NUM_INFERENCE_STEPS
        self.pipe.transformer.__class__.rel_l1_thresh = 0.6  # 2.0x speedup threshold
        self.pipe.transformer.__class__.accumulated_rel_l1_distance = 0
        self.pipe.transformer.__class__.previous_input = None
        self.pipe.transformer.__class__.previous_result = None

        # Monkey patch the forward method with TeaCache logic
        self.patch_transformer_forward()

    def patch_transformer_forward(self):
        """Patch the transformer forward method with TeaCache logic."""
        import numpy as np

        original_forward = self.pipe.transformer.forward

        def teacache_forward(
            hidden_states,
            encoder_hidden_states=None,
            pooled_projections=None,
            timestep=None,
            img_ids=None,
            txt_ids=None,
            guidance=None,
            joint_attention_kwargs=None,
            **kwargs,
        ):
            if (
                hasattr(self.pipe.transformer, "enable_teacache")
                and self.pipe.transformer.enable_teacache
            ):
                # Simplified TeaCache logic - determine if we should skip computation
                should_calc = True

                if hasattr(self.pipe.transformer, "cnt") and hasattr(
                    self.pipe.transformer, "num_steps"
                ):
                    # Use hidden_states directly for change detection (simplified approach)
                    current_input = hidden_states.clone()

                    if (
                        self.pipe.transformer.cnt == 0
                        or self.pipe.transformer.cnt
                        == self.pipe.transformer.num_steps - 1
                    ):
                        should_calc = True
                        self.pipe.transformer.accumulated_rel_l1_distance = 0
                    else:
                        # TeaCache coefficients for FLUX (from official implementation)
                        coefficients = [
                            4.98651651e02,
                            -2.83781631e02,
                            5.58554382e01,
                            -3.82021401e00,
                            2.64230861e-01,
                        ]
                        rescale_func = np.poly1d(coefficients)

                        if (
                            hasattr(self.pipe.transformer, "previous_input")
                            and self.pipe.transformer.previous_input is not None
                        ):
                            # Calculate relative L1 distance between inputs
                            rel_diff = (
                                current_input - self.pipe.transformer.previous_input
                            ).abs().mean() / self.pipe.transformer.previous_input.abs().mean()

                            self.pipe.transformer.accumulated_rel_l1_distance += (
                                rescale_func(rel_diff.cpu().item())
                            )

                            if (
                                self.pipe.transformer.accumulated_rel_l1_distance
                                < self.pipe.transformer.rel_l1_thresh
                            ):
                                should_calc = False
                            else:
                                should_calc = True
                                self.pipe.transformer.accumulated_rel_l1_distance = 0

                    self.pipe.transformer.previous_input = current_input
                    self.pipe.transformer.cnt += 1

                    if self.pipe.transformer.cnt == self.pipe.transformer.num_steps:
                        self.pipe.transformer.cnt = 0

                    # Use cached result if we're skipping computation
                    if (
                        not should_calc
                        and hasattr(self.pipe.transformer, "previous_result")
                        and self.pipe.transformer.previous_result is not None
                    ):
                        # Return cached result
                        return self.pipe.transformer.previous_result
                    else:
                        # Compute normally and cache the result
                        result = original_forward(
                            hidden_states,
                            encoder_hidden_states,
                            pooled_projections,
                            timestep,
                            img_ids,
                            txt_ids,
                            guidance,
                            joint_attention_kwargs,
                            **kwargs,
                        )
                        self.pipe.transformer.previous_result = result
                        return result

            # Fallback to original forward if TeaCache is disabled
            return original_forward(
                hidden_states,
                encoder_hidden_states,
                pooled_projections,
                timestep,
                img_ids,
                txt_ids,
                guidance,
                joint_attention_kwargs,
                **kwargs,
            )

        # Replace the forward method
        self.pipe.transformer.forward = teacache_forward

    @modal.method()
    def inference(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> bytes:
        print("🎨 generating image...")

        # Use provided seed or generate a random one
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = torch.Generator(device="cuda")

        out = self.pipe(
            prompt,
            width=width,
            height=height,
            output_type="pil",
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="PNG")
        return byte_stream.getvalue()


# Create a separate FastAPI app to handle the web interface
@app.function(image=flux_image, cpu="0.5", memory="2GiB")
@modal.asgi_app()
def fastapi_app():
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.responses import Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    from typing import Optional
    import os

    class GenerateRequest(BaseModel):
        prompt: str
        width: int = 1024
        height: int = 1024
        seed: Optional[int] = None

    web_app = FastAPI(
        title="Flux Image Generator",
        description="Generate images using Flux model",
        version="1.0.0",
    )

    @web_app.post("/generate")
    async def generate_image(
        request: GenerateRequest,
        token: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        """
        Generate an image using Flux model.

        - **prompt**: Text description of the image you want to generate
        - **width**: Width of the generated image (default: 1024)
        - **height**: Height of the generated image (default: 1024)
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
            model = Model()
            result_bytes = model.inference.remote(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                seed=request.seed,
            )

            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={"Content-Disposition": "inline; filename=generated_image.png"},
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating image: {str(e)}"
            )

    return web_app


# ## Calling our inference function

# To generate an image we just need to call the `Model`'s `generate` method
# with `.remote` appended to it.
# You can call `.generate.remote` from any Python environment that has access to your Modal credentials.
# The local environment will get back the image as bytes.

# Here, we wrap the call in a Modal [`local_entrypoint`](https://modal.com/docs/reference/modal.App#local_entrypoint)
# so that it can be run with `modal run`:

# ```bash
# modal run flux.py
# ```

# By default, we call `generate` twice to demonstrate how much faster
# the inference is after cold start. In our tests, clients received images in about 1.2 seconds.
# We save the output bytes to a temporary file.


@app.local_entrypoint()
def main(
    prompt: str = "a computer screen showing ASCII terminal art of the"
    " word 'Modal' in neon green. two programmers are pointing excitedly"
    " at the screen.",
    twice: bool = True,
    compile: bool = False,
):
    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(prompt)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(prompt)
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")

    output_path = Path("/tmp") / "flux" / "output.jpg"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"🎨 saving output to {output_path}")
    output_path.write_bytes(image_bytes)


# ## Speeding up Flux with `torch.compile`

# By default, we do some basic optimizations, like adjusting memory layout
# and re-expressing the attention head projections as a single matrix multiplication.
# But there are additional speedups to be had!

# PyTorch 2 added a compiler that optimizes the
# compute graphs created dynamically during PyTorch execution.
# This feature helps close the gap with the performance of static graph frameworks
# like TensorRT and TensorFlow.

# Here, we follow the suggestions from Hugging Face's
# [guide to fast diffusion inference](https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion),
# which we verified with our own internal benchmarks.
# Review that guide for detailed explanations of the choices made below.

# The resulting compiled Flux `schnell` deployment returns images to the client in under a second (~700 ms), according to our testing.
# _Super schnell_!

# Compilation takes up to twenty minutes on first iteration.
# As of time of writing in late 2024,
# the compilation artifacts cannot be fully serialized,
# so some compilation work must be re-executed every time a new container is started.
# That includes when scaling up an existing deployment or the first time a Function is invoked with `modal run`.

# We cache compilation outputs from `nvcc`, `triton`, and `inductor`,
# which can reduce compilation time by up to an order of magnitude.
# For details see [this tutorial](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

# You can turn on compilation with the `--compile` flag.
# Try it out with:

# ```bash
# modal run flux.py --compile
# ```

# The `compile` option is passed by a [`modal.parameter`](https://modal.com/docs/reference/modal.parameter#modalparameter) on our class.
# Each different choice for a `parameter` creates a [separate auto-scaling deployment](https://modal.com/docs/guide/parameterized-functions).
# That means your client can use arbitrary logic to decide whether to hit a compiled or eager endpoint.


def optimize(pipe, compile=True):
    # fuse QKV projections in Transformer and VAE
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags with enhanced configuration
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls
    config.shape_padding = True  # new optimization from blog post

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    # using max-autotune-no-cudagraphs and dynamic=True as per blog post
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="max-autotune-no-cudagraphs",
        dynamic=True,
        fullgraph=True,
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune-no-cudagraphs", dynamic=True, fullgraph=True
    )

    # trigger torch compilation
    print("🔦 running torch compilation (may take up to 20 minutes)...")

    pipe(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        height=1024,
        width=1024,
        num_inference_steps=NUM_INFERENCE_STEPS,  # use ~50 for [dev], smaller for [schnell]
        num_images_per_prompt=1,
    ).images[0]

    print("🔦 finished torch compilation")

    return pipe
