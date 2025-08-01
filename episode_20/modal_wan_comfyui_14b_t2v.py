import json
import subprocess
import uuid
import os
from pathlib import Path
from typing import Dict

import modal
import modal.experimental

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(python_version="3.11")  # start from basic Linux with Python
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    .pip_install("comfy-cli==1.4.1")  # install comfy-cli
    .run_commands(  # use comfy-cli to install ComfyUI and its dependencies
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.47"
    )
)

# ## Downloading custom nodes

# comfyui-kjnodes

# We'll also use `comfy-cli` to download custom nodes, in this case the popular [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui).

# Use the [ComfyUI Registry](https://registry.comfy.org/) to find the specific custom node name to use with this command.

image = (
    image.run_commands(  # download a custom node
        "comfy node install --fast-deps was-node-suite-comfyui@1.0.2",
        "git clone https://github.com/ChenDarYen/ComfyUI-NAG.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-NAG",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
    )
    # Add .run_commands(...) calls for any other custom nodes you want to download
)


# See [this post](https://modal.com/blog/comfyui-custom-nodes) for more examples
# on how to install popular custom nodes like ComfyUI Impact Pack and ComfyUI IPAdapter Plus.

# ## Downloading models

# `comfy-cli` also supports downloading models, but we've found it's faster to use
# [`hf_hub_download`](https://huggingface.co/docs/huggingface_hub/en/guides/download#download-a-single-file)
# directly by:

# 1. Enabling [faster downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads)
# 2. Mounting the cache directory to a [Volume](https://modal.com/docs/guide/volumes)

# By persisting the cache to a Volume, you avoid re-downloading the models every time you rebuild your image.

secrets = [modal.Secret.from_name("flux-app-secrets", required_keys=["HF_TOKEN"])]


def hf_download():
    from huggingface_hub import hf_hub_download

    # Debug: Check if HF_TOKEN is available
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")

    wan_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
        cache_dir="/cache",
        token=hf_token,
    )

    subprocess.run(
        f"ln -s {wan_model} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
        shell=True,
        check=True,
    )

    wan_model_low_noise = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
        cache_dir="/cache",
        token=hf_token,
    )

    subprocess.run(
        f"ln -s {wan_model_low_noise} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
        shell=True,
        check=True,
    )

    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan_2.1_vae.safetensors",
        cache_dir="/cache",
        token=hf_token,
    )

    subprocess.run(
        f"ln -s {vae_model} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors",
        shell=True,
        check=True,
    )

    t5_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
        token=hf_token,
    )

    subprocess.run(
        f"ln -s {t5_model} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        shell=True,
        check=True,
    )


vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        hf_download,
        # persist the HF cache to a Modal Volume so future runs don't re-download models
        volumes={"/cache": vol},
        secrets=secrets,  # Add secrets here so HF_TOKEN is available during image build
    )
)

# Lastly, copy the ComfyUI workflow JSON to the container.
# image = image.add_local_file(
#     Path(__file__).parent / "workflow_api.json", "/root/workflow_api.json"
# )


# ## Running ComfyUI interactively

# Spin up an interactive ComfyUI server by wrapping the `comfy launch` command in a Modal Function
# and serving it as a [web server](https://modal.com/docs/guide/webhooks#non-asgi-web-servers).

app = modal.App(name="example-comfyui", image=image)


@app.function(
    max_containers=1,  # limit interactive session to 1 container
    gpu="L40S",  # good starter GPU for inference
    volumes={"/cache": vol},  # mounts our cached models
    secrets=secrets,  # Add secrets here so HF_TOKEN is available during runtime
)
@modal.concurrent(
    max_inputs=10
)  # required for UI startup process which runs several API calls concurrently
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
