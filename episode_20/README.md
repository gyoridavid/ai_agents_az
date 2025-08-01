# Episode 20: Use Wan 2.2, ComfyUI and n8n to generate videos for free

In this video, we show you how you can run Wan 2.2 from Alibaba on modal.com using ComfyUI and n8n. You can use your $30 free credits you receive every month to generate stunning AI videos.

<table>
  <tr>
    <td>
      <a href="https://www.skool.com/ai-agents-az/about">📚 Join our Skool community for support, premium content and more!</a>
      <p>Be part of a growing community and help us create more content like this</p>
    </td>
    <td>
      <img width="548" height="596" alt="Screenshot 2025-07-29 at 3 23 33 PM" src="https://github.com/user-attachments/assets/d687b58d-92d0-44c0-93f8-7be23d3cb80c" />
    </td>
  </tr>
</table>

## Watch the video

[![Wan 2.2 for FREE (NO GPU NEEDED) - Best VEO3 and Seedance alternative (ComfyUI + n8n workflow)](https://img.youtube.com/vi/rZ45_IhojLY/0.jpg)](https://www.youtube.com/watch?v=rZ45_IhojLY)

## Free n8n JSON workflows

- [Wan 2.2 5b Image to Video n8n workflow](n8n_wan_2.2_5b_i2v.json)
- [Wan 2.2 5b Text to Video n8n workflow](n8n_wan_2.2_5b_t2v.json)
- [Wan 2.2 14b Image to Video n8n workflow](n8n_wan_2.2_14b_i2v.json)
- [Wan 2.2 14b Text to Video n8n workflow](n8n_wan_2.2_14b_t2v.json)

## Modal application

- [python modal application to run Wan 2.2 5b with ComfyUI](modal_wan_comfyui_5b.py)
- [python modal application to run Wan 2.2 14b Text to Video with ComfyUI](modal_wan_comfyui_14b_t2v.py)
- [python modal application to run Wan 2.2 14b Image to Video with ComfyUI](modal_wan_comfyui_14b_i2v.py)

## Instructions

### Pre-requisites

- Watch the video!
- Have a Modal account (you can sign up for free)
- Have an n8n setup

#### Windows setup

- Install Python 3.10+ from the [official website](https://www.python.org/downloads/)
- Install the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install) and set up a Linux distribution (like Ubuntu)
- Install Python venv in WSL with `sudo apt install python3-venv`

#### MacOS + Linux setup

- Make sure you have Python 3.10+ installed
- Make sure you have Python venv installed

### Steps to run the application

1. Copy the modal file you want to run (e.g., `modal_wan_comfyui_5b.py`) to your local machine
2. Open a terminal and navigate to the directory where you copied the file
3. Create a virtual environment with `python3 -m venv .venv`
4. Activate the virtual environment with `source .venv/bin/activate`
5. Install the modal package with `pip install modal`
6. Run `modal setup` to configure your Modal account and approve it on the Modal website
7. Run `modal serve {THE_MODAL_PYTHON_FILE}` to start the Modal application (replace `{THE_MODAL_PYTHON_FILE}` with the name of the file you copied, e.g., `modal_wan_comfyui_5b.py`)
8. (optional) Deploy the application with `modal deploy {THE_MODAL_PYTHON_FILE}`

## Additional resources

- [Join n8n](https://n8n.partnerlinks.io/fenoo5ekqs1g)
- [Modal.com](https://modal.com)
- [ComfyUI documentation](https://docs.comfy.org/)

## ComfyUI official workflows from the [ComfyUI documentation](https://docs.comfy.org/tutorials/video/wan/wan2_2)

- [Wan 2.2 5b ti2v ComfyUI workflow](comfyui_wan2_2_5B_ti2v.json)
- [Wan 2.2 14b t2v ComfyUI workflow](comfyui_wan2_2_14B_t2v.json)
- [Wan 2.2 14b i2v ComfyUI workflow](comfyui_wan2_2_14B_i2v.json)

## ComfyUI API workflows

- [Wan 2.2 5b t2v ComfyUI API workflow](comfyui_api_wan2_2_5B_t2v.json)
- [Wan 2.2 5b i2v ComfyUI API workflow](comfyui_api_wan2_2_5B_i2v.json)
- [Wan 2.2 14b t2v ComfyUI API workflow](comfyui_api_wan2_2_14B_t2v.json)
- [Wan 2.2 14b i2v ComfyUI API workflow](comfyui_api_wan2_2_14B_i2v.json)
