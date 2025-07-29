# Episode 19: Run FLUX.1 Kontext [dev] with modal.com

In this video, we show you how to use Modal’s serverless GPU infrastructure with n8n to do image generation with Flux Kontext Dev for free.

## [📚 Join our Skool community for support, premium content and more!](https://www.skool.com/ai-agents-az/about)

### Be part of a growing community and help us create more content like this

## Free n8n JSON workflow

- [n8n workflow to run FLUX.1 Kontext [dev] on modal.com](flux_kontext_dev_modal.json)

## Modal application

- [python modal application to run FLUX.1 Kontext [dev]](flux_kontext_dev_modal.py)

## Instructions

1. Install python3 and venv
2. Create a new virtual environment with `python3 -m venv .venv`
3. Activate the virtual environment with `source .venv/bin/activate`
4. Install modal with `pip install modal`
5. Download the [python application](flux_kontext_dev_modal.py) and put it in the same directory
6. Run `modal deploy flux_kontext_dev_modal.py` to deploy the application
7. Copy the URL of the deployed application
8. Add the URL to your n8n workflow in the `Configure me` node and run the workflow

## Additional resources

- [Join n8n](https://n8n.partnerlinks.io/fenoo5ekqs1g)
- [Modal.com](https://modal.com)

## Watch the video

[![Run Flux Kontext 100% FREE without having a GPU - n8n automation](https://img.youtube.com/vi/ndMi2Mo7znA/0.jpg)](https://www.youtube.com/watch?v=ndMi2Mo7znA)
