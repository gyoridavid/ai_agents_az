# Episode 24: generate images with Qwen Image, Flux.1 [dev] and Flux.1 Schnell with modal.com and Cloudflare Workers AI

In this video, we show you how to generate almost 30,000 AI images per month (or 1K per day) completely free for your automations. We use Cloudflare Workers AI to generate images using Flux Schnell, and [Modal.com](http://Modal.com) to generate images with Flux.1 [dev] and Qwen Image.

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

## Free n8n JSON workflows

- [n8n subworkflow to run Flux.1 [dev] on Cloudflare Workers AI](n8n_cloudflare_flux_schnell.json)
- [n8n subworkflow to run FLUX.1 [dev] on modal.com](n8n_modal_flux_dev.json)
- [n8n subworkflow to run FLUX.1 Schnell on modal.com](n8n_modal_flux_schnell.json)
- [n8n subworkflow to run Qwen Image on modal.com](n8n_modal_qwen.json)
- [n8n workflow to create TikTok scary story videos](n8n_tiktok_scary.json)

## Modal applications

- [python modal application to run FLUX.1 [dev]](modal_nunchaku_flux_dev.py)
- [python modal application to run FLUX.1 Schnell](modal_nunchaku_flux_schnell.py)
- [python modal application to run Qwen Image](modal_nunchaku_qwen.py)

## Instructions

1. Install python3 and venv
2. Create a new virtual environment with `python3 -m venv .venv`
3. Activate the virtual environment with `source .venv/bin/activate`
4. Install modal with `pip install modal`
5. Download the modal python applications and put it in the same directory
6. Run `modal deploy <<YOUR_APP>>` to deploy the application
7. Copy the URL of the deployed application
8. Import and configure the subworkflows in n8n

## Additional resources

- [Join n8n](https://n8n.partnerlinks.io/fenoo5ekqs1g)
- [Modal.com](https://modal.com)

## Watch the video

[![100% FREE AI image generation with n8n (Flux and Qwen)
](https://img.youtube.com/vi/2tycZNP5_IA/0.jpg)](https://www.youtube.com/watch?v=2tycZNP5_IA)
