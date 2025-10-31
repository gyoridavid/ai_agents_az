# Episode 35: Instagram influencer machine

Create your own AI influencers with this free n8n workflow. It uses fal.ai's Veo 3.1 and nanobanana models to generate images and captions, then posts them to Instagram.

## [📚 Join our Skool community for support, premium content and more!](https://www.skool.com/ai-agents-az/about?gw11)

### Get the premium versions of the workflows and the exclusive content - with the hosted GPU media server

## Free n8n JSON workflow

- [n8n workflow: influencer machine](workflow-influencer-machine.json)
- [n8n subWorkflow: fal.ai Veo 3.1](subworkflow-fal-veo31.json)
- [n8n subWorkflow: fal.ai nanobanana](subworkflow-fal-nanobanana.json)

## Instructions

### Create these datatables in your n8n database

- influencer
  - name (string)
  - bio (string)
  - image (string)
  - instagram_business_id (string)
- influencer_weekly_plans
  - influencer_id (string)
  - week (string)
  - plan (string)
- influencer_posts
  - influencer_id (string)
  - post_summary (string)

## Additional resources

- [Guide to connect your Instagram Business account to n8n](guide-instagram.md)
- [Join n8n](https://n8n.partnerlinks.io/fenoo5ekqs1g)
- [Fal.ai API keys](https://fal.ai/dashboard/keys)

## Watch the video

[![Make your own AI influencers with this free n8n workflow](https://img.youtube.com/vi/PjXYr6M4fjY/0.jpg)](https://www.youtube.com/watch?v=PjXYr6M4fjY)
