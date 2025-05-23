# How to use the server?

I tried to make the simples possible server to use. You can find that there's not a lot of thing hard-coded to the server, you need to send the desired background video URL and person image to the server, and it will do the rest.
The language selection was also simplified: just send the selected kokoro voice and the server will figure out which language it is.

## REST API

The available endpoints are

- GET `/health`
- GET `/api/videos`
- POST `/api/videos`
- GET `/api/videos/:id`
- GET `/api/videos/:id/status`
- DELETE `/api/videos/:id`
- GET `/api/queue`

## Parameters to use for the create video endpoint

Required parameters:

- `text`: The text to be converted to speech
- `person_image_url`: The URL of the image of the person to be used in the video
- `bg_video_url`: The URL of the background video to be used in the video
- `person_name`: The name of the person to be used in the video

Optional parameters:

- `voice`: The voice to be used in the video. If not provided, the server will use english language by default, and pick a random voice.

## MCP server

The SSE endpoint can be reached at `/mcp/sse` and the message endpoint at `/mcp/messages`.

There are two tools available

- `create_video_mcp`
- `list_languages_mcp`
