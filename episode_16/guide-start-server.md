# How to start the server

## Requirements

- Docker (or [Docker Desktop](https://docs.docker.com/desktop/)) installed on your machine
- at least 4vCPUs and at least 8GB RAM

## Run the server locally

1. Open terminal on mac/linux, or Docker Desktop terminal (bottom-right corner) on Windows
2. Run the following command to start the server:

   ```bash
        docker run --rm -p 8000:8000 -it gyoridavid/ai-agents-no-code-tools:latest
   ```

3. If you have an NVidia GPU and have the [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit) installed, you can run the server with GPU support:

   ```bash
        docker run --rm --gpus=all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -p 8000:8000 -it gyoridavid/ai-agents-no-code-tools:latest-cuda
   ```

## Deploy the server to the cloud

We recommend using Coolify to deploy the server to the cloud. Use the Docker image `gyoridavid/ai-agents-no-code-tools:latest` to deploy the server, and map port `8000` to the host.

We recommend using [Runpod Pods](https://runpod.io?ref=2zyvy637) for GPU instances.

![alt text](coolify.png)

## How to find the server URL for n8n

|                                                   | n8n is running locally, using `n8n start`                                                       | n8n is running locally using                                                                                                                                                                            | n8n is running in the cloud                                                                      |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `AI Agents No-Code Tools` is running locally      | `http://localhost:8000`                                                                         | You can either use `http://host.docker.internal:8000` configure the server's docker image to share the network with n8n, and use the name of the image as the url `http://ai-agents-no-code-tools:8000` | ⛔ won’t work - deploy `AI Agents No-Code Tools` to the cloud                                    |
| `AI Agents No-Code Tools` is running in the cloud | You should use your IP address `http://{YOUR_IP}:8000` or the domain you deployed the server to | You should use your IP address `http://{YOUR_IP}:8000`                                                                                                                                                  | You should use your IP address `http://{YOUR_IP}:8000` or the domain you deployed the service to |
