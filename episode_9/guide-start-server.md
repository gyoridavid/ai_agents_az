# How to run the server

## Windows

0. Install `Docker Desktop` from the Windows Store
1. Install `wsl2` if it's not already installed
2. Start `Docker Desktop`
3. Click on `Terminal` in the bottom right corner
4. Enter the following command to start the server

```bash
docker run -it --rm --name narrated-story-creator -p 8000:8000 gyoridavid/narrated-story-creator:latest
```

Now, you can open [http://localhost:8000/health](http://localhost:8000/health) as a smoke-test. 
You can use `http://localhost:8000` as the server configuration inside n8n, however, if you are running n8n in a Docker container you need to use the `http://host.docker.internal:8000` server address.

## MacOS, Linux

Open a terminal, and run the docker command

```bash
docker run -it --rm --name narrated-story-creator -p 8000:8000 gyoridavid/narrated-story-creator:latest
```

Now, you can open [http://localhost:8000/health](http://localhost:8000/health) as a smoke-test. 
You can use `http://localhost:8000` as the server configuration inside n8n, however, if you are running n8n in a Docker container you need to use the `http://host.docker.internal:8000` server address.

## Nvidia / Cuda support

If you have an NVidia GPU and the CUDA drivers are installed, you can unlock full GPU support for the whole video generation.
Start the container with the following command:

```bash
docker run --rm --gpus=all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -p 8000:8000 -it gyoridavid/narrated-story-creator:latest-cuda
```

Now, you can open [http://localhost:8000/health](http://localhost:8000/health) as a smoke-test. 
You can use `http://localhost:8000` as the server configuration inside n8n, however, if you are running n8n in a Docker container you need to use the `http://host.docker.internal:8000` server address.

## What if I'm runnign n8n in the cloud?

1. If you can run the server on the same VPS, you can just use either `http://localhost:8000` or `http://host.docker.internal:8000` depending on your situation.
2. If you are running n8n cloud, you should use the address of the VPS - and the right port. Server management is out of the scope if this document.
3. You can use [localtunnel](https://theboroer.github.io/localtunnel-www/) to proxy your locally running server to the cloud.
