FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ENVID_WHISPERX_DEVICE=cuda \
    ENVID_WHISPERX_COMPUTE_TYPE=float16
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY microservices/audio-translation/code/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements.txt

COPY microservices/audio-translation/code/whisperxService.py /app/whisperxService.py
COPY microservices/audio-translation/code/whisperx_adapter.py /app/whisperx_adapter.py
COPY microservices/audio-translation/code/writers.py /app/writers.py
COPY microservices/audio-translation/code/whisperx_batch.py /app/whisperx_batch.py

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV ENVID_WHISPERX_VAD_METHOD=silero

EXPOSE 5088
CMD ["python3", "-m", "uvicorn", "whisperxService:app", "--host", "0.0.0.0", "--port", "5088"]
