FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY code/whisperxService/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install whisperx \
    && python3 -m pip install -r /app/requirements.txt

COPY code/whisperxService /app
COPY code/whisperx_adapter.py /app/whisperx_adapter.py
COPY code/writers.py /app/writers.py
COPY code/whisperx_batch.py /app/whisperx_batch.py

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV ENVID_WHISPERX_VAD_METHOD=silero

EXPOSE 5088
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5088"]
