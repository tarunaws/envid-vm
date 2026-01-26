FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV ONNXRUNTIME_EXECUTION_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-turbo8 \
    zlib1g \
    libxcb1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY microservices/moderation/code/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY microservices/moderation/code/localModerationNudeNet.py /app/localModerationNudeNet.py

ENV PORT=5081
EXPOSE 5081

CMD ["python", "-m", "flask", "--app", "/app/localModerationNudeNet.py", "run", "--host", "0.0.0.0", "--port", "5081"]
