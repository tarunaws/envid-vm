FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1

ENV ENVID_FFMPEG_USE_GPU=true \
	ENVID_FFMPEG_GPU_PRESET=fast
ENV NVIDIA_VISIBLE_DEVICES=all \
	NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY microservices/transcoder/code/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
	&& python -m pip install -r /app/requirements.txt

COPY microservices/transcoder/code/ffmpeg.py /app/ffmpeg.py

EXPOSE 5091

CMD ["python", "/app/ffmpeg.py"]
