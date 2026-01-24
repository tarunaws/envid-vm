FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        build-essential \
        git \
        docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY code/requirements.txt /app/code/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/code/requirements.txt

COPY . /app

ENV PYTHONPATH="/app:/app/code"

RUN chmod +x /app/code/envidMetadataGCP/ffmpeg_proxy.sh /app/code/envidMetadataGCP/ffprobe_proxy.sh

EXPOSE 5016

CMD ["python", "code/envidMetadataGCP/app.py"]
