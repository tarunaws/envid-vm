FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends \
        python3 \
        python3-pip \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY microservices/text-on-video/code/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements.txt

COPY microservices/text-on-video/code/text-on-video.py /app/text-on-video.py

ENV PORT=5083
EXPOSE 5083

CMD ["python3", "-m", "flask", "--app", "/app/text-on-video.py", "run", "--host", "0.0.0.0", "--port", "5083"]
