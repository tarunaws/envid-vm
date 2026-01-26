FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY microservices/text-on-video/code/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY microservices/text-on-video/code/text-on-video.py /app/text-on-video.py

ENV PORT=5083
EXPOSE 5083

CMD ["python", "-m", "flask", "--app", "/app/text-on-video.py", "run", "--host", "0.0.0.0", "--port", "5083"]
