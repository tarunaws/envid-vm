FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     ffmpeg \
     tesseract-ocr \
     tesseract-ocr-eng \
     tesseract-ocr-hin \
     libgl1 \
     libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY code/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY code /app/code
COPY shared /app/shared

ENV PYTHONPATH="/app:/app/code"

WORKDIR /app/code/envidMetadataGCP

EXPOSE 5016

CMD ["python", "app.py"]
