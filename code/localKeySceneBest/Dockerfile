FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ffmpeg + ffprobe for TransNetV2 video decoding
RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU wheels first (large)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

ENV PORT=5085
EXPOSE 5085

CMD ["python", "-m", "flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "5085"]
