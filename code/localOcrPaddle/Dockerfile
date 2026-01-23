FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PORT=5084

ENV DEBIAN_FRONTEND=noninteractive \
  TZ=Etc/UTC

# Minimal runtime deps often needed by paddle/opencv
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 5084
CMD ["python", "/app/app.py"]
