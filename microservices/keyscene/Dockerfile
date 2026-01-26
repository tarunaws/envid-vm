FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive \
  TZ=Etc/UTC

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY microservices/keyscene/code/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY microservices/keyscene/code/localKeySceneBest.py /app/localKeySceneBest.py

ENV PORT=5085
EXPOSE 5085

CMD ["python", "-m", "flask", "--app", "localKeySceneBest", "run", "--host", "0.0.0.0", "--port", "5085"]
