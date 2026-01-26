FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY microservices/frame-extractor/code/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY microservices/frame-extractor/code/frame-extractor.py /app/frame-extractor.py

ENV PORT=5093
EXPOSE 5093

CMD ["python", "-m", "flask", "--app", "/app/frame-extractor.py", "run", "--host", "0.0.0.0", "--port", "5093"]
