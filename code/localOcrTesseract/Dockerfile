FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     tesseract-ocr \
     tesseract-ocr-eng \
     tesseract-ocr-hin \
  && rm -rf /var/lib/apt/lists/*

CMD ["tesseract", "--version"]
