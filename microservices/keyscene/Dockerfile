FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ENV DEBIAN_FRONTEND=noninteractive \
  TZ=Etc/UTC


WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY microservices/keyscene/code/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY microservices/keyscene/code/localKeySceneBest.py /app/localKeySceneBest.py
COPY microservices/keyscene/code/transnetv2_local.py /app/transnetv2_local.py
COPY microservices/keyscene/code/transnetv2_pytorch_local.py /app/transnetv2_pytorch_local.py

ENV PORT=5085
EXPOSE 5085

CMD ["python", "-m", "flask", "--app", "localKeySceneBest", "run", "--host", "0.0.0.0", "--port", "5085"]
