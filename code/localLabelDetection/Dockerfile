FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

# System deps:
# - libgl/libglib: required by opencv-python-headless in some environments
RUN set -eux; \
    for i in 1 2 3; do \
        apt-get update -o Acquire::Retries=5 && \
        apt-get install -o Acquire::Retries=5 -y --no-install-recommends \
            libglib2.0-0 \
            libgl1 \
            build-essential \
            git \
            ninja-build \
        && rm -rf /var/lib/apt/lists/* \
        && break; \
        echo "apt-get failed (attempt $i); retrying in 5s..."; \
        sleep 5; \
    done

# Base deps
COPY requirements.txt /app/requirements.txt

# Torch (CPU) first for stable dependency resolution.
# Pin versions so OpenMMLab wheels resolve consistently.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    torch==2.1.2 torchvision==0.16.2

RUN pip install --no-cache-dir -r /app/requirements.txt

# Engines (Linux-friendly installs)
#
# MMDetection is the primary motivation for Docker on macOS: ensure it is actually installed.
# We intentionally install mmcv from OpenMMLab's prebuilt wheel index to avoid source builds.
# Note: On Apple Silicon, you typically want to build/run this image as linux/amd64
# (via docker-compose.amd64.yml) to get prebuilt mmcv wheels with compiled ops.
RUN pip install --no-cache-dir \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html \
    'mmcv==2.1.0' \
 && pip install --no-cache-dir 'mmdet==3.3.0' \
 && python -c "import mmcv, mmdet; from mmcv.ops import nms" \
 && pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' \
 && python -c "import detectron2" \
 \
 # Some compiled ML wheels are not yet NumPy 2 compatible.
 # Enforce numpy<2 after installing optional engines, in case a dependency upgraded it.
 && pip install --no-cache-dir 'numpy>=1.26,<2'

COPY app.py /app/app.py

ENV PORT=5083
EXPOSE 5083

CMD ["python", "-m", "flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "5083"]
