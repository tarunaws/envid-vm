FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY microservices/synopsys/code/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY microservices/synopsys/code/synopsys.py /app/synopsys.py

EXPOSE 5095

CMD ["python", "/app/synopsys.py"]
