FROM tarunaws/ai:moderation

WORKDIR /app

# Note: Base image already includes runtime deps; skip apt-get to avoid build layer errors.

COPY microservices/moderation/code/requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY microservices/moderation/code/localModerationNudeNet.py /app/localModerationNudeNet.py

ENV PORT=5081
EXPOSE 5081

CMD ["python", "-m", "flask", "--app", "/app/localModerationNudeNet.py", "run", "--host", "0.0.0.0", "--port", "5081"]
