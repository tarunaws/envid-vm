FROM tarunaws/ai:moderation

WORKDIR /app

COPY microservices/moderation/code/localModerationNudeNet.py /app/localModerationNudeNet.py

ENV PORT=5081
EXPOSE 5081

CMD ["python", "-m", "flask", "--app", "/app/localModerationNudeNet.py", "run", "--host", "0.0.0.0", "--port", "5081"]
