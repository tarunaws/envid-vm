FROM tarunaws/ai:moderation

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ffmpeg \
		libgl1 \
		libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

COPY microservices/moderation/code/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
	&& python -m pip install -r /app/requirements.txt

COPY microservices/moderation/code/localModerationNudeNet.py /app/localModerationNudeNet.py

ENV PORT=5081
EXPOSE 5081

CMD ["python", "-m", "flask", "--app", "/app/localModerationNudeNet.py", "run", "--host", "0.0.0.0", "--port", "5081"]
