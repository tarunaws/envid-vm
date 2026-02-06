FROM tarunaws/ai:audio-transcription-squashed

WORKDIR /app

COPY microservices/audio-transcription/code /app
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt
RUN python3 -m pip uninstall -y openai-whisper || true
RUN rm -rf /root/.cache/whisper /root/.cache/torch/whisper

ENV PYTHONPATH=/app

EXPOSE 5088
CMD ["python3", "-m", "uvicorn", "openai_whisper_service:app", "--host", "0.0.0.0", "--port", "5088"]
