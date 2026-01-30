FROM tarunaws/ai:audio-transcription

WORKDIR /app

COPY microservices/audio-transcription/code/openai_whisper_service.py /app/openai_whisper_service.py
COPY microservices/audio-transcription/code/openai_whisper_adapter.py /app/openai_whisper_adapter.py
COPY microservices/audio-transcription/code/writers.py /app/writers.py

ENV PYTHONPATH=/app

EXPOSE 5088
CMD ["python3", "-m", "uvicorn", "openai_whisper_service:app", "--host", "0.0.0.0", "--port", "5088"]
