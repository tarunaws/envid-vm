FROM tarunaws/ai:frame-extractor

WORKDIR /app

COPY microservices/frame-extractor/code/frame-extractor.py /app/frame-extractor.py

ENV PORT=5093
EXPOSE 5093

CMD ["python", "-m", "flask", "--app", "/app/frame-extractor.py", "run", "--host", "0.0.0.0", "--port", "5093"]
