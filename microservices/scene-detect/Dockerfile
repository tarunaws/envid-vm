FROM tarunaws/ai:scene-detect

WORKDIR /app

COPY microservices/scene-detect/code/scene-detect.py /app/scene-detect.py

EXPOSE 5094

CMD ["python", "/app/scene-detect.py"]
