FROM tarunaws/ai:transcoder

ENV ENVID_FFMPEG_USE_GPU=
ENV ENVID_FFMPEG_REQUIRE_GPU=
ENV ENVID_FFMPEG_GPU_PRESET=fast

WORKDIR /app

COPY microservices/transcoder/code/ffmpeg.py /app/ffmpeg.py

EXPOSE 5091

CMD ["python3", "/app/ffmpeg.py"]
