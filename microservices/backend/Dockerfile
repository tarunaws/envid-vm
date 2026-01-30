FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV ENVID_GRAMMAR_CORRECTION_URL=http://translate:8010 \
    ENVID_LIBRETRANSLATE_URL=http://translate:5000 \
    ENVID_METADATA_LOCAL_MODERATION_URL=http://moderation:5081 \
    ENVID_METADATA_LOCAL_KEYSCENE_URL=http://keyscene:5085 \
    ENVID_METADATA_LOCAL_KEYSCENE_HEALTH_TIMEOUT_SECONDS=10 \
    ENVID_STORAGE_BASE_DIR=/mnt/gcs/envid-metadata \
    ENVID_METADATA_VM_NAME=gpu-g2-vm \
    ENVID_TRANSCRIPT_NLP_CORRECTION_MODE=gemini \
    ENVID_DICTIONARY_ENABLE=true \
    ENVID_GRAMMAR_CORRECTION_USE_LOCAL= \
    ENVID_TRANSCRIPT_GRAMMAR_CORRECTION=true \
    ENVID_TRANSCRIPT_PUNCTUATION_ENHANCE=true \
    ENVID_TRANSCRIPT_SEGMENT_CORRECTION=true \
    ENVID_TRANSCRIPT_SEGMENT_CORRECTION_MODE= \
    ENVID_TRANSCRIPT_VERIFY_STRICT=true \
    ENVID_TRANSCRIPT_VERIFY_REQUIRE_CORRECTION=true \
    ENVID_TRANSCRIPT_VERIFY_MIN_WORDS=6 \
    ENVID_TRANSCRIPT_VERIFY_ALPHA_RATIO=0.20 \
    ENVID_TRANSCRIPT_VERIFY_HI_DEVANAGARI_RATIO=0.30 \
    ENVID_TRANSCRIPT_VERIFY_UNIQUE_RATIO=0.20 \
    ENVID_TRANSCRIPT_MIN_SIMILARITY=0.60 \
    ENVID_TRANSCRIPT_FORCE_PUNCTUATION_END=true \
    ENVID_WHISPER_LANGUAGE_CODE=hi \
    ENVID_WHISPER_MODEL_SIZE=large-v3 \
    ENVID_WHISPER_QUALITY=best \
    ENVID_WHISPER_TEMPERATURE=0.0 \
    ENVID_WHISPER_BEAM_SIZE=10 \
    ENVID_WHISPER_BEST_OF=10 \
    ENVID_WHISPER_CONDITION_ON_PREVIOUS_TEXT=true \
    ENVID_WHISPER_AUDIO_PREPROCESS= \
    ENVID_WHISPER_AUDIO_FILTERS= \
    ENVID_WHISPER_AUDIO_ATEMPO= \
    ENVID_WHISPER_FP16=true \
    ENVID_WHISPER_DEVICE=cuda

RUN apt-get -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update \
    && apt-get -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /app

COPY microservices/backend/code/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY microservices/backend/code/backend.py /app/backend.py

ENV PYTHONPATH="/app"

RUN if [ -f /app/ffmpeg_proxy.sh ]; then chmod +x /app/ffmpeg_proxy.sh; fi \
    && if [ -f /app/ffprobe_proxy.sh ]; then chmod +x /app/ffprobe_proxy.sh; fi

EXPOSE 5016

CMD ["python", "backend.py"]
