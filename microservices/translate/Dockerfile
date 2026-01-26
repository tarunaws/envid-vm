FROM libretranslate/libretranslate:v1.6.0

ARG LT_VERSION=6.4

USER root

ENV LT_DISABLE_WEB_UI=true \
    LT_UPDATE_MODELS=true \
    LT_WORKERS=2 \
    LT_BATCH_ENABLED=true \
    LT_BATCH_SIZE=10
ENV HOME=/home/libretranslate \
    XDG_DATA_HOME=/home/libretranslate/.local/share \
    XDG_CACHE_HOME=/home/libretranslate/.local/cache \
    ARGOS_TRANSLATE_DATA_DIR=/home/libretranslate/.local/share/argos-translate

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        supervisor \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/libretranslate/.local/share/argos-translate /home/libretranslate/.local/cache \
    && chown -R 1000:1000 /home/libretranslate/.local

RUN wget -O /tmp/languagetool.zip https://languagetool.org/download/LanguageTool-${LT_VERSION}.zip \
    && unzip /tmp/languagetool.zip -d /opt \
    && mv /opt/LanguageTool-${LT_VERSION} /opt/languagetool \
    && rm /tmp/languagetool.zip \
    && chown -R 1000:1000 /opt/languagetool

COPY microservices/translate/code/requirements.txt /app/requirements.txt
RUN /app/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

USER 1000

COPY microservices/translate/code/argos-install-all.py /usr/local/bin/argos-install-all.py
COPY microservices/translate/code/supervisord.conf /etc/supervisor/supervisord.conf

EXPOSE 5000 8010

ENTRYPOINT ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
