FROM tarunaws/ai:translate

USER root

RUN (getent group 1000 >/dev/null || groupadd -g 1000 appuser) \
	&& (id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 -g 1000 appuser)

COPY microservices/translate/code/argos-install-all.py /usr/local/bin/argos-install-all.py
COPY microservices/translate/code/supervisord.conf /etc/supervisor/supervisord.conf
COPY microservices/translate/code/requirements.txt /tmp/translate-requirements.txt
COPY microservices/translate/code/indictrans2_service.py /app/indictrans2_service.py
COPY microservices/text-normalizer/code/text-normalizer.py /app/text-normalizer.py

RUN /app/venv/bin/pip install --no-cache-dir -r /tmp/translate-requirements.txt

USER 1000

EXPOSE 5000 8010 5098 5102

ENTRYPOINT ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
