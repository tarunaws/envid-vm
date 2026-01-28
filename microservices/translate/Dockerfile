FROM tarunaws/ai:translate

USER root

COPY microservices/translate/code/argos-install-all.py /usr/local/bin/argos-install-all.py
COPY microservices/translate/code/supervisord.conf /etc/supervisor/supervisord.conf
COPY microservices/text-normalizer/code/text-normalizer.py /app/text-normalizer.py

USER 1000

EXPOSE 5000 8010 5098

ENTRYPOINT ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
