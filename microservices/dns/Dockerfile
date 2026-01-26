FROM alpine:3.20

RUN apk add --no-cache dnsmasq

COPY microservices/dns/code/hosts /etc/hosts.docker

EXPOSE 53/udp 53/tcp

CMD ["dnsmasq", "-k", "--log-facility=-", "--no-resolv", "--domain-needed", "--addn-hosts=/etc/hosts.docker", "--server=1.1.1.1", "--server=8.8.8.8"]
