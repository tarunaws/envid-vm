FROM mysql:8.0

ENV MYSQL_ROOT_PASSWORD=admin \
	MYSQL_DATABASE=envid_metadata \
	MYSQL_USER=admin \
	MYSQL_PASSWORD=admin

COPY microservices/db/code/init/ /docker-entrypoint-initdb.d/
