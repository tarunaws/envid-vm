up:
	@docker compose up -d

down:
	@docker compose down

logs:
	@docker compose logs -f --tail=200

health:
	@curl -fsS http://localhost:8080/languages | head -c 200

models:
	@bash scripts/download-models.sh
