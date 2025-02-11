run-all:
	@docker compose up -d

build-api:
	@docker build -t stock-predictions-api ./stock-predictions-api

stop-api:
	@docker compose down stock-predictions-api

start-api:
	@docker compose up -d stock-predictions-api

rebuild-api: build-api stop-api start-api
