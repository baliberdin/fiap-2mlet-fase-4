run-all:
	@docker compose up -d

build-api:
	@docker build -t stock-predictions-api ./stock-predictions-api

build-client-app:
	@docker build -t client-app ./client-app

stop-api:
	@docker compose down stock-predictions-api

start-api:
	@docker compose up -d stock-predictions-api

rebuild-api: build-api stop-api start-api

run-cross-validation-training:
# @export TRAINING_SLICES=5 && export TRAINING_PAGE=0 && python model/training.py
	@export TRAINING_SLICES=5 && export TRAINING_PAGE=0 && python model/cross_validation_training.py
