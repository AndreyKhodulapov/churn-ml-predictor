VENV := .venv
BIN := $(VENV)/bin
IMAGE := churn-ml-predictor
# Named volume for the trained model and the training history, so they outlive the container.
MODELS_VOLUME := churn-models
# Settings from .env, if it exists: uvicorn and docker run take the same flag.
ENV_FILE := $(if $(wildcard .env),--env-file .env)

.PHONY: install run test lint docker-build docker-run

install:
	python3 -m venv $(VENV)
	$(BIN)/pip install -r requirements-dev.txt

run:
	$(BIN)/uvicorn src.main:app --reload $(ENV_FILE)

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --name $(IMAGE) -p 8000:8000 -v $(MODELS_VOLUME):/app/models $(ENV_FILE) $(IMAGE)
