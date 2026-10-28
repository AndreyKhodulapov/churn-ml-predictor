VENV := .venv
BIN := $(VENV)/bin
IMAGE := churn-ml-predictor
# Named volume for the trained model and the training history, so they outlive the container.
MODELS_VOLUME := churn-models

.PHONY: install run test docker-build docker-run

install:
	python3 -m venv $(VENV)
	$(BIN)/pip install -r requirements-dev.txt

run:
	$(BIN)/uvicorn src.main:app --reload

test:
	$(BIN)/pytest

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --name $(IMAGE) -p 8000:8000 -v $(MODELS_VOLUME):/app/models $(IMAGE)
