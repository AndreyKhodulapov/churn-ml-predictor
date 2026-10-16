VENV := .venv
BIN := $(VENV)/bin

.PHONY: install run test

install:
	python3 -m venv $(VENV)
	$(BIN)/pip install -r requirements-dev.txt

run:
	$(BIN)/uvicorn src.main:app --reload

test:
	$(BIN)/pytest
