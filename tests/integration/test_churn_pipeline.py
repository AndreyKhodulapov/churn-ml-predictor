"""The whole churn pipeline through TestClient: the app starts as uvicorn starts it, reads a CSV from disk,
trains, reports the status and predicts.

Unlike the httpx client of the other integration tests, TestClient used as a context manager runs the lifespan,
so the dataset, the saved model and the history are loaded by the startup itself. The dataset is a fixed sample of
150 rows of the bundled one, stratified by churn: small enough to train in milliseconds, and the same on every run.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.core.openapi_examples import (
    DATASET_UNAVAILABLE_ERROR,
    INVALID_C_ERROR,
    LOYAL_CUSTOMER,
    MODEL_NOT_TRAINED_ERROR,
    RISKY_CUSTOMER,
)
from src.main import app
from src.services.dataset import get_dataset
from src.services.model_storage import get_model_store
from src.services.preprocessing import get_split
from src.services.training_history import get_training_history

SAMPLE_PATH = Path(__file__).parent.parent / "data" / "churn_sample.csv"


@pytest.fixture
def use_dataset(monkeypatch) -> Iterator[Callable[[Path], None]]:
    """Make the service read the dataset from a path, the startup and every endpoint alike."""

    def use(path: Path) -> None:
        monkeypatch.setattr("src.services.dataset.DATASET_PATH", path)
        get_dataset.cache_clear()
        get_split.cache_clear()

    yield use
    # Drop the dataset and the split of the path so later tests get the default ones again.
    get_dataset.cache_clear()
    get_split.cache_clear()


@pytest.fixture
def start_app() -> Callable[[], TestClient]:
    """Start the app as a fresh process would: with empty stores filled by the lifespan from the saved files."""

    def start() -> TestClient:
        get_model_store.cache_clear()
        get_training_history.cache_clear()
        return TestClient(app)

    return start


@pytest.fixture
def pipeline_client(use_dataset, start_app) -> Iterator[TestClient]:
    use_dataset(SAMPLE_PATH)
    with start_app() as client:
        yield client


def test_sample_is_read_from_disk_at_startup(pipeline_client):
    info = pipeline_client.get("/dataset/info").json()
    split = pipeline_client.get("/dataset/split-info").json()

    assert info["row_count"] == 150
    assert {label: part["count"] for label, part in info["target_distribution"].items()} == {"0": 120, "1": 30}
    assert split["train_size"] == 120
    assert split["test_size"] == 30
    # The test part keeps the churn share, so the metrics have churners to score.
    assert split["test_distribution"]["1"]["count"] == 6


def test_train_status_predict(pipeline_client):
    assert pipeline_client.get("/model/status").json()["trained"] is False
    assert pipeline_client.post("/predict", json=LOYAL_CUSTOMER).status_code == 503

    trained = pipeline_client.post("/model/train")

    assert trained.status_code == 200
    metrics = trained.json()
    assert metrics.keys() == {"accuracy", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())

    status = pipeline_client.get("/model/status").json()

    assert status["trained"] is True
    assert status["model_type"] == "logreg"
    assert status["metrics"] == metrics

    predicted = pipeline_client.post("/predict", json=[LOYAL_CUSTOMER, RISKY_CUSTOMER])

    assert predicted.status_code == 200
    loyal, risky = predicted.json()
    for prediction in (loyal, risky):
        assert 0 <= prediction["probability_churn"] <= 1
        assert prediction["probability_no_churn"] + prediction["probability_churn"] == pytest.approx(1)
        assert prediction["churn"] == int(prediction["probability_churn"] > 0.5)
    assert risky["probability_churn"] > loyal["probability_churn"]


def test_pipeline_is_reproducible(pipeline_client):
    config = {"model_type": "random_forest", "hyperparameters": {"n_estimators": 20}}
    customers = [LOYAL_CUSTOMER, RISKY_CUSTOMER]

    first = pipeline_client.post("/model/train", json=config).json()
    first_predictions = pipeline_client.post("/predict", json=customers).json()
    second = pipeline_client.post("/model/train", json=config).json()
    second_predictions = pipeline_client.post("/predict", json=customers).json()

    assert second == first
    assert second_predictions == first_predictions


def test_restarted_app_predicts_with_the_saved_model(pipeline_client, start_app):
    pipeline_client.post("/model/train")
    status = pipeline_client.get("/model/status").json()
    predictions = pipeline_client.post("/predict", json=[LOYAL_CUSTOMER, RISKY_CUSTOMER]).json()

    with start_app() as restarted:
        assert restarted.get("/model/status").json() == status
        assert restarted.post("/predict", json=[LOYAL_CUSTOMER, RISKY_CUSTOMER]).json() == predictions
        assert len(restarted.get("/model/metrics").json()["history"]) == 1


def test_failed_training_leaves_the_service_without_model(pipeline_client):
    response = pipeline_client.post("/model/train", json={"hyperparameters": {"C": 0}})

    assert response.status_code == 422
    assert response.json() == INVALID_C_ERROR
    assert pipeline_client.get("/model/status").json()["trained"] is False
    assert pipeline_client.post("/predict", json=LOYAL_CUSTOMER).json() == MODEL_NOT_TRAINED_ERROR


def test_app_without_dataset_file_starts_and_answers_503(use_dataset, start_app, tmp_path, caplog):
    use_dataset(tmp_path / "missing.csv")

    with caplog.at_level(logging.WARNING), start_app() as client:
        trained = client.post("/model/train")
        predicted = client.post("/predict", json=LOYAL_CUSTOMER)

    assert "Starting without a usable dataset: Dataset file not found" in caplog.text
    assert trained.status_code == 503
    assert trained.json() == DATASET_UNAVAILABLE_ERROR
    assert predicted.status_code == 503
    assert predicted.json() == MODEL_NOT_TRAINED_ERROR


def test_app_with_single_class_dataset_answers_409(use_dataset, start_app, tmp_path):
    path = tmp_path / "loyal_only.csv"
    sample = pd.read_csv(SAMPLE_PATH)
    sample[sample["churn"] == 0].to_csv(path, index=False)
    use_dataset(path)

    with start_app() as client:
        response = client.post("/model/train")
        status = client.get("/model/status").json()

    assert response.status_code == 409
    assert response.json() == {
        "code": "dataset_not_trainable",
        "message": "Dataset cannot be used for training: Dataset has only churn class 0, a classifier needs both",
        "details": None,
    }
    assert status["trained"] is False
