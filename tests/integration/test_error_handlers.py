import logging

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.exceptions import DatasetLoadError, DatasetNotTrainableError
from src.main import app, lifespan
from src.services.dataset import get_dataset


@pytest.fixture
async def make_failing_client():
    clients = []

    async def make(error: Exception) -> AsyncClient:
        def broken_dataset():
            raise error

        app.dependency_overrides[get_dataset] = broken_dataset
        # The app re-raises unhandled exceptions after responding; keep them out of the test to inspect the response.
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        client = AsyncClient(transport=transport, base_url="http://test")
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.aclose()
    app.dependency_overrides.clear()


async def test_unhandled_error_returns_json_500(make_failing_client):
    client = await make_failing_client(RuntimeError("Broken state in /secret/path.csv"))

    response = await client.get("/dataset/preview")

    assert response.status_code == 500
    assert response.json() == {"code": "internal_error", "message": "Internal server error", "details": None}


async def test_unhandled_error_hides_exception_details(make_failing_client):
    client = await make_failing_client(RuntimeError("Broken state in /secret/path.csv"))

    response = await client.get("/dataset/preview")

    assert "/secret/path.csv" not in response.text


async def test_unknown_path_returns_json_404(client):
    response = await client.get("/no/such/path")

    assert response.status_code == 404
    assert response.json() == {"code": "not_found", "message": "Not Found", "details": None}


async def test_wrong_method_returns_json_405_with_allowed_methods(client):
    response = await client.get("/predict")

    assert response.status_code == 405
    assert response.json() == {"code": "method_not_allowed", "message": "Method Not Allowed", "details": None}
    assert response.headers["allow"] == "POST"


async def test_validation_error_lists_every_invalid_value(client):
    response = await client.get("/dataset/preview", params={"n": 0})

    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_error",
        "message": "Request failed validation, see details for every error",
        "details": [
            {
                "loc": ["query", "n"],
                "message": "Input should be greater than or equal to 1",
                "type": "greater_than_equal",
            }
        ],
    }


async def test_validation_error_does_not_echo_the_input(client, customer_features):
    customer_features["region"] = "antarctica"

    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 422
    assert "antarctica" not in response.text


async def test_invalid_json_returns_validation_error(client):
    response = await client.post("/predict", content=b"{bad", headers={"Content-Type": "application/json"})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert response.json()["details"][0]["type"] == "json_invalid"


async def test_unavailable_dataset_returns_json_503(make_failing_client, caplog):
    client = await make_failing_client(DatasetLoadError("Dataset file not found: /secret/path.csv"))

    with caplog.at_level(logging.WARNING):
        response = await client.get("/dataset/info")

    assert response.status_code == 503
    assert response.json() == {"code": "dataset_unavailable", "message": "Dataset is not available", "details": None}
    assert "/secret/path.csv" in caplog.text


async def test_unavailable_dataset_hides_file_path(make_failing_client):
    client = await make_failing_client(DatasetLoadError("Dataset file not found: /secret/path.csv"))

    response = await client.get("/dataset/preview")

    assert "/secret/path.csv" not in response.text


async def test_startup_survives_unavailable_dataset(monkeypatch, caplog):
    def broken_dataset():
        raise DatasetLoadError("Dataset file not found: /missing.csv")

    monkeypatch.setattr("src.main.get_dataset", broken_dataset)

    with caplog.at_level(logging.WARNING):
        async with lifespan(app):
            pass

    assert "Starting without a usable dataset: Dataset file not found: /missing.csv" in caplog.text


async def test_startup_survives_dataset_it_cannot_train_on(monkeypatch, caplog):
    def untrainable_split():
        raise DatasetNotTrainableError("Dataset has no labeled rows")

    monkeypatch.setattr("src.main.get_split", untrainable_split)

    with caplog.at_level(logging.WARNING):
        async with lifespan(app):
            pass

    assert "Starting without a usable dataset: Dataset has no labeled rows" in caplog.text
