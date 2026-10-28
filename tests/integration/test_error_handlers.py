import logging

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.exceptions import DatasetLoadError, DatasetNotTrainableError
from src.core.openapi_examples import DATASET_UNAVAILABLE_ERROR, INVALID_N_ERROR, NO_LABELED_ROWS_ERROR
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


async def test_unhandled_error_logs_the_request(make_failing_client, caplog):
    client = await make_failing_client(RuntimeError("Broken state in /secret/path.csv"))

    with caplog.at_level(logging.ERROR):
        await client.get("/dataset/preview")

    assert "GET /dataset/preview failed with an unhandled RuntimeError" in caplog.text


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
    assert response.json() == INVALID_N_ERROR


async def test_validation_error_does_not_echo_the_input(client, customer_features):
    customer_features["region"] = "antarctica"

    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 422
    assert "antarctica" not in response.text


async def test_validation_error_is_logged_without_the_input(client, customer_features, caplog):
    customer_features["region"] = "antarctica"

    with caplog.at_level(logging.WARNING):
        await client.post("/predict", json=[customer_features, customer_features])

    assert "POST /predict failed validation with 2 errors" in caplog.text
    assert "first at body.list[FeatureVectorChurn].0.region: literal_error" in caplog.text
    assert "antarctica" not in caplog.text


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
    assert response.json() == DATASET_UNAVAILABLE_ERROR
    assert "/secret/path.csv" in caplog.text


async def test_unavailable_dataset_hides_file_path(make_failing_client):
    client = await make_failing_client(DatasetLoadError("Dataset file not found: /secret/path.csv"))

    response = await client.get("/dataset/preview")

    assert "/secret/path.csv" not in response.text


async def test_openapi_shows_dataset_errors(documented_errors):
    assert documented_errors("get", "/dataset/preview", 422) == {"invalid_n": INVALID_N_ERROR}
    for path in ["/dataset/preview", "/dataset/info", "/dataset/split-info"]:
        assert documented_errors("get", path, 503) == {"dataset_unavailable": DATASET_UNAVAILABLE_ERROR}
    assert documented_errors("get", "/dataset/split-info", 409) == {"no_labeled_rows": NO_LABELED_ROWS_ERROR}


async def test_openapi_documents_no_validation_error_schema_of_fastapi(client):
    # FastAPI documents its own 422 body on every operation with parameters unless the operation sets 422 itself.
    schema = (await client.get("/openapi.json")).json()

    assert "HTTPValidationError" not in schema["components"]["schemas"]
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if "422" in operation["responses"]:
                assert operation["responses"]["422"]["content"]["application/json"]["schema"] == {
                    "$ref": "#/components/schemas/ErrorResponse"
                }, f"{method} {path}"


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
