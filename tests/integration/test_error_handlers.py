import logging

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.exceptions import DatasetLoadError
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
    assert response.json() == {"detail": "Internal server error"}


async def test_unhandled_error_hides_exception_details(make_failing_client):
    client = await make_failing_client(RuntimeError("Broken state in /secret/path.csv"))

    response = await client.get("/dataset/preview")

    assert "/secret/path.csv" not in response.text


async def test_unavailable_dataset_returns_json_503(make_failing_client, caplog):
    client = await make_failing_client(DatasetLoadError("Dataset file not found: /secret/path.csv"))

    with caplog.at_level(logging.WARNING):
        response = await client.get("/dataset/info")

    assert response.status_code == 503
    assert response.json() == {"detail": "Dataset is not available"}
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

    assert "Starting without a dataset: Dataset file not found: /missing.csv" in caplog.text
