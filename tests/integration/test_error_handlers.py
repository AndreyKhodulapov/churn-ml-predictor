import pytest
from httpx import ASGITransport, AsyncClient

from src.core.exceptions import DatasetLoadError
from src.main import app
from src.services.dataset import get_dataset


@pytest.fixture
async def failing_client():
    def broken_dataset():
        raise DatasetLoadError("Dataset file not found: /secret/path.csv")

    app.dependency_overrides[get_dataset] = broken_dataset
    # The app re-raises unhandled exceptions after responding; keep them out of the test to inspect the response.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_unhandled_error_returns_json_500(failing_client):
    response = await failing_client.get("/dataset/preview")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


async def test_unhandled_error_hides_exception_details(failing_client):
    response = await failing_client.get("/dataset/preview")

    assert "/secret/path.csv" not in response.text
