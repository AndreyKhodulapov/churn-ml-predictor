import pytest

from src.main import app
from src.schemas.dataset import DatasetRowChurn
from src.services.dataset import get_dataset


@pytest.fixture
def small_dataset(customer_features):
    rows = [DatasetRowChurn(**customer_features, churn=index % 2) for index in range(3)]
    app.dependency_overrides[get_dataset] = lambda: rows
    yield rows
    app.dependency_overrides.clear()


async def test_preview_returns_first_five_rows_by_default(client):
    response = await client.get("/dataset/preview")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    assert body[0] == {
        "monthly_fee": 9.99,
        "usage_hours": 27.92,
        "support_requests": 1,
        "account_age_months": 14,
        "failed_payments": 1,
        "region": "america",
        "device_type": "desktop",
        "payment_method": "card",
        "autopay_enabled": 1,
        "churn": 1,
    }


async def test_preview_returns_requested_number_of_rows(client, small_dataset):
    response = await client.get("/dataset/preview", params={"n": 2})

    assert response.status_code == 200
    assert response.json() == [row.model_dump() for row in small_dataset[:2]]


async def test_preview_returns_whole_dataset_when_n_exceeds_its_size(client, small_dataset):
    response = await client.get("/dataset/preview", params={"n": 10})

    assert response.status_code == 200
    assert len(response.json()) == len(small_dataset)


@pytest.mark.parametrize("n", [0, -5, 101, "abc"])
async def test_preview_rejects_invalid_n(client, n):
    response = await client.get("/dataset/preview", params={"n": n})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "n"]
