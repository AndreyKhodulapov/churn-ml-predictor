import pytest

from src.main import app
from src.services.dataset import DatasetProcessor, get_dataset


@pytest.fixture
def small_dataset(write_csv, csv_header, make_csv_row) -> DatasetProcessor:
    rows = [make_csv_row(churn) for churn in (0, 1, 0)]
    processor = DatasetProcessor(write_csv("\n".join([csv_header, *rows]) + "\n"))
    processor.load()
    app.dependency_overrides[get_dataset] = lambda: processor
    yield processor
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
    assert response.json() == [row.model_dump() for row in small_dataset.preview(2)]


async def test_preview_returns_whole_dataset_when_n_exceeds_its_size(client, small_dataset):
    response = await client.get("/dataset/preview", params={"n": 10})

    assert response.status_code == 200
    assert len(response.json()) == 3


@pytest.mark.parametrize("n", [0, -5, 101, "abc"])
async def test_preview_rejects_invalid_n(client, n):
    response = await client.get("/dataset/preview", params={"n": n})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "n"]


async def test_info_describes_default_dataset(client):
    response = await client.get("/dataset/info")

    assert response.status_code == 200
    assert response.json() == {
        "row_count": 2000,
        "column_count": 10,
        "feature_names": [
            "monthly_fee",
            "usage_hours",
            "support_requests",
            "account_age_months",
            "failed_payments",
            "region",
            "device_type",
            "payment_method",
            "autopay_enabled",
        ],
        "target_distribution": {
            "0": {"count": 1597, "share": 0.7985},
            "1": {"count": 403, "share": 0.2015},
        },
        "target_missing": {"count": 0, "share": 0.0},
    }


async def test_info_reflects_served_dataset(client, small_dataset):
    response = await client.get("/dataset/info")

    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 3
    assert body["target_distribution"] == {
        "0": {"count": 2, "share": 2 / 3},
        "1": {"count": 1, "share": 1 / 3},
    }


async def test_info_reports_unlabeled_rows_apart_from_classes(client, write_csv, csv_header, make_csv_row):
    rows = [make_csv_row(0), make_csv_row(1).removesuffix("1")]
    processor = DatasetProcessor(write_csv("\n".join([csv_header, *rows]) + "\n"))
    processor.load()
    app.dependency_overrides[get_dataset] = lambda: processor

    response = await client.get("/dataset/info")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["target_distribution"] == {"0": {"count": 1, "share": 0.5}}
    assert body["target_missing"] == {"count": 1, "share": 0.5}


async def test_info_does_not_depend_on_preview_being_called_first(client):
    first = await client.get("/dataset/info")
    await client.get("/dataset/preview")
    second = await client.get("/dataset/info")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
