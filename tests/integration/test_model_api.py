import pandas as pd
import pytest

from src.core.exceptions import DatasetLoadError
from src.main import app
from src.services.preprocessing import SplitData, get_split


@pytest.fixture
def separable_split(customer_features) -> SplitData:
    # Churn is decided by failed payments alone, so even a tiny model gets the test part right.
    def frame(failed_payments: list[int]) -> pd.DataFrame:
        return pd.DataFrame([{**customer_features, "failed_payments": value} for value in failed_payments])

    split = SplitData(
        X_train=frame([0, 0, 0, 0, 5, 5, 5, 5]),
        X_test=frame([0, 5]),
        y_train=pd.Series([0, 0, 0, 0, 1, 1, 1, 1]),
        y_test=pd.Series([0, 1]),
    )
    app.dependency_overrides[get_split] = lambda: split
    yield split
    app.dependency_overrides.clear()


async def test_train_returns_test_metrics_of_default_dataset(client):
    response = await client.post("/model/train")

    assert response.status_code == 200
    body = response.json()
    assert body.keys() == {"accuracy", "f1"}
    assert all(0 <= value <= 1 for value in body.values())


async def test_train_is_reproducible(client):
    first = await client.post("/model/train")
    second = await client.post("/model/train")

    assert first.json() == second.json()


async def test_train_with_balanced_class_weight_finds_more_churners(client):
    default = (await client.post("/model/train")).json()
    balanced = (await client.post("/model/train", params={"class_weight": "balanced"})).json()

    # Only 20% of customers churn: weighting them up trades accuracy for catching more of them.
    assert balanced["f1"] > default["f1"]
    assert balanced["accuracy"] < default["accuracy"]


async def test_train_rejects_unknown_class_weight(client):
    response = await client.post("/model/train", params={"class_weight": "heavy"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "class_weight"]


async def test_train_fits_and_scores_served_split(client, separable_split):
    response = await client.post("/model/train")

    assert response.status_code == 200
    assert response.json() == {"accuracy": 1.0, "f1": 1.0}


async def test_train_answers_503_without_dataset(client):
    def broken_split():
        raise DatasetLoadError("Dataset file not found: /secret/path.csv")

    app.dependency_overrides[get_split] = broken_split
    response = await client.post("/model/train")
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Dataset is not available"}
