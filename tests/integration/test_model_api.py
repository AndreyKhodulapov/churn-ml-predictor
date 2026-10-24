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


async def test_train_answers_409_on_header_only_dataset(client, serve_csv, csv_header):
    serve_csv(f"{csv_header}\n")

    response = await client.post("/model/train")

    assert response.status_code == 409
    assert response.json() == {"detail": "Dataset cannot be used for training: Dataset has no labeled rows"}


async def test_train_answers_409_on_single_class_dataset(client, serve_csv, csv_header, make_csv_row):
    serve_csv("\n".join([csv_header, *[make_csv_row(0)] * 10]) + "\n")

    response = await client.post("/model/train")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Dataset cannot be used for training: Dataset has only churn class 0, a classifier needs both"
    }


async def test_train_answers_409_on_dataset_without_labels(client, serve_csv, csv_header, make_csv_row):
    unlabeled = make_csv_row(0).removesuffix("0")
    serve_csv(f"{csv_header}\n{unlabeled}\n{unlabeled}\n")

    response = await client.post("/model/train")

    assert response.status_code == 409
    assert response.json()["detail"].endswith("Dataset has no labeled rows")


async def test_train_fills_feature_gaps_inside_the_pipeline(client, serve_csv, csv_header, make_csv_row):
    def with_gaps(churn: int) -> str:
        # Blank monthly_fee and region: the split keeps the gaps and the imputer step of the pipeline fills them.
        values = make_csv_row(churn).split(",")
        values[0] = values[5] = ""
        return ",".join(values)

    rows = [make_csv_row(0)] * 4 + [with_gaps(0)] + [make_csv_row(1)] * 4 + [with_gaps(1)]
    serve_csv("\n".join([csv_header, *rows]) + "\n")

    response = await client.post("/model/train")

    assert response.status_code == 200
    split = get_split()
    assert split.X_train.isna().sum().sum() + split.X_test.isna().sum().sum() == 4
