from datetime import UTC, datetime

import pandas as pd
import pytest

from src.core.exceptions import DatasetLoadError
from src.main import app
from src.services.model_storage import load_churn_model
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


async def test_train_serves_and_saves_the_trained_model(client, model_store):
    before = datetime.now(UTC)
    response = await client.post("/model/train", params={"class_weight": "balanced"})

    model = model_store.model
    assert model.metrics.model_dump() == response.json()
    assert model.class_weight == "balanced"
    assert before <= model.trained_at <= datetime.now(UTC)
    saved = load_churn_model(model_store.path)
    assert (saved.metrics, saved.class_weight, saved.trained_at) == (model.metrics, "balanced", model.trained_at)


async def test_train_replaces_the_previous_model(client, model_store):
    await client.post("/model/train", params={"class_weight": "balanced"})
    response = await client.post("/model/train")

    assert model_store.model.class_weight is None
    assert load_churn_model(model_store.path).metrics.model_dump() == response.json()


async def test_failed_training_keeps_the_previous_model(client, model_store, serve_csv, csv_header):
    await client.post("/model/train")
    trained = model_store.model
    serve_csv(f"{csv_header}\n")

    response = await client.post("/model/train")

    assert response.status_code == 409
    assert model_store.model is trained
    assert load_churn_model(model_store.path).trained_at == trained.trained_at


async def test_status_reports_untrained_model(client):
    response = await client.get("/model/status")

    assert response.status_code == 200
    assert response.json() == {"trained": False, "trained_at": None, "metrics": None, "class_weight": None}


async def test_status_reports_the_last_training(client, model_store):
    await client.post("/model/train")
    metrics = (await client.post("/model/train", params={"class_weight": "balanced"})).json()

    response = await client.get("/model/status")

    assert response.status_code == 200
    body = response.json()
    assert body["trained"] is True
    assert body["metrics"] == metrics
    assert body["class_weight"] == "balanced"
    assert datetime.fromisoformat(body["trained_at"]) == model_store.model.trained_at


async def test_status_does_not_need_the_dataset(client, model_store):
    await client.post("/model/train")

    def broken_split():
        raise DatasetLoadError("Dataset file not found: /secret/path.csv")

    app.dependency_overrides[get_split] = broken_split
    response = await client.get("/model/status")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["trained"] is True


async def test_train_rejects_unknown_class_weight(client):
    response = await client.post("/model/train", params={"class_weight": "heavy"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "class_weight"]


async def test_train_fits_and_scores_served_split(client, separable_split):
    response = await client.post("/model/train")

    assert response.status_code == 200
    assert response.json() == {"accuracy": 1.0, "f1": 1.0}


async def test_train_answers_503_without_dataset(client, model_store):
    def broken_split():
        raise DatasetLoadError("Dataset file not found: /secret/path.csv")

    app.dependency_overrides[get_split] = broken_split
    response = await client.post("/model/train")
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Dataset is not available"}
    assert model_store.model is None
    assert not model_store.path.exists()


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
