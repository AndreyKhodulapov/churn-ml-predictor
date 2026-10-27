import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.core.constants import FEATURES, MAX_TUNING_FITS
from src.core.exceptions import DatasetLoadError
from src.core.openapi_examples import TRAIN_EXAMPLES, TUNE_EXAMPLES
from src.main import app, lifespan
from src.services.model_storage import get_model_store, load_churn_model
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


@pytest.mark.parametrize("config", [{}, {"model_type": "random_forest", "hyperparameters": {"n_estimators": 20}}])
async def test_train_is_reproducible(client, config):
    first = await client.post("/model/train", json=config)
    second = await client.post("/model/train", json=config)

    assert first.json() == second.json()


async def test_train_without_config_trains_logreg_with_sklearn_defaults(client, model_store):
    without_body = (await client.post("/model/train")).json()
    model = model_store.model

    assert (model.model_type, model.hyperparameters) == ("logreg", {})
    assert type(model.pipeline.named_steps["model"]) is LogisticRegression
    assert (await client.post("/model/train", json={})).json() == without_body


async def test_train_random_forest_with_hyperparameters(client, model_store):
    config = {"model_type": "random_forest", "hyperparameters": {"n_estimators": 20, "max_depth": 5}}

    response = await client.post("/model/train", json=config)

    assert response.status_code == 200
    assert all(0 <= value <= 1 for value in response.json().values())
    forest = model_store.model.pipeline.named_steps["model"]
    assert type(forest) is RandomForestClassifier
    assert (forest.n_estimators, forest.max_depth) == (20, 5)


async def test_train_with_balanced_class_weight_finds_more_churners(client):
    default = (await client.post("/model/train")).json()
    balanced = (await client.post("/model/train", json={"hyperparameters": {"class_weight": "balanced"}})).json()

    # Only 20% of customers churn: weighting them up trades accuracy for catching more of them.
    assert balanced["f1"] > default["f1"]
    assert balanced["accuracy"] < default["accuracy"]


async def test_train_accepts_class_weight_dict_from_json(client, model_store):
    # JSON object keys are strings: the classes 0 and 1 arrive as "0" and "1".
    response = await client.post("/model/train", json={"hyperparameters": {"class_weight": {"0": 1, "1": 4}}})

    assert response.status_code == 200
    assert model_store.model.pipeline.named_steps["model"].class_weight == {0: 1, 1: 4}
    status = (await client.get("/model/status")).json()
    assert status["hyperparameters"]["class_weight"] == {"0": 1, "1": 4}


async def test_train_serves_and_saves_the_trained_model(client, model_store):
    before = datetime.now(UTC)
    response = await client.post("/model/train", json={"hyperparameters": {"class_weight": "balanced"}})

    model = model_store.model
    assert model.metrics.model_dump() == response.json()
    assert (model.model_type, model.hyperparameters) == ("logreg", {"class_weight": "balanced"})
    assert before <= model.trained_at <= datetime.now(UTC)
    saved = load_churn_model(model_store.path)
    assert (saved.metrics, saved.trained_at) == (model.metrics, model.trained_at)
    assert (saved.model_type, saved.hyperparameters) == ("logreg", {"class_weight": "balanced"})


async def test_train_replaces_the_previous_model(client, model_store):
    await client.post("/model/train", json={"hyperparameters": {"class_weight": "balanced"}})
    response = await client.post("/model/train")

    assert model_store.model.hyperparameters == {}
    assert load_churn_model(model_store.path).metrics.model_dump() == response.json()


async def test_failed_training_keeps_the_previous_model(client, model_store, serve_csv, csv_header):
    await client.post("/model/train")
    trained = model_store.model
    serve_csv(f"{csv_header}\n")

    response = await client.post("/model/train")

    assert response.status_code == 409
    assert model_store.model is trained
    assert load_churn_model(model_store.path).trained_at == trained.trained_at


async def test_openapi_shows_optional_train_config_with_examples(client):
    response = await client.get("/openapi.json")

    request_body = response.json()["paths"]["/model/train"]["post"]["requestBody"]
    assert not request_body.get("required", False)
    examples = request_body["content"]["application/json"]["examples"]
    assert {name: example["value"] for name, example in examples.items()} == {
        name: example["value"] for name, example in TRAIN_EXAMPLES.items()
    }


@pytest.mark.parametrize("name", TRAIN_EXAMPLES)
async def test_train_accepts_every_docs_example(client, model_store, name):
    config = TRAIN_EXAMPLES[name]["value"]

    response = await client.post("/model/train", json=config)

    assert response.status_code == 200
    assert model_store.model.model_type == config["model_type"]


async def test_status_reports_untrained_model(client):
    response = await client.get("/model/status")

    assert response.status_code == 200
    assert response.json() == {
        "trained": False,
        "trained_at": None,
        "metrics": None,
        "model_type": None,
        "hyperparameters": None,
    }


async def test_status_reports_the_last_training(client, model_store):
    await client.post("/model/train")
    config = {"model_type": "random_forest", "hyperparameters": {"n_estimators": 20, "class_weight": "balanced"}}
    metrics = (await client.post("/model/train", json=config)).json()

    response = await client.get("/model/status")

    assert response.status_code == 200
    body = response.json()
    assert body["trained"] is True
    assert body["metrics"] == metrics
    assert body["model_type"] == "random_forest"
    assert body["hyperparameters"] == {
        "n_estimators": 20,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    }
    assert datetime.fromisoformat(body["trained_at"]) == model_store.model.trained_at


async def test_status_shows_sklearn_defaults_of_model_trained_without_config(client):
    await client.post("/model/train")

    body = (await client.get("/model/status")).json()

    assert body["model_type"] == "logreg"
    assert body["hyperparameters"] == {
        "C": 1.0,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "max_iter": 100,
        "class_weight": None,
    }


async def test_status_does_not_need_the_dataset(client, model_store):
    await client.post("/model/train")

    def broken_split():
        raise DatasetLoadError("Dataset file not found: /secret/path.csv")

    app.dependency_overrides[get_split] = broken_split
    response = await client.get("/model/status")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["trained"] is True


@pytest.mark.parametrize(
    ("config", "loc"),
    [
        ({"model_type": "xgboost"}, ["body", "model_type"]),
        ({"hyperparams": {"C": 0.5}}, ["body", "hyperparams"]),
    ],
)
async def test_train_rejects_invalid_config(client, model_store, config, loc):
    response = await client.post("/model/train", json=config)

    assert response.status_code == 422
    assert response.json()["details"][0]["loc"] == loc
    assert model_store.model is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"hyperparameters": {"n_estimators": 10}},
            "Unknown hyperparameters of logreg: n_estimators; valid ones are C,",
        ),
        (
            {"hyperparameters": {"class_weight": "heavy"}},
            "Cannot train logreg with these hyperparameters: The 'class_weight'",
        ),
        (
            {"hyperparameters": {"l1_ratio": 1}},
            "Cannot train logreg with these hyperparameters: Solver lbfgs supports",
        ),
        (
            {"model_type": "random_forest", "hyperparameters": {"max_depth": 0}},
            "Cannot train random_forest with these hyperparameters: The 'max_depth' parameter",
        ),
    ],
)
async def test_train_answers_422_on_hyperparameters_the_model_refuses(client, model_store, config, message):
    await client.post("/model/train")
    trained = model_store.model

    response = await client.post("/model/train", json=config)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_hyperparameters"
    assert response.json()["message"].startswith(message)
    assert model_store.model is trained
    assert load_churn_model(model_store.path).trained_at == trained.trained_at


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
    assert response.json() == {"code": "dataset_unavailable", "message": "Dataset is not available", "details": None}
    assert model_store.model is None
    assert not model_store.path.exists()


async def test_train_answers_409_on_header_only_dataset(client, serve_csv, csv_header):
    serve_csv(f"{csv_header}\n")

    response = await client.post("/model/train")

    assert response.status_code == 409
    assert response.json() == {
        "code": "dataset_not_trainable",
        "message": "Dataset cannot be used for training: Dataset has no labeled rows",
        "details": None,
    }


async def test_train_answers_409_on_single_class_dataset(client, serve_csv, csv_header, make_csv_row):
    serve_csv("\n".join([csv_header, *[make_csv_row(0)] * 10]) + "\n")

    response = await client.post("/model/train")

    assert response.status_code == 409
    assert response.json() == {
        "code": "dataset_not_trainable",
        "message": "Dataset cannot be used for training: Dataset has only churn class 0, a classifier needs both",
        "details": None,
    }


async def test_train_answers_409_on_dataset_without_labels(client, serve_csv, csv_header, make_csv_row):
    unlabeled = make_csv_row(0).removesuffix("0")
    serve_csv(f"{csv_header}\n{unlabeled}\n{unlabeled}\n")

    response = await client.post("/model/train")

    assert response.status_code == 409
    assert response.json()["message"].endswith("Dataset has no labeled rows")


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


async def test_trained_model_survives_restart(client, model_store):
    config = {"model_type": "random_forest", "hyperparameters": {"n_estimators": 20, "class_weight": "balanced"}}
    await client.post("/model/train", json=config)
    status_before = (await client.get("/model/status")).json()

    # A restarted process begins with an empty store; the lifespan fills it from the saved file.
    get_model_store.cache_clear()
    async with lifespan(app):
        status_after = (await client.get("/model/status")).json()

    restarted = get_model_store()
    assert restarted is not model_store
    assert status_after == status_before
    X_test = get_split().X_test
    np.testing.assert_array_equal(
        restarted.model.pipeline.predict_proba(X_test), model_store.model.pipeline.predict_proba(X_test)
    )


async def test_startup_without_saved_model_reports_untrained_model(client):
    async with lifespan(app):
        response = await client.get("/model/status")

    assert response.json()["trained"] is False


async def test_startup_survives_damaged_model_file(client, model_store, caplog):
    model_store.path.parent.mkdir(parents=True)
    model_store.path.write_bytes(b"not a model")

    with caplog.at_level(logging.WARNING):
        async with lifespan(app):
            status = (await client.get("/model/status")).json()
            retrained = await client.post("/model/train")

    assert "Starting without a trained churn model: Cannot read churn model" in caplog.text
    assert status["trained"] is False
    assert retrained.status_code == 200
    assert load_churn_model(model_store.path).metrics.model_dump() == retrained.json()


@pytest.fixture
def tuning_split(customer_features) -> SplitData:
    # Ten customers of each class in the train part, enough for 5 folds; churn follows failed payments
    # except for one customer of each class, so the combinations score differently.
    def frame(failed_payments: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {**customer_features, "failed_payments": value, "usage_hours": float(hours)}
                for hours, value in enumerate(failed_payments)
            ]
        )

    split = SplitData(
        X_train=frame([0] * 9 + [5] + [5] * 9 + [0]),
        X_test=frame([0, 5]),
        y_train=pd.Series([0] * 10 + [1] * 10),
        y_test=pd.Series([0, 1]),
    )
    app.dependency_overrides[get_split] = lambda: split
    yield split
    app.dependency_overrides.clear()


async def test_tune_searches_default_logreg_grid_of_default_dataset(client):
    response = await client.post("/model/tune")

    assert response.status_code == 200
    body = response.json()
    assert body.keys() == {"best_config", "scoring", "best_score", "candidates"}
    assert body["scoring"] == "f1"
    assert body["best_config"]["model_type"] == "logreg"
    assert body["best_config"]["hyperparameters"].keys() == {"C", "class_weight"}
    assert len(body["candidates"]) == 5
    assert body["candidates"][0] == {
        "hyperparameters": body["best_config"]["hyperparameters"],
        "mean_score": body["best_score"],
        "std_score": body["candidates"][0]["std_score"],
    }


async def test_best_config_of_tune_is_sent_to_train_as_is(client, model_store, tuning_split):
    config = {"param_grid": {"C": [0.1, 1.0], "class_weight": [{"0": 1, "1": 4}]}, "cv": 3}
    best_config = (await client.post("/model/tune", json=config)).json()["best_config"]

    response = await client.post("/model/train", json=best_config)

    assert response.status_code == 200
    assert model_store.model.pipeline.named_steps["model"].class_weight == {0: 1, 1: 4}
    status = (await client.get("/model/status")).json()
    assert status["model_type"] == best_config["model_type"]
    assert status["hyperparameters"].items() >= best_config["hyperparameters"].items()


async def test_tune_does_not_change_the_served_model(client, model_store, tuning_split):
    await client.post("/model/train")
    trained = model_store.model
    status = (await client.get("/model/status")).json()

    response = await client.post("/model/tune", json={"param_grid": {"class_weight": ["balanced"]}, "cv": 3})

    assert response.status_code == 200
    assert model_store.model is trained
    assert load_churn_model(model_store.path).trained_at == trained.trained_at
    assert (await client.get("/model/status")).json() == status


async def test_tune_does_not_train_a_model(client, model_store, tuning_split):
    await client.post("/model/tune", json={"cv": 3})

    assert model_store.model is None
    assert not model_store.path.exists()


@pytest.mark.parametrize(
    ("config", "loc"),
    [
        ({"scoring": "precision"}, ["body", "scoring"]),
        ({"cv": 11}, ["body", "cv"]),
        ({"param_grid": {"C": []}}, ["body", "param_grid", "C"]),
        ({"grid": {"C": [0.5]}}, ["body", "grid"]),
        ({"param_grid": {"C": list(range(1, MAX_TUNING_FITS))}}, ["body"]),
    ],
)
async def test_tune_rejects_invalid_config(client, config, loc):
    response = await client.post("/model/tune", json=config)

    assert response.status_code == 422
    assert response.json()["details"][0]["loc"] == loc


@pytest.mark.parametrize(
    ("param_grid", "message"),
    [
        ({"n_estimators": [10]}, "Unknown hyperparameters of logreg: n_estimators; valid ones are C,"),
        (
            {"C": [0, 1.0]},
            'Cannot train logreg with 1 of 2 combinations of the grid: {"C": 0}; the first one fails with: The',
        ),
    ],
)
async def test_tune_answers_422_on_hyperparameters_the_model_refuses(client, tuning_split, param_grid, message):
    response = await client.post("/model/tune", json={"param_grid": param_grid, "cv": 3})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_hyperparameters"
    assert response.json()["message"].startswith(message)


async def test_tune_answers_409_when_a_class_is_smaller_than_cv(client, separable_split):
    response = await client.post("/model/tune", json={"cv": 5})

    assert response.status_code == 409
    assert response.json() == {
        "code": "dataset_not_trainable",
        "message": "Dataset cannot be used for training: 5-fold cross-validation needs at least 5 customers "
        "of each churn class in the train part, the rarest class has 4",
        "details": None,
    }


async def test_tune_answers_503_without_dataset(client):
    def broken_split():
        raise DatasetLoadError("Dataset file not found: /secret/path.csv")

    app.dependency_overrides[get_split] = broken_split
    response = await client.post("/model/tune")
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"code": "dataset_unavailable", "message": "Dataset is not available", "details": None}


async def test_openapi_shows_optional_tune_config_with_examples(client):
    response = await client.get("/openapi.json")

    request_body = response.json()["paths"]["/model/tune"]["post"]["requestBody"]
    assert not request_body.get("required", False)
    examples = request_body["content"]["application/json"]["examples"]
    assert {name: example["value"] for name, example in examples.items()} == {
        name: example["value"] for name, example in TUNE_EXAMPLES.items()
    }


@pytest.mark.parametrize("name", TUNE_EXAMPLES)
async def test_tune_accepts_every_docs_example(client, tuning_split, name):
    config = TUNE_EXAMPLES[name]["value"]

    response = await client.post("/model/tune", json=config)

    assert response.status_code == 200
    assert response.json()["best_config"]["model_type"] == config["model_type"]
    assert response.json()["scoring"] == config["scoring"]


async def test_schema_describes_features_before_the_first_training(client):
    response = await client.get("/model/schema")

    assert response.status_code == 200
    features = response.json()["features"]
    assert [feature["name"] for feature in features] == list(FEATURES)
    assert features[0] == {
        "name": "monthly_fee",
        "kind": "numeric",
        "type": "number",
        "allowed_values": None,
        "minimum": 0,
    }


async def test_predict_accepts_customer_built_from_schema(client):
    await client.post("/model/train")
    features = (await client.get("/model/schema")).json()["features"]
    # The smallest value of every numeric feature and the first allowed value of the others.
    customer = {
        feature["name"]: feature["minimum"] if feature["allowed_values"] is None else feature["allowed_values"][0]
        for feature in features
    }

    response = await client.post("/predict", json=customer)

    assert response.status_code == 200
