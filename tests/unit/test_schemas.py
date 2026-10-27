import math

import pytest
from pydantic import ValidationError

from src.core.constants import DEFAULT_PARAM_GRIDS, MAX_TUNING_FITS
from src.schemas.dataset import DatasetRowChurn
from src.schemas.learning import TrainingConfigChurn, TuningCandidate, TuningConfigChurn, TuningResultChurn
from src.schemas.prediction import FeatureVectorChurn


def test_feature_vector_accepts_valid_customer(customer_features):
    features = FeatureVectorChurn(**customer_features)

    assert features.model_dump() == customer_features


def test_feature_vector_coerces_numeric_strings(customer_features):
    customer_features.update(monthly_fee="19.5", support_requests="3")

    features = FeatureVectorChurn(**customer_features)

    assert features.monthly_fee == 19.5
    assert features.support_requests == 3


@pytest.mark.parametrize(
    "field, value",
    [
        ("region", "antarctica"),
        ("device_type", "smartwatch"),
        ("payment_method", "cash"),
        ("autopay_enabled", 2),
        ("monthly_fee", -1.0),
        ("failed_payments", -1),
        ("account_age_months", 1.5),
    ],
)
def test_feature_vector_rejects_invalid_value(customer_features, field, value):
    customer_features[field] = value

    with pytest.raises(ValidationError):
        FeatureVectorChurn(**customer_features)


def test_feature_vector_requires_all_fields(customer_features):
    del customer_features["usage_hours"]

    with pytest.raises(ValidationError):
        FeatureVectorChurn(**customer_features)


def test_dataset_row_has_same_features_as_feature_vector():
    assert DatasetRowChurn.model_fields.keys() - {"churn"} == FeatureVectorChurn.model_fields.keys()


def test_dataset_row_accepts_features_with_label(customer_features):
    row = DatasetRowChurn(**customer_features, churn=1)

    assert row.model_dump() == {**customer_features, "churn": 1}


def test_dataset_row_accepts_missing_values(customer_features):
    row = DatasetRowChurn(**dict.fromkeys(customer_features), churn=None)

    assert set(row.model_dump().values()) == {None}


@pytest.mark.parametrize("field, value", [("churn", 5), ("monthly_fee", -1.0), ("region", "antarctica")])
def test_dataset_row_rejects_invalid_present_value(customer_features, field, value):
    row = {**customer_features, "churn": 1, field: value}

    with pytest.raises(ValidationError):
        DatasetRowChurn(**row)


def test_dataset_row_requires_all_columns(customer_features):
    del customer_features["region"]

    with pytest.raises(ValidationError):
        DatasetRowChurn(**customer_features, churn=1)


def test_training_config_defaults_to_logreg_with_sklearn_defaults():
    config = TrainingConfigChurn()

    assert config.model_dump() == {"model_type": "logreg", "hyperparameters": {}}


def test_training_config_accepts_model_type_with_hyperparameters():
    config = TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": 300, "max_depth": None})

    assert config.model_type == "random_forest"
    assert config.hyperparameters == {"n_estimators": 300, "max_depth": None}


def test_training_configs_do_not_share_hyperparameters():
    first, second = TrainingConfigChurn(), TrainingConfigChurn()
    first.hyperparameters["C"] = 0.5

    assert second.hyperparameters == {}


@pytest.mark.parametrize(
    "fields",
    [
        {"model_type": "xgboost"},
        {"hyperparameters": ["C", 0.5]},
        {"hyperparams": {"C": 0.5}},
    ],
)
def test_training_config_rejects_invalid_fields(fields):
    with pytest.raises(ValidationError):
        TrainingConfigChurn(**fields)


def test_training_config_restores_int_class_labels_of_class_weight():
    config = TrainingConfigChurn(hyperparameters={"C": 0.5, "class_weight": {"0": 1, "1": 5}})

    assert config.hyperparameters == {"C": 0.5, "class_weight": {0: 1, 1: 5}}


@pytest.mark.parametrize("class_weight", ["balanced", None, {0: 1, 1: 5}, {"stays": 1}])
def test_training_config_keeps_other_class_weights(class_weight):
    config = TrainingConfigChurn(hyperparameters={"class_weight": class_weight})

    assert config.hyperparameters == {"class_weight": class_weight}


def test_tuning_config_defaults_to_logreg_grid_scored_by_f1():
    config = TuningConfigChurn()

    assert (config.model_type, config.param_grid, config.scoring, config.cv) == ("logreg", None, "f1", 5)
    assert config.grid == DEFAULT_PARAM_GRIDS["logreg"]


def test_tuning_config_searches_the_default_grid_of_its_model_type():
    assert TuningConfigChurn(model_type="random_forest").grid == DEFAULT_PARAM_GRIDS["random_forest"]


def test_tuning_config_searches_the_requested_grid():
    config = TuningConfigChurn(model_type="random_forest", param_grid={"max_depth": [None, 5]}, scoring="roc_auc")

    assert config.grid == {"max_depth": [None, 5]}


def test_tuning_config_grid_does_not_share_the_default_grid():
    TuningConfigChurn().grid["C"].append(100.0)

    assert 100.0 not in TuningConfigChurn().grid["C"]


@pytest.mark.parametrize("model_type", DEFAULT_PARAM_GRIDS)
def test_default_grids_fit_the_limit_with_most_folds(model_type):
    combinations = math.prod(len(values) for values in DEFAULT_PARAM_GRIDS[model_type].values())

    assert combinations * 10 <= MAX_TUNING_FITS


def test_tuning_config_restores_int_class_labels_in_the_grid():
    config = TuningConfigChurn(param_grid={"class_weight": [None, "balanced", {"0": 1, "1": 4}]})

    assert config.grid == {"class_weight": [None, "balanced", {0: 1, 1: 4}]}


def test_tuning_config_accepts_the_largest_grid_within_the_limit():
    config = TuningConfigChurn(param_grid={"C": list(range(1, MAX_TUNING_FITS // 5 + 1))}, cv=5)

    assert len(config.grid["C"]) * config.cv == MAX_TUNING_FITS


def test_tuning_config_rejects_grid_over_the_fit_limit():
    with pytest.raises(ValidationError, match=f"needs {(MAX_TUNING_FITS // 5 + 1) * 5} fits"):
        TuningConfigChurn(param_grid={"C": list(range(1, MAX_TUNING_FITS // 5 + 2))}, cv=5)


@pytest.mark.parametrize(
    "fields",
    [
        {"scoring": "precision"},
        {"cv": 1},
        {"cv": 11},
        {"param_grid": {}},
        {"param_grid": {"C": []}},
        {"param_grid": {"C": 0.5}},
        {"model_type": "xgboost"},
        {"grid": {"C": [0.5]}},
    ],
)
def test_tuning_config_rejects_invalid_fields(fields):
    with pytest.raises(ValidationError):
        TuningConfigChurn(**fields)


def test_tuning_result_best_config_is_a_valid_train_body():
    best_config = TrainingConfigChurn(hyperparameters={"C": 0.1, "class_weight": {0: 1, 1: 4}})
    result = TuningResultChurn(
        best_config=best_config,
        scoring="f1",
        best_score=0.4,
        candidates=[TuningCandidate(hyperparameters=best_config.hyperparameters, mean_score=0.4, std_score=0.02)],
    )

    # The client sends best_config from the JSON response back to /model/train as it is.
    train_body = result.model_dump(mode="json")["best_config"]
    assert TrainingConfigChurn.model_validate(train_body) == best_config
