from typing import get_args

import pandas as pd
import pytest

from src.core.constants import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, DeviceType, PaymentMethod, Region
from src.core.exceptions import ModelFeaturesMismatchError, ModelNotTrainedError
from src.schemas.learning import TrainMetrics
from src.schemas.prediction import FeatureDescription, FeatureVectorChurn
from src.services.learning import train_churn_model
from src.services.model_storage import TrainedChurnModel
from src.services.prediction import build_model_input, describe_features, predict_churn


@pytest.fixture
def make_customer(customer_features):
    def make(failed_payments: int) -> FeatureVectorChurn:
        return FeatureVectorChurn(**{**customer_features, "failed_payments": failed_payments})

    return make


@pytest.fixture
def make_model(make_customer):
    def make(columns: list[str]) -> TrainedChurnModel:
        # Churn is decided by failed payments alone, so even a tiny model separates the customers.
        X = pd.DataFrame([make_customer(value).model_dump() for value in [0, 0, 0, 5, 5, 5]])
        pipeline = train_churn_model(X.reindex(columns=columns, fill_value=0), pd.Series([0, 0, 0, 1, 1, 1]))
        return TrainedChurnModel(pipeline, TrainMetrics(accuracy=1.0, f1=1.0), "logreg", {})

    return make


@pytest.fixture
def model(make_model) -> TrainedChurnModel:
    return make_model(list(FEATURES))


def test_predicts_one_customer(model, make_customer):
    [prediction] = predict_churn(model, make_customer(5))

    assert prediction.churn == 1
    assert prediction.probability_churn > 0.5
    assert prediction.probability_no_churn + prediction.probability_churn == pytest.approx(1)


def test_predicts_several_customers_in_order(model, make_customer):
    predictions = predict_churn(model, [make_customer(0), make_customer(5)])

    assert [prediction.churn for prediction in predictions] == [0, 1]


def test_matches_the_pipeline_on_the_same_rows(model, make_customer):
    customers = [make_customer(0), make_customer(5)]

    predictions = predict_churn(model, customers)

    X = pd.DataFrame([customer.model_dump() for customer in customers])
    assert [prediction.probability_churn for prediction in predictions] == list(model.pipeline.predict_proba(X)[:, 1])


def test_model_input_has_the_training_columns_in_their_order(make_model, make_customer):
    # FeatureVectorChurn lists autopay_enabled last, training puts it before the categorical features.
    assert list(FeatureVectorChurn.model_fields) != list(FEATURES)
    model = make_model(list(FEATURES))

    X = build_model_input(model.pipeline, [make_customer(0)])

    assert list(X.columns) == list(FEATURES)
    assert X.iloc[0].to_dict() == make_customer(0).model_dump()


def test_model_input_follows_the_column_order_of_the_served_model(make_model, make_customer):
    columns = list(reversed(FEATURES))
    model = make_model(columns)

    X = build_model_input(model.pipeline, [make_customer(0), make_customer(5)])

    assert list(X.columns) == columns
    assert [prediction.churn for prediction in predict_churn(model, [make_customer(0), make_customer(5)])] == [0, 1]


def test_refuses_model_trained_on_other_features(make_model, make_customer):
    # Without the check discount would arrive as NaN, be filled like a gap and the model would predict anyway.
    model = make_model([*FEATURES, "discount"])

    with pytest.raises(ModelFeaturesMismatchError, match="discount"):
        predict_churn(model, make_customer(0))


def test_returns_no_predictions_for_no_customers(model):
    assert predict_churn(model, []) == []


def test_refuses_without_trained_model(make_customer):
    with pytest.raises(ModelNotTrainedError):
        predict_churn(None, make_customer(0))


def test_describe_features_lists_training_features_in_their_order():
    assert [feature.name for feature in describe_features().features] == list(FEATURES)


def test_describe_features_names_how_the_model_treats_each_feature():
    kinds = {feature.name: feature.kind for feature in describe_features().features}

    assert {name for name, kind in kinds.items() if kind == "categorical"} == set(CATEGORICAL_FEATURES)
    assert {name for name, kind in kinds.items() if kind == "binary"} == set(BINARY_FEATURES)
    assert kinds["monthly_fee"] == kinds["failed_payments"] == "numeric"


def test_describe_features_gives_types_and_values_feature_vector_accepts():
    features = {feature.name: feature for feature in describe_features().features}

    assert features["monthly_fee"] == FeatureDescription(name="monthly_fee", kind="numeric", type="number", minimum=0)
    assert features["support_requests"].type == "integer"
    assert features["autopay_enabled"] == FeatureDescription(
        name="autopay_enabled", kind="binary", type="integer", allowed_values=[0, 1]
    )
    assert features["region"].allowed_values == list(get_args(Region))
    assert features["device_type"].allowed_values == list(get_args(DeviceType))
    assert features["payment_method"].allowed_values == list(get_args(PaymentMethod))
    assert all(feature.type == "string" for feature in features.values() if feature.kind == "categorical")
