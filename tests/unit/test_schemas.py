import pytest
from pydantic import ValidationError

from src.schemas.dataset import DatasetRowChurn
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
