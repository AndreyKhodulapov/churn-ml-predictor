import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.core.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.schemas.learning import TrainMetrics
from src.services.learning import (
    build_churn_pipeline,
    build_feature_transformer,
    evaluate_churn_model,
    train_churn_model,
)


@pytest.fixture
def make_X(customer_features):
    def make(rows: int, **overrides: list) -> pd.DataFrame:
        X = pd.DataFrame([customer_features] * rows)
        for column, values in overrides.items():
            X[column] = values
        return X

    return make


@pytest.fixture
def X_train(make_X) -> pd.DataFrame:
    return make_X(
        4,
        monthly_fee=[10.0, 20.0, 30.0, None],
        usage_hours=[5.0, 15.0, 25.0, 35.0],
        autopay_enabled=[0, 1, 1, None],
        region=["asia", "europe", None, "europe"],
    )


y_train = pd.Series([0, 1, 0, 1])


def test_churn_pipeline_fills_gaps_transforms_and_classifies_with_given_classifier():
    classifier = LogisticRegression()
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, classifier)

    assert list(pipeline.named_steps) == ["impute", "transform", "model"]
    assert pipeline.named_steps["model"] is classifier


def test_fitted_churn_pipeline_predicts_raw_rows_with_gaps(X_train):
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, LogisticRegression()).fit(X_train, y_train)

    predictions = pipeline.predict(X_train)

    check_is_fitted(pipeline)
    assert set(predictions) <= {0, 1}
    assert len(predictions) == len(X_train)


def test_churn_pipeline_learns_gap_filling_from_train_part(X_train, make_X):
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, LogisticRegression()).fit(X_train, y_train)
    X_test = make_X(1, monthly_fee=[None])

    filled = pipeline.named_steps["impute"].transform(X_test)

    # The median of the train fees [10, 20, 30].
    assert filled.loc[0, "monthly_fee"] == 20.0


def test_feature_transformer_scales_numeric_passes_binary_and_one_hot_encodes_categorical(make_X):
    X = make_X(2, monthly_fee=[10.0, 30.0], autopay_enabled=[0, 1], region=["asia", "europe"])
    transformer = build_feature_transformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(X)

    transformed = pd.DataFrame(transformer.transform(X), columns=transformer.get_feature_names_out())

    assert transformed["monthly_fee"].tolist() == [-1.0, 1.0]
    assert transformed["autopay_enabled"].tolist() == [0, 1]
    assert transformed[["region_asia", "region_europe"]].values.tolist() == [[1, 0], [0, 1]]
    assert "region" not in transformed.columns


def test_feature_transformer_ignores_category_unseen_in_train_part(make_X):
    transformer = build_feature_transformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(
        make_X(2, region=["asia", "europe"])
    )

    transformed = pd.DataFrame(
        transformer.transform(make_X(1, region=["africa"])), columns=transformer.get_feature_names_out()
    )

    assert np.all(transformed[["region_asia", "region_europe"]] == 0)


def test_train_churn_model_returns_fitted_pipeline_without_class_weights(X_train):
    pipeline = train_churn_model(X_train, y_train)

    check_is_fitted(pipeline)
    assert pipeline.named_steps["model"].class_weight is None


def test_train_churn_model_passes_class_weight_to_model(X_train):
    pipeline = train_churn_model(X_train, y_train, class_weight="balanced")

    assert pipeline.named_steps["model"].class_weight == "balanced"


def test_train_churn_model_returns_separate_model_on_every_call(X_train):
    first, second = train_churn_model(X_train, y_train), train_churn_model(X_train, y_train)

    assert first is not second


def fixed_predictions(predictions: list[int]) -> Mock:
    pipeline = Mock(spec=Pipeline)
    pipeline.predict.return_value = np.array(predictions)
    return pipeline


def test_evaluate_churn_model_scores_test_predictions():
    pipeline = fixed_predictions([0, 1, 0, 0])
    X_test = pd.DataFrame(index=range(4))

    metrics = evaluate_churn_model(pipeline, X_test, pd.Series([0, 1, 1, 0]))

    pipeline.predict.assert_called_once_with(X_test)
    # 3 of 4 rows are right; churn precision 1/1 and recall 1/2 give F1 = 2/3.
    assert metrics.accuracy == 0.75
    assert metrics.f1 == pytest.approx(2 / 3)


def test_evaluate_churn_model_gives_zero_f1_to_model_that_never_predicts_churn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = evaluate_churn_model(
            fixed_predictions([0, 0, 0, 0]), pd.DataFrame(index=range(4)), pd.Series([0, 1, 0, 0])
        )

    assert metrics == TrainMetrics(accuracy=0.75, f1=0.0)
