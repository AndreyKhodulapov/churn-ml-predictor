import logging
from dataclasses import replace
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from src.core.exceptions import ModelLoadError
from src.schemas.learning import TrainMetrics
from src.services.learning import train_churn_model
from src.services.model_storage import TrainedChurnModel, load_churn_model, save_churn_model


@pytest.fixture
def X(customer_features) -> pd.DataFrame:
    return pd.DataFrame([{**customer_features, "failed_payments": value} for value in (0, 0, 5, 5)])


@pytest.fixture
def trained_model(X) -> TrainedChurnModel:
    return TrainedChurnModel(
        pipeline=train_churn_model(X, pd.Series([0, 0, 1, 1]), class_weight="balanced"),
        metrics=TrainMetrics(accuracy=0.75, f1=0.5),
        class_weight="balanced",
        trained_at=datetime(2026, 10, 25, 12, 30, tzinfo=UTC),
    )


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "churn_model.joblib"


def test_trained_model_records_training_time_in_utc_and_running_sklearn_version(trained_model):
    model = TrainedChurnModel(pipeline=trained_model.pipeline, metrics=trained_model.metrics)

    assert model.trained_at.tzinfo is UTC
    assert model.sklearn_version == sklearn.__version__


def test_loaded_model_predicts_like_the_saved_one(trained_model, model_path, X):
    save_churn_model(trained_model, model_path)

    loaded = load_churn_model(model_path)

    assert loaded.pipeline is not trained_model.pipeline
    np.testing.assert_array_equal(loaded.pipeline.predict_proba(X), trained_model.pipeline.predict_proba(X))


def test_loaded_model_keeps_metrics_training_time_and_class_weight(trained_model, model_path):
    save_churn_model(trained_model, model_path)

    loaded = load_churn_model(model_path)

    assert loaded.metrics == TrainMetrics(accuracy=0.75, f1=0.5)
    assert loaded.trained_at == datetime(2026, 10, 25, 12, 30, tzinfo=UTC)
    assert loaded.class_weight == "balanced"
    assert loaded.sklearn_version == sklearn.__version__


def test_save_creates_missing_directory(trained_model, tmp_path):
    path = tmp_path / "models" / "churn_model.joblib"

    save_churn_model(trained_model, path)

    assert path.is_file()


def test_save_replaces_previous_model_and_leaves_no_temporary_file(trained_model, model_path):
    save_churn_model(trained_model, model_path)
    save_churn_model(replace(trained_model, metrics=TrainMetrics(accuracy=1.0, f1=1.0)), model_path)

    assert load_churn_model(model_path).metrics == TrainMetrics(accuracy=1.0, f1=1.0)
    assert list(model_path.parent.iterdir()) == [model_path]


def test_load_rejects_missing_file(model_path):
    with pytest.raises(ModelLoadError, match="Cannot read churn model"):
        load_churn_model(model_path)


def test_load_rejects_damaged_file(model_path):
    model_path.write_bytes(b"not a model")

    with pytest.raises(ModelLoadError, match="Cannot read churn model"):
        load_churn_model(model_path)


def test_load_rejects_file_with_bare_pipeline(trained_model, model_path):
    joblib.dump(trained_model.pipeline, model_path)

    with pytest.raises(ModelLoadError, match="holds Pipeline, not a churn model"):
        load_churn_model(model_path)


def test_load_warns_about_model_saved_with_other_sklearn_version(trained_model, model_path, caplog):
    save_churn_model(replace(trained_model, sklearn_version="1.0.0"), model_path)

    with caplog.at_level(logging.WARNING):
        loaded = load_churn_model(model_path)

    assert loaded.sklearn_version == "1.0.0"
    assert f"saved with scikit-learn 1.0.0 and runs on {sklearn.__version__}" in caplog.text
