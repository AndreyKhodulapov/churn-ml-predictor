import pandas as pd
import pytest

from src.schemas.learning import TrainMetrics
from src.services.model import train_and_save_churn_model
from src.services.model_storage import ChurnModelStore, load_churn_model
from src.services.preprocessing import SplitData


@pytest.fixture
def make_X(customer_features):
    def make(failed_payments: list[int]) -> pd.DataFrame:
        return pd.DataFrame([{**customer_features, "failed_payments": value} for value in failed_payments])

    return make


@pytest.fixture
def separable_split(make_X) -> SplitData:
    # Churn is decided by failed payments alone, so even a tiny model gets the test part right.
    return SplitData(
        X_train=make_X([0, 0, 0, 5, 5, 5]),
        X_test=make_X([0, 5]),
        y_train=pd.Series([0, 0, 0, 1, 1, 1]),
        y_test=pd.Series([0, 1]),
    )


@pytest.fixture
def store(tmp_path) -> ChurnModelStore:
    return ChurnModelStore(tmp_path / "churn_model.joblib")


def test_returns_test_metrics(separable_split, store):
    metrics = train_and_save_churn_model(separable_split, None, store)

    assert metrics == TrainMetrics(accuracy=1.0, f1=1.0)


def test_serves_and_saves_the_trained_model(separable_split, store):
    metrics = train_and_save_churn_model(separable_split, "balanced", store)

    assert store.model.metrics == metrics
    assert store.model.class_weight == "balanced"
    saved = load_churn_model(store.path)
    assert (saved.metrics, saved.class_weight, saved.trained_at) == (metrics, "balanced", store.model.trained_at)
    assert list(saved.pipeline.predict(separable_split.X_test)) == [0, 1]


def test_failed_training_keeps_the_previous_model(separable_split, store):
    train_and_save_churn_model(separable_split, None, store)
    trained = store.model
    single_class_split = separable_split._replace(y_train=pd.Series([0] * 6))

    # LogisticRegression refuses to fit on a single class.
    with pytest.raises(ValueError):
        train_and_save_churn_model(single_class_split, None, store)

    assert store.model is trained
    assert load_churn_model(store.path).trained_at == trained.trained_at
