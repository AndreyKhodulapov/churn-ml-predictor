import logging
import threading

import pandas as pd
import pytest

from src.core.exceptions import InvalidHyperparametersError
from src.schemas.learning import TrainingConfigChurn, TrainingRecord, TrainMetrics, TuningConfigChurn
from src.services.model import train_and_save_churn_model, tune_churn_model_on_split
from src.services.model_storage import ChurnModelStore, load_churn_model
from src.services.preprocessing import SplitData
from src.services.training_history import TrainingHistory, load_history


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


@pytest.fixture
def history(tmp_path) -> TrainingHistory:
    return TrainingHistory(tmp_path / "training_history.json")


def test_returns_test_metrics(separable_split, store, history):
    metrics = train_and_save_churn_model(separable_split, TrainingConfigChurn(), store, history)

    assert metrics == TrainMetrics(accuracy=1.0, f1=1.0, roc_auc=1.0)


def test_serves_and_saves_the_trained_model_with_its_type_and_hyperparameters(separable_split, store, history):
    config = TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": 10})

    metrics = train_and_save_churn_model(separable_split, config, store, history)

    assert store.model.metrics == metrics
    assert (store.model.model_type, store.model.hyperparameters) == ("random_forest", {"n_estimators": 10})
    saved = load_churn_model(store.path)
    assert (saved.metrics, saved.trained_at) == (metrics, store.model.trained_at)
    assert (saved.model_type, saved.hyperparameters) == ("random_forest", {"n_estimators": 10})
    assert list(saved.pipeline.predict(separable_split.X_test)) == [0, 1]


def test_failed_training_keeps_the_previous_model(separable_split, store, history):
    train_and_save_churn_model(separable_split, TrainingConfigChurn(), store, history)
    trained = store.model

    # LogisticRegression refuses a negative regularization strength only once it is fitted.
    with pytest.raises(InvalidHyperparametersError):
        train_and_save_churn_model(separable_split, TrainingConfigChurn(hyperparameters={"C": -1}), store, history)

    assert store.model is trained
    assert load_churn_model(store.path).trained_at == trained.trained_at


def test_records_every_training_with_values_of_its_hyperparameters(separable_split, store, history):
    train_and_save_churn_model(separable_split, TrainingConfigChurn(), store, history)
    config = TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": 10})
    metrics = train_and_save_churn_model(separable_split, config, store, history)

    newest, oldest = history.records()
    assert newest == TrainingRecord(
        trained_at=store.model.trained_at,
        model_type="random_forest",
        hyperparameters={
            "n_estimators": 10,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
        },
        metrics=metrics,
    )
    assert oldest.model_type == "logreg"
    assert load_history(history.path) == [oldest, newest]


def test_failed_training_leaves_no_record(separable_split, store, history):
    train_and_save_churn_model(separable_split, TrainingConfigChurn(), store, history)
    recorded = history.records()

    with pytest.raises(InvalidHyperparametersError):
        train_and_save_churn_model(separable_split, TrainingConfigChurn(hyperparameters={"C": -1}), store, history)

    assert history.records() == recorded


def test_records_the_training_while_no_other_one_can_be_served(separable_split, store, history, monkeypatch):
    lock_free_while_recording = []
    append = history.append

    def checked_append(record):
        # Another training finishing now runs in its own thread and must wait for the store.
        attempt = threading.Thread(target=lambda: lock_free_while_recording.append(store.lock.acquire(blocking=False)))
        attempt.start()
        attempt.join()
        append(record)

    monkeypatch.setattr(history, "append", checked_append)
    train_and_save_churn_model(separable_split, TrainingConfigChurn(), store, history)

    assert lock_free_while_recording == [False]
    assert len(history.records()) == 1


def test_serves_the_model_when_its_record_cannot_be_written(separable_split, store, history, monkeypatch, caplog):
    def failing_save(records, path):
        raise OSError("No space left on device")

    monkeypatch.setattr("src.services.training_history.save_history", failing_save)
    with caplog.at_level(logging.ERROR):
        metrics = train_and_save_churn_model(separable_split, TrainingConfigChurn(), store, history)

    assert store.model.metrics == metrics
    assert history.records() == []
    assert "is served but not recorded in the history" in caplog.text


def test_tuning_uses_the_train_part_only(separable_split):
    # A test part the pipeline cannot even read: tuning must not touch it.
    split = separable_split._replace(X_test=None, y_test=None)

    result = tune_churn_model_on_split(split, TuningConfigChurn(param_grid={"C": [0.1, 1.0]}, cv=3))

    assert result.best_score == 1.0
