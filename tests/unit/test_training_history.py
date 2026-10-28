import logging
from datetime import UTC, datetime

import pandas as pd
import pytest

from src.core.constants import HISTORY_PATH
from src.core.exceptions import HistoryLoadError
from src.schemas.learning import TrainingRecord, TrainMetrics
from src.services.learning import train_churn_model
from src.services.model_storage import TrainedChurnModel
from src.services.training_history import (
    TrainingHistory,
    describe_training,
    get_training_history,
    load_history,
    save_history,
)


def make_record(day: int, model_type: str = "logreg", **hyperparameters) -> TrainingRecord:
    return TrainingRecord(
        trained_at=datetime(2026, 10, day, 12, 30, tzinfo=UTC),
        model_type=model_type,
        hyperparameters=hyperparameters,
        metrics=TrainMetrics(accuracy=0.75, f1=0.5, roc_auc=0.8),
    )


@pytest.fixture
def history_path(tmp_path):
    return tmp_path / "models" / "training_history.json"


def test_describe_training_records_hyperparameter_values_the_model_was_trained_with(customer_features):
    X = pd.DataFrame([{**customer_features, "failed_payments": value} for value in (0, 0, 5, 5)])
    model = TrainedChurnModel(
        pipeline=train_churn_model(X, pd.Series([0, 0, 1, 1]), "logreg", {"C": 0.5}),
        metrics=TrainMetrics(accuracy=0.75, f1=0.5, roc_auc=0.8),
        model_type="logreg",
        hyperparameters={"C": 0.5},
    )

    assert describe_training(model) == TrainingRecord(
        trained_at=model.trained_at,
        model_type="logreg",
        hyperparameters={"C": 0.5, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 100, "class_weight": None},
        metrics=TrainMetrics(accuracy=0.75, f1=0.5, roc_auc=0.8),
    )


def test_saved_records_load_back_equal(history_path):
    records = [make_record(25), make_record(26, "random_forest", max_depth=None, class_weight={0: 1, 1: 4})]

    save_history(records, history_path)

    # JSON object keys are strings; the record turns the class_weight ones back into the classes 0 and 1.
    assert load_history(history_path) == records


def test_save_replaces_previous_history_and_leaves_no_temporary_file(history_path):
    save_history([make_record(25)], history_path)
    save_history([make_record(26)], history_path)

    assert load_history(history_path) == [make_record(26)]
    assert list(history_path.parent.iterdir()) == [history_path]


@pytest.mark.parametrize("content", ["not json", '{"trained_at": "2026-10-25"}', '[{"model_type": "xgboost"}]'])
def test_load_rejects_file_without_training_records(history_path, content):
    history_path.parent.mkdir()
    history_path.write_text(content)

    with pytest.raises(HistoryLoadError, match="Cannot read training history"):
        load_history(history_path)


def test_history_starts_empty(history_path):
    assert TrainingHistory(history_path).records() == []


def test_history_lists_records_newest_first_and_writes_them_to_disk(history_path):
    history = TrainingHistory(history_path)

    history.append(make_record(25))
    history.append(make_record(26))

    assert history.records() == [make_record(26), make_record(25)]
    assert load_history(history_path) == [make_record(25), make_record(26)]


def test_history_lists_records_of_one_model_type(history_path):
    history = TrainingHistory(history_path)
    for record in (make_record(24), make_record(25, "random_forest"), make_record(26)):
        history.append(record)

    assert history.records("logreg") == [make_record(26), make_record(24)]
    assert history.records("random_forest") == [make_record(25, "random_forest")]


def test_history_drops_the_oldest_records_beyond_the_limit(history_path):
    history = TrainingHistory(history_path, max_records=2)
    for day in (24, 25, 26):
        history.append(make_record(day))

    assert history.records() == [make_record(26), make_record(25)]
    assert load_history(history_path) == [make_record(25), make_record(26)]


def test_history_keeps_previous_records_when_write_fails(history_path, monkeypatch):
    history = TrainingHistory(history_path)
    history.append(make_record(25))

    def failing_save(records, path):
        raise OSError("No space left on device")

    monkeypatch.setattr("src.services.training_history.save_history", failing_save)
    with pytest.raises(OSError):
        history.append(make_record(26))

    assert history.records() == [make_record(25)]


def test_history_load_keeps_records_saved_by_previous_run(history_path):
    TrainingHistory(history_path).append(make_record(25))
    history = TrainingHistory(history_path)

    history.load()

    assert history.records() == [make_record(25)]


def test_history_load_keeps_the_newest_records_within_the_limit(history_path):
    save_history([make_record(day) for day in (24, 25, 26)], history_path)
    history = TrainingHistory(history_path, max_records=2)

    history.load()

    assert history.records() == [make_record(26), make_record(25)]


def test_history_load_without_saved_file_stays_empty(history_path, caplog):
    history = TrainingHistory(history_path)

    with caplog.at_level(logging.INFO):
        history.load()

    assert history.records() == []
    assert f"No saved training history at {history_path}" in caplog.text


def test_history_load_rejects_damaged_file_and_stays_empty(history_path):
    history_path.parent.mkdir()
    history_path.write_text("not json")
    history = TrainingHistory(history_path)

    with pytest.raises(HistoryLoadError):
        history.load()

    assert history.records() == []


def test_service_uses_one_history_at_default_path():
    get_training_history.cache_clear()

    assert get_training_history() is get_training_history()
    assert get_training_history().path == HISTORY_PATH
    get_training_history.cache_clear()
