import logging

from src.schemas.learning import TrainingConfigChurn, TrainMetrics, TuningConfigChurn, TuningResultChurn
from src.services.learning import evaluate_churn_model, train_churn_model, tune_churn_model
from src.services.model_storage import ChurnModelStore, TrainedChurnModel
from src.services.preprocessing import SplitData
from src.services.training_history import TrainingHistory, describe_training

logger = logging.getLogger(__name__)


def train_and_save_churn_model(
    split: SplitData, config: TrainingConfigChurn, store: ChurnModelStore, history: TrainingHistory
) -> TrainMetrics:
    """Train a churn model as config says on the split, score it on the test part, make it the served one and record it.

    The store is changed only after training and scoring succeed, so a failed run keeps the previous model and
    leaves no record: it has no metrics to compare.
    """
    pipeline = train_churn_model(split.X_train, split.y_train, config.model_type, config.hyperparameters)
    metrics = evaluate_churn_model(pipeline, split.X_test, split.y_test)
    model = TrainedChurnModel(pipeline, metrics, config.model_type, config.hyperparameters)
    # Two trainings finishing together are recorded in the order they are served, so the latest record is always
    # the served model.
    with store.lock:
        store.save(model)
        try:
            history.append(describe_training(model))
        # The new model is already served: failing the request would make the client believe the training failed.
        except OSError:
            logger.exception("Churn model trained at %s is served but not recorded in the history", model.trained_at)
    return metrics


def tune_churn_model_on_split(split: SplitData, config: TuningConfigChurn) -> TuningResultChurn:
    """Search the grid of config on the train part of the split.

    It takes no store: tuning only suggests a config, the served model changes when /model/train trains it.
    """
    return tune_churn_model(split.X_train, split.y_train, config)
