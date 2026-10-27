from src.schemas.learning import TrainingConfigChurn, TrainMetrics, TuningConfigChurn, TuningResultChurn
from src.services.learning import evaluate_churn_model, train_churn_model, tune_churn_model
from src.services.model_storage import ChurnModelStore, TrainedChurnModel
from src.services.preprocessing import SplitData


def train_and_save_churn_model(split: SplitData, config: TrainingConfigChurn, store: ChurnModelStore) -> TrainMetrics:
    """Train a churn model as config says on the split, score it on the test part and make it the served one.

    The store is changed only after training and scoring succeed, so a failed run keeps the previous model.
    """
    pipeline = train_churn_model(split.X_train, split.y_train, config.model_type, config.hyperparameters)
    metrics = evaluate_churn_model(pipeline, split.X_test, split.y_test)
    store.save(TrainedChurnModel(pipeline, metrics, config.model_type, config.hyperparameters))
    return metrics


def tune_churn_model_on_split(split: SplitData, config: TuningConfigChurn) -> TuningResultChurn:
    """Search the grid of config on the train part of the split.

    It takes no store: tuning only suggests a config, the served model changes when /model/train trains it.
    """
    return tune_churn_model(split.X_train, split.y_train, config)
