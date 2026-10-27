import logging
import threading
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from src.core.constants import MODEL_PATH, ModelType
from src.core.exceptions import ModelFeaturesMismatchError, ModelLoadError
from src.schemas.learning import ModelStatus, TrainMetrics
from src.schemas.prediction import FeatureVectorChurn
from src.services.learning import describe_hyperparameters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainedChurnModel:
    """A fitted churn pipeline saved together with what is reported about it after a restart."""

    pipeline: Pipeline
    metrics: TrainMetrics
    model_type: ModelType
    # As requested for training; the rest are the scikit-learn defaults.
    hyperparameters: dict[str, Any]
    trained_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # A pickled estimator is only guaranteed to behave the same under the scikit-learn version that saved it.
    sklearn_version: str = sklearn.__version__


def check_model_features(pipeline: Pipeline) -> list[str]:
    """Features the pipeline was fitted on, in their order; they have to be exactly the ones a request carries.

    Raise ModelFeaturesMismatchError otherwise: the pipeline would not refuse such a request by itself, see
    build_model_input().
    """
    # Remembered by fit() from the columns of the train part, so they are the features of this very model.
    features = list(pipeline.feature_names_in_)
    sent = FeatureVectorChurn.model_fields.keys()
    if set(features) != sent:
        raise ModelFeaturesMismatchError(
            f"Churn model is trained on features {features}, a request carries {sorted(sent)}"
        )
    return features


def save_churn_model(model: TrainedChurnModel, path: Path) -> None:
    """Write the model to path; the previous file is replaced only once the new one is complete.

    The model is dumped to a temporary file next to path and then renamed over it. A rename within one
    directory is atomic, so a crash halfway through never leaves a broken model at path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    joblib.dump(model, tmp_path)
    tmp_path.replace(path)
    logger.info("Saved churn model to %s", path)


def load_churn_model(path: Path) -> TrainedChurnModel:
    """Read a model written by save_churn_model().

    Unpickling runs code stored in the file, so only a file this service wrote itself may be loaded.
    """
    try:
        model = joblib.load(path)
    # Besides OSError, unpickling a damaged or foreign file fails with almost any exception type.
    except Exception as e:
        raise ModelLoadError(f"Cannot read churn model {path}: {e}") from e
    if not isinstance(model, TrainedChurnModel):
        raise ModelLoadError(f"{path} holds {type(model).__name__}, not a churn model")
    # A file saved before a field was added unpickles without it and would fail only once the field is read.
    missing = [f.name for f in fields(TrainedChurnModel) if f.name not in vars(model)]
    if missing:
        raise ModelLoadError(f"{path} holds a churn model of an older format without {', '.join(missing)}, retrain it")
    # A model saved before a feature was added, renamed or dropped loads fine and would fail only on /predict.
    try:
        check_model_features(model.pipeline)
    except ModelFeaturesMismatchError as e:
        raise ModelLoadError(f"{path} holds a churn model that does not fit the requests: {e}; retrain it") from e
    if model.sklearn_version != sklearn.__version__:
        logger.warning(
            "Churn model %s was saved with scikit-learn %s and runs on %s, its predictions may differ",
            path,
            model.sklearn_version,
            sklearn.__version__,
        )
    return model


class ChurnModelStore:
    """The churn model the service currently serves, kept in memory and in the file at path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._model: TrainedChurnModel | None = None
        # /model/train runs in a thread pool: two trainings finishing together must not leave
        # one model on disk and the other in memory.
        self._lock = threading.Lock()

    @property
    def model(self) -> TrainedChurnModel | None:
        return self._model

    def save(self, model: TrainedChurnModel) -> None:
        """Write the model to disk first and serve it only then, so a failed write keeps the previous one."""
        with self._lock:
            save_churn_model(model, self.path)
            self._model = model

    def load(self) -> None:
        """Serve the model saved by a previous run, if there is one; a missing file is not an error."""
        if not self.path.exists():
            logger.info("No saved churn model at %s, the service starts untrained", self.path)
            return
        self._model = load_churn_model(self.path)
        logger.info("Loaded churn model trained at %s from %s", self._model.trained_at.isoformat(), self.path)

    def get_status(self) -> ModelStatus:
        # Read once: a training finishing meanwhile must not mix the fields of two models.
        model = self._model
        if model is None:
            return ModelStatus(trained=False)
        return ModelStatus(
            trained=True,
            trained_at=model.trained_at,
            metrics=model.metrics,
            model_type=model.model_type,
            hyperparameters=describe_hyperparameters(model.pipeline, model.model_type, model.hyperparameters),
        )


@lru_cache
def get_model_store() -> ChurnModelStore:
    """One store for the whole service, so a model trained by one request is seen by the following ones."""
    return ChurnModelStore(MODEL_PATH)
