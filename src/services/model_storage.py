import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from src.core.constants import MODEL_PATH, ClassWeight
from src.core.exceptions import ModelLoadError
from src.schemas.learning import TrainMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainedChurnModel:
    """A fitted churn pipeline saved together with what is reported about it after a restart."""

    pipeline: Pipeline
    metrics: TrainMetrics
    class_weight: ClassWeight | None = None
    trained_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # A pickled estimator is only guaranteed to behave the same under the scikit-learn version that saved it.
    sklearn_version: str = sklearn.__version__


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


@lru_cache
def get_model_store() -> ChurnModelStore:
    """One store for the whole service, so a model trained by one request is seen by the following ones."""
    return ChurnModelStore(MODEL_PATH)
