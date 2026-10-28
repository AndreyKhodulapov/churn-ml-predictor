import logging
import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from src.core.constants import HISTORY_PATH, MAX_HISTORY_RECORDS, ChurnMetric, ModelType
from src.core.exceptions import HistoryLoadError
from src.schemas.learning import ModelMetricsChurn, TrainingRecord
from src.services.learning import describe_hyperparameters
from src.services.model_storage import TrainedChurnModel

logger = logging.getLogger(__name__)

# The history file is a JSON list of records, oldest first: the order they were appended in.
RECORDS = TypeAdapter(list[TrainingRecord])


def describe_training(model: TrainedChurnModel) -> TrainingRecord:
    """Record of the training that produced model, with the hyperparameter values it was trained with."""
    return TrainingRecord(
        trained_at=model.trained_at,
        model_type=model.model_type,
        hyperparameters=describe_hyperparameters(model.pipeline, model.model_type, model.hyperparameters),
        metrics=model.metrics,
    )


def save_history(records: Sequence[TrainingRecord], path: Path) -> None:
    """Write the records to path; the previous file is replaced only once the new one is complete.

    As with the model file, the records go to a temporary file next to path that is then renamed over it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(RECORDS.dump_json(list(records), indent=2))
    tmp_path.replace(path)


def load_history(path: Path) -> list[TrainingRecord]:
    """Read the records written by save_history(); unlike a pickle, a JSON file runs no code when read."""
    try:
        return RECORDS.validate_json(path.read_bytes())
    except (OSError, ValidationError) as e:
        raise HistoryLoadError(f"Cannot read training history {path}: {e}") from e


class TrainingHistory:
    """Successful trainings of the churn model, kept in memory and in the file at path."""

    def __init__(self, path: Path, max_records: int = MAX_HISTORY_RECORDS) -> None:
        self.path = path
        self.max_records = max_records
        # Replaced as a whole rather than changed in place, so a reader never sees an append halfway through.
        self._records: tuple[TrainingRecord, ...] = ()
        # /model/train runs in a thread pool: two appends at once must not lose one of the records.
        self._lock = threading.Lock()

    def append(self, record: TrainingRecord) -> None:
        """Write the history with record to disk first and keep it only then, so a failed write keeps the previous one.

        The oldest records beyond max_records are dropped.
        """
        with self._lock:
            records = (*self._records, record)[-self.max_records :]
            save_history(records, self.path)
            self._records = records

    def load(self) -> None:
        """Keep the records saved by a previous run, if there are any; a missing file is not an error."""
        if not self.path.exists():
            logger.info("No saved training history at %s, the service starts with an empty one", self.path)
            return
        self._records = tuple(load_history(self.path)[-self.max_records :])
        logger.info("Loaded %d training records from %s", len(self._records), self.path)

    def records(self, model_type: ModelType | None = None) -> list[TrainingRecord]:
        """Recorded trainings, newest first; only those of model_type if it is given."""
        return [record for record in reversed(self._records) if model_type is None or record.model_type == model_type]


def summarize_history(
    history: TrainingHistory, model_type: ModelType | None, limit: int, best_by: ChurnMetric | None
) -> ModelMetricsChurn:
    """The latest training, the limit newest ones and the best one by best_by; of model_type only if it is given."""
    # Read once: a training recorded meanwhile must not make latest differ from the first one of history.
    records = history.records(model_type)
    best = None
    if best_by is not None and records:
        # max() keeps the first of equal scores, so a tie goes to the newest training.
        best = max(records, key=lambda record: getattr(record.metrics, best_by))
    return ModelMetricsChurn(latest=records[0] if records else None, history=records[:limit], best=best)


@lru_cache
def get_training_history() -> TrainingHistory:
    """One history for the whole service, so a training recorded by one request is seen by the following ones."""
    return TrainingHistory(HISTORY_PATH)
