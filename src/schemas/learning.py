from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field

from src.core.constants import ClassWeight, ModelType
from src.schemas.base import BaseSchema


class TrainingConfigChurn(BaseSchema):
    """Which churn model to train and the hyperparameters set over its scikit-learn defaults."""

    # A misspelled field would otherwise be dropped silently and the model trained with defaults.
    model_config = ConfigDict(extra="forbid")

    model_type: ModelType = "logreg"
    # Names and values are checked by the chosen model, as they differ from one model type to another.
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainMetrics(BaseSchema):
    """Quality of the trained churn model on the test part."""

    accuracy: float
    # F1 of the churn class: accuracy alone looks good for a model that never predicts churn.
    f1: float


class ModelStatus(BaseSchema):
    """Whether the service has a trained churn model, when it was trained and how it scored on the test part."""

    trained: bool
    trained_at: datetime | None = None
    metrics: TrainMetrics | None = None
    # Without it the metrics are hard to compare: balanced weights trade accuracy for churn F1.
    class_weight: ClassWeight | None = None
