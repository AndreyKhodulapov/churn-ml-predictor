from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from src.core.constants import ModelType
from src.schemas.base import BaseSchema


def restore_class_labels(class_weight: Any) -> Any:
    """class_weight with the churn classes 0 and 1 as int keys again: JSON object keys are always strings.

    Any other value, like "balanced" or None, is returned as is; scikit-learn checks it when the model is fitted.
    """
    if not isinstance(class_weight, dict):
        return class_weight
    return {
        int(label) if isinstance(label, str) and label.isdecimal() else label: weight
        for label, weight in class_weight.items()
    }


class TrainingConfigChurn(BaseSchema):
    """Which churn model to train and the hyperparameters set over its scikit-learn defaults."""

    # A misspelled field would otherwise be dropped silently and the model trained with defaults.
    model_config = ConfigDict(extra="forbid")

    model_type: ModelType = "logreg"
    # Names and values are checked by the chosen model, as they differ from one model type to another.
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("hyperparameters")
    @classmethod
    def restore_class_weight_labels(cls, hyperparameters: dict[str, Any]) -> dict[str, Any]:
        if "class_weight" not in hyperparameters:
            return hyperparameters
        return {**hyperparameters, "class_weight": restore_class_labels(hyperparameters["class_weight"])}


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
    # Without them the metrics are hard to compare: e.g. balanced class weights trade accuracy for churn F1.
    model_type: ModelType | None = None
    # The main hyperparameters of model_type and every requested one, with the values the model was trained with.
    hyperparameters: dict[str, Any] | None = None
