import math
from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from src.core.constants import DEFAULT_PARAM_GRIDS, MAX_TUNING_FITS, ModelType, TuningScoring
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
    # Scores the churn probabilities rather than the labels: how well the model ranks churners above the others,
    # whatever the 0.5 threshold makes of them.
    roc_auc: float


class ModelStatus(BaseSchema):
    """Whether the service has a trained churn model, when it was trained and how it scored on the test part."""

    trained: bool
    trained_at: datetime | None = None
    metrics: TrainMetrics | None = None
    # Without them the metrics are hard to compare: e.g. balanced class weights trade accuracy for churn F1.
    model_type: ModelType | None = None
    # The main hyperparameters of model_type and every requested one, with the values the model was trained with.
    hyperparameters: dict[str, Any] | None = None


class TuningConfigChurn(BaseSchema):
    """Which churn model to tune, the hyperparameter values to try and the metric to pick the best combination by."""

    model_config = ConfigDict(extra="forbid")

    model_type: ModelType = "logreg"
    # Every combination of the listed values is tried; None searches DEFAULT_PARAM_GRIDS of model_type.
    # Names and values are checked by the chosen model, as in TrainingConfigChurn.
    param_grid: dict[str, Annotated[list[Any], Field(min_length=1)]] | None = Field(default=None, min_length=1)
    scoring: TuningScoring = "f1"
    # Number of cross-validation folds of the train part; each combination is fitted once per fold.
    cv: int = Field(default=5, ge=2, le=10)

    @field_validator("param_grid")
    @classmethod
    def restore_class_weight_labels(cls, param_grid: dict[str, list[Any]] | None) -> dict[str, list[Any]] | None:
        if param_grid is None or "class_weight" not in param_grid:
            return param_grid
        return {**param_grid, "class_weight": [restore_class_labels(value) for value in param_grid["class_weight"]]}

    @property
    def grid(self) -> dict[str, list[Any]]:
        """The grid to search: the requested one or the default of model_type."""
        grid = DEFAULT_PARAM_GRIDS[self.model_type] if self.param_grid is None else self.param_grid
        return {name: list(values) for name, values in grid.items()}

    @model_validator(mode="after")
    def check_fit_count(self) -> Self:
        fits = math.prod(len(values) for values in self.grid.values()) * self.cv
        if fits > MAX_TUNING_FITS:
            raise ValueError(
                f"The grid needs {fits} fits (combinations times cv folds), at most {MAX_TUNING_FITS} are allowed"
            )
        return self


class TuningCandidate(BaseSchema):
    """One combination of the grid and its score on the cross-validation folds of the train part."""

    hyperparameters: dict[str, Any]
    mean_score: float
    std_score: float


class TuningResultChurn(BaseSchema):
    """The best combination of the grid search, ready to be sent to POST /model/train as is."""

    best_config: TrainingConfigChurn
    scoring: TuningScoring
    # Mean over the folds of the train part, so it differs from the test metrics /model/train reports.
    best_score: float
    # The best combinations, best first; the first one is best_config.
    candidates: list[TuningCandidate]
