from datetime import datetime

from src.core.constants import ClassWeight
from src.schemas.base import BaseSchema


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
