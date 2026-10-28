from typing import Literal

from src.schemas.base import BaseSchema


class HealthStatus(BaseSchema):
    """What the churn service can do right now: serve the dataset, train a model, predict."""

    # ok when every check below passes; degraded means the service is up but some endpoints answer 503 or 409.
    status: Literal["ok", "degraded"]
    # The dataset is read and validated: the /dataset endpoints answer.
    dataset_loaded: bool
    # The dataset is also split into train and test: /model/train and /model/tune can run.
    dataset_trainable: bool
    # A trained model is served: /predict answers.
    model_available: bool
