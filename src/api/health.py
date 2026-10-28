from typing import Annotated

from fastapi import APIRouter, Depends

from src.schemas.health import HealthStatus
from src.services.health import check_health
from src.services.model_storage import ChurnModelStore, get_model_store

router = APIRouter(tags=["health"])


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "ml churn service is running"}


@router.get(
    "/health",
    summary="Report what the service can do right now",
    response_description="Whether the dataset is loaded and trainable and a model is served",
)
def get_health(store: Annotated[ChurnModelStore, Depends(get_model_store)]) -> HealthStatus:
    """Report whether the dataset is loaded, whether it can be trained on and whether a churn model is served.

    `status` is `ok` when all three hold and `degraded` otherwise. The answer is `200` either way: a service without a
    model is up and can train one, so a container healthcheck must not restart it. Read `status` and the fields to
    know which endpoints work: without `dataset_loaded` the `/dataset` ones answer `503`, without `dataset_trainable`
    `/model/train` and `/model/tune` answer `503` or `409`, without `model_available` `/predict` answers `503`.

    The check loads nothing: a dataset that failed to load is retried by the next request that needs it.
    """
    return check_health(store)
