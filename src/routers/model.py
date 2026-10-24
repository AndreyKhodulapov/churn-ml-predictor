from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.core.constants import ClassWeight
from src.schemas.learning import TrainMetrics
from src.services.learning import evaluate_churn_model, train_churn_model
from src.services.preprocessing import SplitData, get_split

router = APIRouter(tags=["Model"], prefix="/model")


# A plain def: fitting is CPU-bound, FastAPI runs it in a thread pool instead of blocking the event loop.
@router.post("/train")
def train_model(
    split: Annotated[SplitData, Depends(get_split)],
    class_weight: Annotated[
        ClassWeight | None, Query(description="Class weights; omit to weight every row equally")
    ] = None,
) -> TrainMetrics:
    pipeline = train_churn_model(split.X_train, split.y_train, class_weight)
    return evaluate_churn_model(pipeline, split.X_test, split.y_test)
