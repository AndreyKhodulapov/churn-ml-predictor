from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.core.constants import ClassWeight
from src.schemas.learning import ModelStatus, TrainMetrics
from src.services.model import train_and_save_churn_model
from src.services.model_storage import ChurnModelStore, get_model_store
from src.services.preprocessing import SplitData, get_split

router = APIRouter(tags=["Model"], prefix="/model")


# A plain def: fitting is CPU-bound, FastAPI runs it in a thread pool instead of blocking the event loop.
@router.post("/train")
def train_model(
    split: Annotated[SplitData, Depends(get_split)],
    store: Annotated[ChurnModelStore, Depends(get_model_store)],
    class_weight: Annotated[
        ClassWeight | None, Query(description="Class weights; omit to weight every row equally")
    ] = None,
) -> TrainMetrics:
    return train_and_save_churn_model(split, class_weight, store)


@router.get("/status")
def get_model_status(store: Annotated[ChurnModelStore, Depends(get_model_store)]) -> ModelStatus:
    return store.get_status()
