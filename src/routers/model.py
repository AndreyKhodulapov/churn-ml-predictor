from typing import Annotated

from fastapi import APIRouter, Body, Depends

from src.core.openapi_examples import TRAIN_EXAMPLES
from src.schemas.learning import ModelStatus, TrainingConfigChurn, TrainMetrics
from src.services.model import train_and_save_churn_model
from src.services.model_storage import ChurnModelStore, get_model_store
from src.services.preprocessing import SplitData, get_split

router = APIRouter(tags=["Model"], prefix="/model")


# A plain def: fitting is CPU-bound, FastAPI runs it in a thread pool instead of blocking the event loop.
@router.post(
    "/train",
    summary="Train the churn model and serve it",
    response_description="Accuracy and churn F1 of the new model on the test part",
)
def train_model(
    split: Annotated[SplitData, Depends(get_split)],
    store: Annotated[ChurnModelStore, Depends(get_model_store)],
    # Without a body the service trains logreg with the scikit-learn defaults.
    config: Annotated[TrainingConfigChurn, Body(default_factory=TrainingConfigChurn, openapi_examples=TRAIN_EXAMPLES)],
) -> TrainMetrics:
    """Train a churn model on the train part of the dataset, score it on the test part, save it and serve it.

    The body is optional. `model_type` is `logreg` (logistic regression) or `random_forest`, and `hyperparameters`
    are set over the scikit-learn defaults of that model under their scikit-learn names, for example
    `{"class_weight": "balanced"}`. Without a body the service trains a logistic regression with the defaults.
    `GET /model/status` shows the main hyperparameters the model was trained with, the defaults included.

    Answers `422` when the model does not know a hyperparameter or refuses its value, `409` when the dataset cannot be
    trained on and `503` while it is not available. A failed training keeps the previous model.
    """
    return train_and_save_churn_model(split, config, store)


@router.get("/status")
def get_model_status(store: Annotated[ChurnModelStore, Depends(get_model_store)]) -> ModelStatus:
    return store.get_status()
