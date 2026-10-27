from typing import Annotated

from fastapi import APIRouter, Body, Depends

from src.core.openapi_examples import PREDICT_EXAMPLES, PREDICT_RESPONSES
from src.schemas.prediction import PredictionRequestChurn, PredictionResponseChurn
from src.services.model_storage import ChurnModelStore, get_model_store
from src.services.prediction import predict_churn

router = APIRouter(tags=["Prediction"])


# A plain def: prediction is CPU-bound, FastAPI runs it in a thread pool instead of blocking the event loop.
@router.post(
    "/predict",
    summary="Predict churn of one or several customers",
    # One customer gets a list of one prediction too, so clients always read the answer the same way.
    response_description="Predictions in the order the customers were sent",
    responses=PREDICT_RESPONSES,
)
def predict(
    features: Annotated[PredictionRequestChurn, Body(openapi_examples=PREDICT_EXAMPLES)],
    store: Annotated[ChurnModelStore, Depends(get_model_store)],
) -> list[PredictionResponseChurn]:
    """Predict whether each customer leaves the service next month.

    Send one customer object or a list of them. The answer is always a list with one prediction per customer, in the
    order they were sent: the predicted class `churn` (1 means the customer leaves) and the probabilities of both
    classes, which sum to 1. The class is 1 when `probability_churn` is above 0.5.

    A model trained with the hyperparameter `"class_weight": "balanced"` overstates `probability_churn`: read it as a
    risk score to rank customers by rather than as the real chance of leaving.

    Answers `503` until a model is trained with `POST /model/train`.
    """
    return predict_churn(store.model, features)
