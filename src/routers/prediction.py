from typing import Annotated

from fastapi import APIRouter, Body, Depends

from src.core.constants import PREDICT_EXAMPLES
from src.schemas.prediction import FeatureVectorChurn, PredictionResponseChurn
from src.services.model_storage import ChurnModelStore, get_model_store
from src.services.prediction import predict_churn

router = APIRouter(tags=["Prediction"])


# A plain def: prediction is CPU-bound, FastAPI runs it in a thread pool instead of blocking the event loop.
@router.post(
    "/predict",
    summary="Predict churn of one or several customers",
    # One customer gets a list of one prediction too, so clients always read the answer the same way.
    response_description="Predictions in the order the customers were sent",
    responses={
        503: {
            "description": "No churn model is trained yet",
            "content": {
                "application/json": {
                    "example": {"detail": "Churn model is not trained yet, call POST /model/train first"}
                }
            },
        }
    },
)
def predict(
    features: Annotated[FeatureVectorChurn | list[FeatureVectorChurn], Body(openapi_examples=PREDICT_EXAMPLES)],
    store: Annotated[ChurnModelStore, Depends(get_model_store)],
) -> list[PredictionResponseChurn]:
    return predict_churn(store.model, features)
