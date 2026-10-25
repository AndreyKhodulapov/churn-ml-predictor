from fastapi import APIRouter, Depends
from typing import Annotated

from src.schemas.prediction import FeatureVectorChurn, PredictionResponseChurn
from src.services.model_storage import ChurnModelStore, get_model_store
from src.services.prediction import predict_churn

router = APIRouter(tags=["Prediction"])


@router.post("/predict")
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn],
    model_storage: Annotated[ChurnModelStore, Depends(get_model_store)]
) -> list[PredictionResponseChurn]:
    model = model_storage.model
    return predict_churn(model, features)
