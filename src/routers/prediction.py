from fastapi import APIRouter

from src.schemas.prediction import FeatureVectorChurn

router = APIRouter(tags=["Prediction"])


@router.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    # No model is trained yet, so the endpoint echoes the validated input back.
    return features
