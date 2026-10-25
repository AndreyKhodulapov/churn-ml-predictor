import pandas as pd

from src.schemas.prediction import FeatureVectorChurn, PredictionResponseChurn
from src.services.model_storage import TrainedChurnModel


def predict_churn(
    model: TrainedChurnModel | None, 
    X: FeatureVectorChurn | list[FeatureVectorChurn]
) -> list[PredictionResponseChurn]:
    items = X if isinstance(X, list) else [X]
    df = pd.DataFrame([item.model_dump() for item in items])
    if model is None:
        raise ValueError("The service has no trained model")
    churn = model.pipeline.predict(df)
    probability = model.pipeline.predict_proba(df)[:, 1]
    return [PredictionResponseChurn(churn=churn, probability=probability) for churn, probability in zip(churn, probability)]
