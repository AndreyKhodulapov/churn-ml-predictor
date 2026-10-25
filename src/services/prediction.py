import pandas as pd

from src.core.exceptions import ModelNotTrainedError
from src.schemas.prediction import FeatureVectorChurn, PredictionResponseChurn
from src.services.model_storage import TrainedChurnModel


def predict_churn(
    model: TrainedChurnModel | None, features: FeatureVectorChurn | list[FeatureVectorChurn]
) -> list[PredictionResponseChurn]:
    """Churn class and class probabilities for every customer, in the order they were sent."""
    if model is None:
        raise ModelNotTrainedError("The service has no trained churn model")
    customers = features if isinstance(features, list) else [features]
    if not customers:
        # sklearn refuses an empty DataFrame; no customers simply means no predictions.
        return []
    # The pipeline picks columns by name, so it needs a DataFrame like the one it was fitted on, not schema objects.
    X = pd.DataFrame([customer.model_dump() for customer in customers])
    predicted_classes = model.pipeline.predict(X)
    # Columns follow pipeline.classes_, which is [0, 1]: training is refused unless both classes are present.
    probabilities = model.pipeline.predict_proba(X)
    return [
        PredictionResponseChurn(churn=churn, probability_no_churn=proba[0], probability_churn=proba[1])
        for churn, proba in zip(predicted_classes, probabilities, strict=True)
    ]
