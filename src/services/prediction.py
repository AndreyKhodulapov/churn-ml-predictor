import pandas as pd
from sklearn.pipeline import Pipeline

from src.core.exceptions import ModelFeaturesMismatchError, ModelNotTrainedError
from src.schemas.prediction import FeatureVectorChurn, PredictionResponseChurn
from src.services.model_storage import TrainedChurnModel


def build_model_input(pipeline: Pipeline, customers: list[FeatureVectorChurn]) -> pd.DataFrame:
    """One row per customer with the columns the pipeline was fitted on, in the same order.

    The ColumnTransformer steps pick columns by name and would silently accept another order, ignore an extra column
    and fill a missing one as a gap, so the set of features is checked here rather than left to them.
    """
    # Remembered by fit() from the columns of the train part, so they are the features of this very model.
    features = list(pipeline.feature_names_in_)
    sent = FeatureVectorChurn.model_fields.keys()
    if set(features) != sent:
        raise ModelFeaturesMismatchError(
            f"Churn model is trained on features {features}, a request carries {sorted(sent)}"
        )
    return pd.DataFrame([customer.model_dump() for customer in customers], columns=features)


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
    X = build_model_input(model.pipeline, customers)
    predicted_classes = model.pipeline.predict(X)
    # Columns follow pipeline.classes_, which is [0, 1]: training is refused unless both classes are present.
    probabilities = model.pipeline.predict_proba(X)
    return [
        PredictionResponseChurn(churn=churn, probability_no_churn=proba[0], probability_churn=proba[1])
        for churn, proba in zip(predicted_classes, probabilities, strict=True)
    ]
