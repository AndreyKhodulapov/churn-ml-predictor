from typing import Literal

import pandas as pd
from sklearn.pipeline import Pipeline

from src.core.constants import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES
from src.core.exceptions import ModelNotTrainedError, PredictionError
from src.schemas.prediction import FeatureDescription, FeatureSchemaChurn, FeatureVectorChurn, PredictionResponseChurn
from src.services.model_storage import TrainedChurnModel, check_model_features


def build_model_input(pipeline: Pipeline, customers: list[FeatureVectorChurn]) -> pd.DataFrame:
    """One row per customer with the columns the pipeline was fitted on, in the same order.

    The ColumnTransformer steps pick columns by name and would silently accept another order, ignore an extra column
    and fill a missing one as a gap, so the set of features is checked here rather than left to them.
    """
    features = check_model_features(pipeline)
    return pd.DataFrame([customer.model_dump() for customer in customers], columns=features)


def predict_churn(
    model: TrainedChurnModel | None, features: FeatureVectorChurn | list[FeatureVectorChurn]
) -> list[PredictionResponseChurn]:
    """Churn class and class probabilities for every customer, in the order they were sent."""
    if model is None:
        raise ModelNotTrainedError
    customers = features if isinstance(features, list) else [features]
    if not customers:
        # /predict refuses an empty batch, but for the service no customers simply means no predictions,
        # while sklearn refuses an empty DataFrame.
        return []
    try:
        # The pipeline picks columns by name, so it needs a DataFrame like the one it was fitted on, not schema objects.
        X = build_model_input(model.pipeline, customers)
        predicted_classes = model.pipeline.predict(X)
        # Columns follow pipeline.classes_, which is [0, 1]: training is refused unless both classes are present.
        probabilities = model.pipeline.predict_proba(X)
    # The customers have passed validation, so whatever fails here is a fault of the model, not of the request.
    except Exception as e:
        raise PredictionError(f"Churn model failed to predict, batch size {len(customers)}: {e}") from e
    return [
        PredictionResponseChurn(churn=churn, probability_no_churn=proba[0], probability_churn=proba[1])
        for churn, proba in zip(predicted_classes, probabilities, strict=True)
    ]


def describe_features() -> FeatureSchemaChurn:
    """Features of FeatureVectorChurn in the training order, with the types and values /predict accepts.

    Types, values and minimums are read from the JSON Schema of FeatureVectorChurn, so the answer cannot drift from
    the validation of /predict; the order and the kind of every feature come from the training feature lists.
    """
    properties = FeatureVectorChurn.model_json_schema()["properties"]
    return FeatureSchemaChurn(
        features=[
            FeatureDescription(
                name=name,
                kind=feature_kind(name),
                type=properties[name]["type"],
                allowed_values=properties[name].get("enum"),
                minimum=properties[name].get("minimum"),
            )
            for name in FEATURES
        ]
    )


def feature_kind(name: str) -> Literal["numeric", "binary", "categorical"]:
    """How the churn pipeline treats the feature: see build_feature_transformer()."""
    if name in CATEGORICAL_FEATURES:
        return "categorical"
    if name in BINARY_FEATURES:
        return "binary"
    return "numeric"
