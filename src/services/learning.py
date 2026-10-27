from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.constants import CATEGORICAL_FEATURES, CLASSIFIERS, NUMERIC_FEATURES, RANDOM_STATE, ModelType
from src.core.exceptions import InvalidHyperparametersError
from src.schemas.learning import TrainMetrics
from src.services.preprocessing import build_imputer, split_numeric_features


def build_feature_transformer(
    numeric_features: tuple[str, ...], categorical_features: tuple[str, ...]
) -> ColumnTransformer:
    """Scale numeric features, pass 0/1 flags as they are and one-hot encode categorical ones.

    Scaling puts fees, hours and counters on one scale, so the regularization of LogisticRegression
    penalizes their coefficients equally. A 0/1 flag is already on that scale.
    """
    scaled, binary = split_numeric_features(numeric_features)
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), scaled),
            ("binary", "passthrough", binary),
            # A category the train part has not seen gets all zeros instead of failing the prediction.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        verbose_feature_names_out=False,
    )


def build_classifier(model_type: ModelType, hyperparameters: Mapping[str, Any]) -> ClassifierMixin:
    """Unfitted classifier of model_type with hyperparameters set over the scikit-learn defaults.

    Only the names are checked here: scikit-learn checks the values when the classifier is fitted.
    """
    classifier = CLASSIFIERS[model_type]()
    valid = classifier.get_params()
    unknown = sorted(set(hyperparameters) - set(valid))
    if unknown:
        raise InvalidHyperparametersError(
            f"Unknown hyperparameters of {model_type}: {', '.join(unknown)}; valid ones are {', '.join(sorted(valid))}"
        )
    # A random forest draws its bootstrap samples at random: a fixed seed makes training on the same split
    # give the same model. A client may still pass its own random_state.
    return classifier.set_params(**{"random_state": RANDOM_STATE, **hyperparameters})


def build_churn_pipeline(
    numeric_features: tuple[str, ...], categorical_features: tuple[str, ...], classifier: ClassifierMixin
) -> Pipeline:
    """Unfitted pipeline from raw features to churn: fill gaps, transform features, classify with classifier.

    Gap filling is a step of the pipeline, so fit() learns its statistics from the train part only
    and the fitted pipeline accepts the same raw rows as the dataset.
    """
    return Pipeline(
        [
            # The imputer returns a DataFrame with the original names, so the next step picks columns by name.
            ("impute", build_imputer(numeric_features, categorical_features)),
            ("transform", build_feature_transformer(numeric_features, categorical_features)),
            ("model", classifier),
        ]
    )


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: ModelType = "logreg",
    hyperparameters: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Fit a new churn pipeline with a model_type classifier on the train part; every call returns a separate model."""
    classifier = build_classifier(model_type, hyperparameters or {})
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, classifier)
    try:
        return pipeline.fit(X_train, y_train)
    # scikit-learn checks hyperparameter values only in fit(): a value out of range raises InvalidParameterError,
    # a subclass of ValueError, and a forbidden combination, like l1_ratio=1 (an l1 penalty) with the lbfgs
    # solver, a plain ValueError. The train part is already validated, so a ValueError here means the
    # hyperparameters do not fit the model.
    except ValueError as e:
        raise InvalidHyperparametersError(f"Cannot train {model_type} with these hyperparameters: {e}") from e


def evaluate_churn_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> TrainMetrics:
    """Accuracy and F1 of the churn class on the test part."""
    predictions = pipeline.predict(X_test)
    # sklearn returns numpy scalars, and f1_score is annotated as an array too for average=None.
    return TrainMetrics(
        accuracy=float(accuracy_score(y_test, predictions)),
        # A model that never predicts churn has no precision to take; its F1 is 0, not a warning.
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
