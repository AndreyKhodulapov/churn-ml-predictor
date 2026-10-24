import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ClassWeight
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


def build_churn_pipeline(
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    class_weight: ClassWeight | None = None,
) -> Pipeline:
    """Unfitted pipeline from raw features to churn: fill gaps, transform features, classify.

    Gap filling is a step of the pipeline, so fit() learns its statistics from the train part only
    and the fitted pipeline accepts the same raw rows as the dataset.
    """
    return Pipeline(
        [
            # The imputer returns a DataFrame with the original names, so the next step picks columns by name.
            ("impute", build_imputer(numeric_features, categorical_features)),
            ("transform", build_feature_transformer(numeric_features, categorical_features)),
            ("model", LogisticRegression(class_weight=class_weight)),
        ]
    )


def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series, class_weight: ClassWeight | None = None) -> Pipeline:
    """Fit a new churn pipeline on the train part; every call returns a separate model."""
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, class_weight)
    return pipeline.fit(X_train, y_train)


def evaluate_churn_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> TrainMetrics:
    """Accuracy and F1 of the churn class on the test part."""
    predictions = pipeline.predict(X_test)
    # sklearn returns numpy scalars, and f1_score is annotated as an array too for average=None.
    return TrainMetrics(
        accuracy=float(accuracy_score(y_test, predictions)),
        # A model that never predicts churn has no precision to take; its F1 is 0, not a warning.
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
