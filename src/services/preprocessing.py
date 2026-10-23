import logging
from typing import NamedTuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.core.constants import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)


class PreparedData(NamedTuple):
    X: pd.DataFrame
    y: pd.Series
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]


def prepare_data(df: pd.DataFrame) -> PreparedData:
    """Separate the features from the churn target; rows without a label cannot be trained on and are dropped.

    Gaps in the features are kept: build_imputer() fills them and must be fitted on the train part only,
    otherwise test statistics leak into training.
    """
    labeled = df.dropna(subset=[TARGET_COLUMN])
    if dropped := len(df) - len(labeled):
        logger.warning("Dropped %d rows without a %r label", dropped, TARGET_COLUMN)

    X = labeled[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    missing = X.isna().sum()
    if missing.any():
        logger.warning("Missing feature values: %s", missing[missing > 0].to_dict())

    return PreparedData(
        X=X,
        # A column with gaps is read as float; after the drop the labels are whole numbers again.
        y=labeled[TARGET_COLUMN].astype(int),
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


def build_imputer(numeric_features: tuple[str, ...], categorical_features: tuple[str, ...]) -> ColumnTransformer:
    """Fill numeric gaps with the train median; binary and categorical ones with the most frequent train value.

    The median keeps tariffs and counters at values that really occur and ignores outliers.
    /predict rejects missing values, so a separate "missing" category would never be seen at inference.
    """
    continuous = [feature for feature in numeric_features if feature not in BINARY_FEATURES]
    binary = [feature for feature in numeric_features if feature in BINARY_FEATURES]
    imputer = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), continuous),
            ("binary", SimpleImputer(strategy="most_frequent"), binary),
            ("categorical", SimpleImputer(strategy="most_frequent"), categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    # Keep a DataFrame with the original column names instead of a bare numpy array.
    imputer.set_output(transform="pandas")
    return imputer
