import logging
from typing import NamedTuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.core.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.core.exceptions import DatasetNotTrainableError

logger = logging.getLogger(__name__)


class PreparedData(NamedTuple):
    X: pd.DataFrame
    y: pd.Series
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]


class SplitData(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_data(df: pd.DataFrame) -> PreparedData:
    """Separate the features from the churn target; rows without a label cannot be trained on and are dropped.

    Gaps in the features are kept: the imputer step of the churn pipeline (build_imputer()) fills them
    and is fitted on the train part only, otherwise test statistics would leak into training.
    """
    labeled = df.dropna(subset=[TARGET_COLUMN])
    if dropped := len(df) - len(labeled):
        logger.warning("Dropped %d rows without a %r label", dropped, TARGET_COLUMN)

    X = labeled[list(FEATURES)]
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


def split_numeric_features(numeric_features: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Numeric features that are filled with the median and scaled, and 0/1 flags that are neither.

    The first group mixes continuous values (fees, hours) with discrete counters, so it is named by what is done to it.
    """
    scaled = [feature for feature in numeric_features if feature not in BINARY_FEATURES]
    binary = [feature for feature in numeric_features if feature in BINARY_FEATURES]
    return scaled, binary


def build_imputer(numeric_features: tuple[str, ...], categorical_features: tuple[str, ...]) -> ColumnTransformer:
    """Fill numeric gaps with the train median; binary and categorical ones with the most frequent train value.

    The median keeps tariffs and counters at values that really occur and ignores outliers.
    /predict rejects missing values, so a separate "missing" category would never be seen at inference.
    """
    scaled, binary = split_numeric_features(numeric_features)
    imputer = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), scaled),
            ("binary", SimpleImputer(strategy="most_frequent"), binary),
            ("categorical", SimpleImputer(strategy="most_frequent"), categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    # Keep a DataFrame with the original column names instead of a bare numpy array.
    imputer.set_output(transform="pandas")
    return imputer


def check_trainable(y: pd.Series) -> None:
    """Reject a target that a churn classifier cannot be trained on.

    A single class still splits fine and only fails later in fit(), so it is caught here with a clear reason.
    """
    if y.empty:
        raise DatasetNotTrainableError("Dataset has no labeled rows")
    if y.nunique() < 2:
        raise DatasetNotTrainableError(f"Dataset has only churn class {y.iloc[0]}, a classifier needs both")


def split_data(prepared: PreparedData) -> SplitData:
    """Split into train and test keeping the churn share of the whole dataset in both parts; gaps stay as they are."""
    check_trainable(prepared.y)
    try:
        parts = train_test_split(
            prepared.X,
            prepared.y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=prepared.y,
        )
    except ValueError as e:
        # Too few rows of a class or too few for the test part to hold every class; sklearn names which.
        raise DatasetNotTrainableError(str(e)) from e
    return SplitData(*parts)
