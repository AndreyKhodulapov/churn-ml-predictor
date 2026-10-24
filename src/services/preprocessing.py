import logging
from functools import lru_cache
from typing import NamedTuple, cast

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.core.constants import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.services.dataset import get_dataset

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


def fill_data_gaps(
    split: SplitData, numeric_features: tuple[str, ...], categorical_features: tuple[str, ...]
) -> tuple[SplitData, ColumnTransformer]:
    """Fill feature gaps in both parts with statistics of the train part; the fitted imputer is returned too."""
    X_train, X_test, y_train, y_test = split
    imputer = build_imputer(numeric_features, categorical_features)
    # set_output(transform="pandas") makes both calls return a DataFrame; sklearn's annotations do not know it.
    X_train = cast(pd.DataFrame, imputer.fit_transform(X_train))  # train on the train part (fitting)
    X_test = cast(pd.DataFrame, imputer.transform(X_test))  # only fill gaps in the test part (no leak!)

    return SplitData(X_train, X_test, y_train, y_test), imputer


def split_data(prepared: PreparedData) -> SplitData:
    """Split into train and test keeping the churn share of the whole dataset in both parts; gaps stay as they are."""
    return SplitData(
        *train_test_split(
            prepared.X,
            prepared.y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=prepared.y,  # for unbalanced data, at least 2 rows of each class in dataset!
        )
    )


@lru_cache
def get_split() -> SplitData:
    """Prepare, split and fill the default dataset once and reuse the result for every request."""
    prepared = prepare_data(get_dataset().get_dataframe())
    split, _ = fill_data_gaps(split_data(prepared), prepared.numeric_features, prepared.categorical_features)
    return split
