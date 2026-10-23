from typing import Literal

from src.schemas.prediction import FeatureVectorChurn


class DatasetRowChurn(FeatureVectorChurn):
    """A labeled row of the training dataset: 1 - the customer left, 0 - stayed."""

    churn: Literal[0, 1]
