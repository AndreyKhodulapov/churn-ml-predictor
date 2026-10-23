from typing import Literal

from src.schemas.base import BaseSchema
from src.schemas.prediction import FeatureVectorChurn


class DatasetRowChurn(FeatureVectorChurn):
    """A labeled row of the training dataset: 1 - the customer left, 0 - stayed."""

    churn: Literal[0, 1]


class ClassDistribution(BaseSchema):
    """How many rows belong to a target class and which share of the dataset they make."""

    count: int
    share: float


class DatasetInfo(BaseSchema):
    row_count: int
    column_count: int
    feature_names: list[str]
    # Keyed by the target class; JSON turns the keys into strings ("0", "1").
    target_distribution: dict[int, ClassDistribution]
