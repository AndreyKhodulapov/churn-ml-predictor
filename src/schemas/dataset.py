from typing import Literal

from pydantic import Field

from src.core.constants import DeviceType, PaymentMethod, Region
from src.schemas.base import BaseSchema


class DatasetRowChurn(BaseSchema):
    """A labeled row of the training dataset: 1 - the customer left, 0 - stayed.

    Unlike FeatureVectorChurn, every value may be missing: gaps are normal in raw data
    and preprocessing decides how to handle them. Every column still has to be present.
    """

    monthly_fee: float | None = Field(ge=0)
    usage_hours: float | None = Field(ge=0)
    support_requests: int | None = Field(ge=0)
    account_age_months: int | None = Field(ge=0)
    failed_payments: int | None = Field(ge=0)
    region: Region | None
    device_type: DeviceType | None
    payment_method: PaymentMethod | None
    autopay_enabled: Literal[0, 1] | None
    churn: Literal[0, 1] | None


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
    # Rows without a label: not a class, so they are reported apart from target_distribution.
    target_missing: ClassDistribution


class SplitDataInfo(BaseSchema):
    """Sizes of the train and test parts and the churn distribution in each of them."""

    train_size: int
    train_distribution: dict[int, ClassDistribution]
    test_size: int
    test_distribution: dict[int, ClassDistribution]
