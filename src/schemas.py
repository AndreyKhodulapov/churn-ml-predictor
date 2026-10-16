from typing import Literal

from pydantic import BaseModel, Field

from src.constants import DeviceType, PaymentMethod, Region


class FeatureVectorChurn(BaseModel):
    """Customer features the churn model makes a prediction from."""

    monthly_fee: float = Field(ge=0, examples=[29.99])
    usage_hours: float = Field(ge=0, examples=[42.5])
    support_requests: int = Field(ge=0, examples=[1])
    account_age_months: int = Field(ge=0, examples=[18])
    failed_payments: int = Field(ge=0, examples=[0])
    region: Region
    device_type: DeviceType
    payment_method: PaymentMethod
    autopay_enabled: Literal[0, 1]


class DatasetRowChurn(FeatureVectorChurn):
    """A labeled row of the training dataset: 1 - the customer left, 0 - stayed."""

    churn: Literal[0, 1]
