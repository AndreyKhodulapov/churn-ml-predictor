from typing import Literal

from pydantic import Field

from src.core.constants import DeviceType, PaymentMethod, Region
from src.schemas.base import BaseSchema


class FeatureVectorChurn(BaseSchema):
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


class PredictionResponseChurn(BaseSchema):
    """Churn prediction response."""

    churn: Literal[0, 1]
    probability: float
