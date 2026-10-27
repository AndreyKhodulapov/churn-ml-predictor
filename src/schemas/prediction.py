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
    """Predicted churn class of one customer and the probabilities of both classes."""

    churn: Literal[0, 1] = Field(examples=[0])
    probability_no_churn: float = Field(ge=0, le=1, examples=[0.83])
    probability_churn: float = Field(ge=0, le=1, examples=[0.17])


class FeatureDescription(BaseSchema):
    """One feature a /predict request carries: how the model treats it and which values it accepts."""

    name: str = Field(examples=["region"])
    # numeric ones are scaled, binary 0/1 flags passed as they are and categorical ones one-hot encoded.
    kind: Literal["numeric", "binary", "categorical"] = Field(examples=["categorical"])
    # JSON Schema type of the value, so a client in any language reads it the same way.
    type: Literal["number", "integer", "string"] = Field(examples=["string"])
    # The only values accepted; None for a numeric feature, which takes any value from minimum on.
    allowed_values: list[str | int] | None = Field(default=None, examples=[["europe", "asia", "america", "africa"]])
    minimum: int | float | None = Field(default=None, examples=[None])


class FeatureSchemaChurn(BaseSchema):
    """Features of a /predict request in the order the churn model is trained on them."""

    features: list[FeatureDescription]
