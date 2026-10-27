from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Discriminator, Field, Tag

from src.core.constants import DeviceType, PaymentMethod, Region
from src.schemas.base import BaseSchema


class FeatureVectorChurn(BaseSchema):
    """Customer features the churn model makes a prediction from."""

    # A field the model does not know would otherwise be dropped silently, and the client would believe the
    # prediction takes it into account; a misspelled feature also gets named as such, not only as missing.
    # Infinity reaches a float field from a JSON number too large for it, like 1e400, and makes the model fail;
    # NaN would be refused by ge=0 with a misleading reason. Both are refused as not finite instead.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    monthly_fee: float = Field(ge=0, examples=[29.99])
    usage_hours: float = Field(ge=0, examples=[42.5])
    support_requests: int = Field(ge=0, examples=[1])
    account_age_months: int = Field(ge=0, examples=[18])
    failed_payments: int = Field(ge=0, examples=[0])
    region: Region
    device_type: DeviceType
    payment_method: PaymentMethod
    autopay_enabled: Literal[0, 1]


def body_shape(body: Any) -> str:
    """Branch of PredictionRequestChurn to check the body against: a list is a batch, anything else one customer."""
    return "list[FeatureVectorChurn]" if isinstance(body, list) else "FeatureVectorChurn"


# A /predict body: one customer or a list of them. The branch is picked by the shape of the body, so a body that fails
# validation gets the errors of that branch only rather than of both. The tags are the names pydantic gives the
# branches of a plain union, and the loc of every error starts with the tag of its branch. An empty list is refused:
# it is most likely a client bug, like a filter that matched no customers, and an empty answer would hide it.
PredictionRequestChurn = Annotated[
    Annotated[FeatureVectorChurn, Tag("FeatureVectorChurn")]
    | Annotated[list[FeatureVectorChurn], Field(min_length=1), Tag("list[FeatureVectorChurn]")],
    Discriminator(body_shape),
]


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
