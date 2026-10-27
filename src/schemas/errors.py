from typing import Any

from pydantic import Field

from src.schemas.base import BaseSchema


class ValidationErrorDetail(BaseSchema):
    """One value of a request that failed validation, an item of the details of a validation_error."""

    # Where the value is: the part of the request (body, query, path) and the keys and indexes down to it.
    loc: list[str | int] = Field(examples=[["body", "FeatureVectorChurn", "region"]])
    message: str = Field(examples=["Input should be 'europe', 'asia', 'america' or 'africa'"])
    # pydantic error type, stable across messages, like missing or literal_error.
    type: str = Field(examples=["literal_error"])


class ErrorResponse(BaseSchema):
    """Body of every error answer of the service, whatever the status."""

    # Stable machine-readable name of the error, for a client to branch on instead of parsing the message.
    code: str = Field(examples=["model_not_trained"])
    # Human-readable reason, safe to show: it never names server paths or holds a traceback.
    message: str = Field(examples=["Churn model is not trained yet, call POST /model/train first"])
    # Structured context of the error, if the code has any, like the fields that failed validation.
    details: Any = Field(default=None, examples=[None])
