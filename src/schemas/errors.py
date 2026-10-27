from typing import Any

from pydantic import Field

from src.schemas.base import BaseSchema


class ErrorResponse(BaseSchema):
    """Body of every error answer of the service, whatever the status."""

    # Stable machine-readable name of the error, for a client to branch on instead of parsing the message.
    code: str = Field(examples=["model_not_trained"])
    # Human-readable reason, safe to show: it never names server paths or holds a traceback.
    message: str = Field(examples=["Churn model is not trained yet, call POST /model/train first"])
    # Structured context of the error, if the code has any, like the fields that failed validation.
    details: Any = Field(default=None, examples=[None])
