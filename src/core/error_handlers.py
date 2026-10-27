import logging
import re
from http import HTTPStatus
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.core.exceptions import (
    DatasetLoadError,
    DatasetNotTrainableError,
    InvalidHyperparametersError,
    ModelNotTrainedError,
    PredictionError,
)
from src.schemas.errors import ErrorResponse, ValidationErrorDetail

logger = logging.getLogger(__name__)


def error_response(
    status_code: int, code: str, message: str, details: Any = None, headers: dict[str, str] | None = None
) -> JSONResponse:
    """JSON answer in the format every error of the service shares: code, message and details."""
    body = ErrorResponse(code=code, message=message, details=details)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"), headers=headers)


def status_phrase(status_code: int) -> str:
    """Reason phrase of the HTTP status, like Not Found for 404; a status HTTP does not name gets a generic one."""
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "HTTP error"


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    # Raised by routing (an unknown path, a wrong method) and by FastAPI itself, whose HTTPException is a subclass.
    # The code is the status phrase in snake case, like not_found or method_not_allowed. The headers, like Allow
    # of a 405, are kept.
    phrase = status_phrase(exc.status_code)
    code = re.sub(r"\W+", "_", phrase.lower())
    # Starlette sets the detail to the status phrase when none is given; a detail other than a string is context
    # for the client rather than a message, so it goes to details.
    if isinstance(exc.detail, str):
        return error_response(exc.status_code, code, exc.detail, headers=exc.headers)
    return error_response(exc.status_code, code, phrase, exc.detail, exc.headers)


async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI would answer with the pydantic errors as they are. They echo the invalid input, which may be a whole
    # batch of customers, and ctx repeats what the message says, so only where, what and which check are kept.
    details = [
        ValidationErrorDetail(loc=list(error["loc"]), message=error["msg"], type=error["type"]).model_dump()
        for error in exc.errors()
    ]
    return error_response(422, "validation_error", "Request failed validation, see details for every error", details)


async def dataset_unavailable_handler(request: Request, exc: DatasetLoadError) -> JSONResponse:
    # The message names the file path, so it goes to the log only; the next request retries the load.
    logger.warning("Dataset is not available: %s", exc)
    return error_response(503, "dataset_unavailable", "Dataset is not available")


async def dataset_not_trainable_handler(request: Request, exc: DatasetNotTrainableError) -> JSONResponse:
    # The request is fine but the loaded data does not allow it: 409 Conflict with the current state,
    # not 422, which is for bad input. The message names counts and classes only, so it is returned as is.
    logger.warning("Dataset cannot be used for training: %s", exc)
    return error_response(409, "dataset_not_trainable", f"Dataset cannot be used for training: {exc}")


async def invalid_hyperparameters_handler(request: Request, exc: InvalidHyperparametersError) -> JSONResponse:
    # The body passes the schema, but the chosen model refuses its hyperparameters: 422 like any other bad input.
    # The message names the model type and hyperparameters only, so it is returned as is.
    return error_response(422, "invalid_hyperparameters", str(exc))


async def model_not_trained_handler(request: Request, exc: ModelNotTrainedError) -> JSONResponse:
    # The service is up but cannot predict yet: 503, like the missing dataset; it is fixed by /model/train.
    # The message names that fix only, so it is returned as is.
    return error_response(503, "model_not_trained", str(exc))


async def prediction_failed_handler(request: Request, exc: PredictionError) -> JSONResponse:
    # The request is valid, the model fails on it. The cause names model internals, so it goes to the log only, with
    # the traceback: unlike an unhandled exception, a handled one is not re-raised to the server log.
    logger.error("%s", exc, exc_info=exc)
    return error_response(500, "prediction_failed", "Churn model failed to make a prediction")


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Starlette re-raises the exception after sending this response, so the traceback still reaches the server log.
    return error_response(500, "internal_error", "Internal server error")
