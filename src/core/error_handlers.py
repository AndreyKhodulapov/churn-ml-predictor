import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src.core.constants import MODEL_NOT_TRAINED_DETAIL
from src.core.exceptions import (
    DatasetLoadError,
    DatasetNotTrainableError,
    InvalidHyperparametersError,
    ModelNotTrainedError,
)

logger = logging.getLogger(__name__)


async def dataset_unavailable_handler(request: Request, exc: DatasetLoadError) -> JSONResponse:
    # The message names the file path, so it goes to the log only; the next request retries the load.
    logger.warning("Dataset is not available: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Dataset is not available"})


async def dataset_not_trainable_handler(request: Request, exc: DatasetNotTrainableError) -> JSONResponse:
    # The request is fine but the loaded data does not allow it: 409 Conflict with the current state,
    # not 422, which is for bad input. The message names counts and classes only, so it is returned as is.
    logger.warning("Dataset cannot be used for training: %s", exc)
    return JSONResponse(status_code=409, content={"detail": f"Dataset cannot be used for training: {exc}"})


async def invalid_hyperparameters_handler(request: Request, exc: InvalidHyperparametersError) -> JSONResponse:
    # The body passes the schema, but the chosen model refuses its hyperparameters: 422 like any other bad input.
    # The message names the model type and hyperparameters only, so it is returned as is.
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def model_not_trained_handler(request: Request, exc: ModelNotTrainedError) -> JSONResponse:
    # The service is up but cannot predict yet: 503, like the missing dataset; it is fixed by /model/train.
    return JSONResponse(status_code=503, content={"detail": MODEL_NOT_TRAINED_DETAIL})


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Starlette re-raises the exception after sending this response, so the traceback still reaches the server log.
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
