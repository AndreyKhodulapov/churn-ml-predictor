import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src.core.exceptions import DatasetLoadError

logger = logging.getLogger(__name__)


async def dataset_unavailable_handler(request: Request, exc: DatasetLoadError) -> JSONResponse:
    # The message names the file path, so it goes to the log only; the next request retries the load.
    logger.warning("Dataset is not available: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Dataset is not available"})


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Starlette re-raises the exception after sending this response, so the traceback still reaches the server log.
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
