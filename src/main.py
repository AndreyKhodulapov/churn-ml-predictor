import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from src.api import dataset, health, model, prediction
from src.core.config import LOG_LEVEL
from src.core.error_handlers import (
    dataset_not_trainable_handler,
    dataset_unavailable_handler,
    http_error_handler,
    internal_error_handler,
    invalid_hyperparameters_handler,
    model_not_trained_handler,
    prediction_failed_handler,
    request_validation_handler,
)
from src.core.exceptions import (
    DatasetLoadError,
    DatasetNotTrainableError,
    HistoryLoadError,
    InvalidHyperparametersError,
    ModelLoadError,
    ModelNotTrainedError,
    PredictionError,
)
from src.services.dataset import get_dataset
from src.services.model_storage import get_model_store
from src.services.split import get_split
from src.services.training_history import get_training_history

logger = logging.getLogger(__name__)


def configure_logging(level_name: str) -> None:
    """Log to stderr at the level LOG_LEVEL names; an unknown name is reported and falls back to INFO.

    A typo in the environment of a container must not stop the service for the sake of a log level.
    """
    level = logging.getLevelNamesMapping().get(level_name)
    logging.basicConfig(
        level=logging.INFO if level is None else level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if level is None:
        logger.warning("Unknown LOG_LEVEL %r, logging at INFO; use DEBUG, INFO, WARNING, ERROR or CRITICAL", level_name)


configure_logging(LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Load and split the dataset at startup so the first request does not wait for it. A missing, broken or too small
    # dataset does not stop the service: endpoints that need it answer 503 or 409 and retry until it is fixed.
    try:
        get_dataset()
        get_split()
    except (DatasetLoadError, DatasetNotTrainableError) as e:
        logger.warning("Starting without a usable dataset: %s", e)
    # Serve the model trained before the restart. A damaged file does not stop the service either:
    # /model/status reports no model until /model/train saves a new one over it.
    try:
        get_model_store().load()
    except ModelLoadError as e:
        logger.warning("Starting without a trained churn model: %s", e)
    # A damaged history file does not stop the service either; the next training writes a new one over it.
    try:
        get_training_history().load()
    except HistoryLoadError as e:
        logger.warning("Starting with an empty training history: %s", e)
    yield


app = FastAPI(
    title="Churn ML Predictor",
    description="Predicts whether a customer will leave the service next month.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_exception_handler(HTTPException, http_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, request_validation_handler)  # type: ignore[arg-type]
app.add_exception_handler(DatasetLoadError, dataset_unavailable_handler)  # type: ignore[arg-type]
app.add_exception_handler(DatasetNotTrainableError, dataset_not_trainable_handler)  # type: ignore[arg-type]
app.add_exception_handler(InvalidHyperparametersError, invalid_hyperparameters_handler)  # type: ignore[arg-type]
app.add_exception_handler(ModelNotTrainedError, model_not_trained_handler)  # type: ignore[arg-type]
app.add_exception_handler(PredictionError, prediction_failed_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, internal_error_handler)
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(dataset.router)
app.include_router(model.router)
