import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.error_handlers import dataset_unavailable_handler, internal_error_handler
from src.core.exceptions import DatasetLoadError
from src.routers import dataset, health, prediction
from src.services.dataset import get_dataset
from src.services.preprocessing import get_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Load and split the dataset at startup so the first request does not wait for it. A missing or broken file
    # does not stop the service: endpoints that need the data answer 503 and retry the load until it succeeds.
    try:
        get_dataset()
        get_split()
    except DatasetLoadError as e:
        logger.warning("Starting without a dataset: %s", e)
    yield


app = FastAPI(
    title="Churn ML Predictor",
    description="Predicts whether a customer will leave the service next month.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_exception_handler(DatasetLoadError, dataset_unavailable_handler)    # type: ignore[arg-type]
app.add_exception_handler(Exception, internal_error_handler)
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(dataset.router)
