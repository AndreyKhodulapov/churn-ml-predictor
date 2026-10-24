import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.error_handlers import internal_error_handler
from src.routers import dataset, health, prediction
from src.services.dataset import get_dataset
from src.services.preprocessing import get_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Load and split the dataset at startup so a missing or broken file stops the service right away.
    get_dataset()
    get_split()
    yield


app = FastAPI(
    title="Churn ML Predictor",
    description="Predicts whether a customer will leave the service next month.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_exception_handler(Exception, internal_error_handler)
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(dataset.router)
