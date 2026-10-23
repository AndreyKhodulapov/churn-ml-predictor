from fastapi import FastAPI

from src.routers import health, prediction, dataset

app = FastAPI(
    title="Churn ML Predictor",
    description="Predicts whether a customer will leave the service next month.",
    version="0.1.0",
)
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(dataset.router)
