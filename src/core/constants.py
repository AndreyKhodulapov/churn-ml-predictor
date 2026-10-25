from pathlib import Path
from typing import Any, Literal

Region = Literal["europe", "asia", "america", "africa"]
DeviceType = Literal["mobile", "desktop", "tablet"]
PaymentMethod = Literal["card", "paypal", "crypto"]
# "balanced" weights each class inversely to its share, so the rare churners count as much as those who stay.
ClassWeight = Literal["balanced"]

TARGET_COLUMN = "churn"
# Set by the meaning of each feature rather than by its dtype; autopay_enabled is a 0/1 flag kept as a number.
NUMERIC_FEATURES = (
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
)
CATEGORICAL_FEATURES = ("region", "device_type", "payment_method")
# Numeric 0/1 flags: a gap is filled with 0 or 1, while a mean or an even-split median would be fractional.
BINARY_FEATURES = ("autopay_enabled",)

TEST_SIZE = 0.2
RANDOM_STATE = 42
# Largest gap between the train and test shares of a churn class that a split may have without a warning.
CLASS_SHARE_TOLERANCE = 0.01

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "churn_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"

# Customers shown as /predict request examples in /docs and reused by the tests.
LOYAL_CUSTOMER: dict[str, Any] = {
    "monthly_fee": 29.99,
    "usage_hours": 42.5,
    "support_requests": 1,
    "account_age_months": 18,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}
RISKY_CUSTOMER: dict[str, Any] = {
    "monthly_fee": 59.99,
    "usage_hours": 3.0,
    "support_requests": 6,
    "account_age_months": 2,
    "failed_payments": 3,
    "region": "asia",
    "device_type": "desktop",
    "payment_method": "crypto",
    "autopay_enabled": 0,
}
PREDICT_EXAMPLES: dict[str, Any] = {
    "one_customer": {"summary": "One customer", "value": LOYAL_CUSTOMER},
    "several_customers": {"summary": "Several customers", "value": [LOYAL_CUSTOMER, RISKY_CUSTOMER]},
}
# Answers of the model trained on the bundled dataset without class weights to the request examples above.
LOYAL_CUSTOMER_PREDICTION: dict[str, Any] = {"churn": 0, "probability_no_churn": 0.957, "probability_churn": 0.043}
RISKY_CUSTOMER_PREDICTION: dict[str, Any] = {"churn": 1, "probability_no_churn": 0.189, "probability_churn": 0.811}

MODEL_NOT_TRAINED_DETAIL = "Churn model is not trained yet, call POST /model/train first"
# Response examples of /predict in /docs; the 200 schema itself is generated from PredictionResponseChurn.
PREDICT_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {
            "application/json": {
                "examples": {
                    "one_customer": {"summary": "One customer", "value": [LOYAL_CUSTOMER_PREDICTION]},
                    "several_customers": {
                        "summary": "Several customers",
                        "value": [LOYAL_CUSTOMER_PREDICTION, RISKY_CUSTOMER_PREDICTION],
                    },
                }
            }
        }
    },
    503: {
        "description": "No churn model is trained yet",
        "content": {"application/json": {"example": {"detail": MODEL_NOT_TRAINED_DETAIL}}},
    },
}
