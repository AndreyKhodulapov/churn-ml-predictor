from typing import Any

from src.core.constants import MODEL_NOT_TRAINED_DETAIL

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

# Training configs shown as /model/train request examples in /docs and reused by the tests.
TRAIN_EXAMPLES: dict[str, Any] = {
    "logreg_defaults": {
        "summary": "Logistic regression with scikit-learn defaults",
        "description": "The same as sending no body.",
        "value": {"model_type": "logreg", "hyperparameters": {}},
    },
    "logreg_balanced": {
        "summary": "Logistic regression with balanced class weights",
        "description": "Finds more churners at the cost of accuracy.",
        "value": {"model_type": "logreg", "hyperparameters": {"class_weight": "balanced"}},
    },
    "random_forest": {
        "summary": "Random forest of shallow trees with balanced class weights",
        "value": {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 300, "max_depth": 5, "class_weight": "balanced"},
        },
    },
}

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
