from typing import Any

from src.core.exceptions import ModelNotTrainedError

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

# Tuning configs shown as /model/tune request examples in /docs and reused by the tests.
TUNE_EXAMPLES: dict[str, Any] = {
    "logreg_default_grid": {
        "summary": "Default logistic regression grid scored by churn F1",
        "description": "The same as sending no body: C and class_weight from the default grid, 5 folds.",
        "value": {"model_type": "logreg", "scoring": "f1", "cv": 5},
    },
    "logreg_class_weights": {
        "summary": "Logistic regression with balanced and hand-set class weights",
        "description": 'Class weights by class are sent with string keys, like {"0": 1, "1": 4}.',
        "value": {
            "model_type": "logreg",
            "param_grid": {"C": [0.1, 1.0], "class_weight": [None, "balanced", {"0": 1, "1": 4}]},
            "scoring": "f1",
            "cv": 5,
        },
    },
    "logreg_roc_auc": {
        "summary": "Logistic regression regularization scored by ROC-AUC",
        "description": "ROC-AUC scores the predicted probabilities, so class weights barely change it.",
        "value": {"model_type": "logreg", "param_grid": {"C": [0.01, 0.1, 1.0, 10.0]}, "scoring": "roc_auc", "cv": 5},
    },
    "random_forest_depth": {
        "summary": "Random forest depth with balanced class weights",
        "value": {
            "model_type": "random_forest",
            "param_grid": {"n_estimators": [100], "max_depth": [3, 5, None], "class_weight": ["balanced"]},
            "scoring": "f1",
            "cv": 5,
        },
    },
}

# Answer of the service to a request that needs a model before the first /model/train.
MODEL_NOT_TRAINED_ERROR: dict[str, Any] = {
    "code": "model_not_trained",
    "message": str(ModelNotTrainedError()),
    "details": None,
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
        "content": {"application/json": {"example": MODEL_NOT_TRAINED_ERROR}},
    },
}
