from typing import Any

from src.core.exceptions import ModelNotTrainedError
from src.schemas.errors import ErrorResponse

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

# Error answers of the service shown as response examples in /docs. The tests send the request each description names
# and check that the service answers exactly this.
MODEL_NOT_TRAINED_ERROR: dict[str, Any] = {
    "code": "model_not_trained",
    "message": str(ModelNotTrainedError()),
    "details": None,
}
PREDICTION_FAILED_ERROR: dict[str, Any] = {
    "code": "prediction_failed",
    "message": "Churn model failed to make a prediction",
    "details": None,
}
DATASET_UNAVAILABLE_ERROR: dict[str, Any] = {
    "code": "dataset_unavailable",
    "message": "Dataset is not available",
    "details": None,
}
NO_LABELED_ROWS_ERROR: dict[str, Any] = {
    "code": "dataset_not_trainable",
    "message": "Dataset cannot be used for training: Dataset has no labeled rows",
    "details": None,
}
CV_TOO_LARGE_ERROR: dict[str, Any] = {
    "code": "dataset_not_trainable",
    "message": "Dataset cannot be used for training: 5-fold cross-validation needs at least 5 customers of each churn "
    "class in the train part, the rarest class has 4",
    "details": None,
}
INVALID_C_ERROR: dict[str, Any] = {
    "code": "invalid_hyperparameters",
    "message": "Cannot train logreg with these hyperparameters: The 'C' parameter of LogisticRegression must be a float "
    "in the range (0.0, inf]. Got 0 instead.",
    "details": None,
}
UNKNOWN_HYPERPARAMETER_ERROR: dict[str, Any] = {
    "code": "invalid_hyperparameters",
    "message": "Unknown hyperparameters of logreg: n_estimators; valid ones are C, class_weight, dual, fit_intercept, "
    "intercept_scaling, l1_ratio, max_iter, n_jobs, penalty, random_state, solver, tol, verbose, warm_start",
    "details": None,
}


def validation_error(*details: dict[str, Any]) -> dict[str, Any]:
    """Answer to a request that fails validation, with one item of details per invalid value."""
    return {
        "code": "validation_error",
        "message": "Request failed validation, see details for every error",
        "details": list(details),
    }


WRONG_TYPE_ERROR = validation_error(
    {
        "loc": ["body", "FeatureVectorChurn", "monthly_fee"],
        "message": "Input should be a valid number, unable to parse string as a number",
        "type": "float_parsing",
    }
)
WRONG_FEATURES_ERROR = validation_error(
    {"loc": ["body", "list[FeatureVectorChurn]", 1, "autopay_enabled"], "message": "Field required", "type": "missing"},
    {
        "loc": ["body", "list[FeatureVectorChurn]", 1, "contract_type"],
        "message": "Extra inputs are not permitted",
        "type": "extra_forbidden",
    },
)
EMPTY_BATCH_ERROR = validation_error(
    {
        "loc": ["body", "list[FeatureVectorChurn]"],
        "message": "List should have at least 1 item after validation, not 0",
        "type": "too_short",
    }
)
UNKNOWN_MODEL_TYPE_ERROR = validation_error(
    {"loc": ["body", "model_type"], "message": "Input should be 'logreg' or 'random_forest'", "type": "literal_error"}
)
TOO_MANY_FITS_ERROR = validation_error(
    {
        "loc": ["body"],
        "message": "Value error, The grid needs 245 fits (combinations times cv folds), at most 200 are allowed",
        "type": "value_error",
    }
)
INVALID_N_ERROR = validation_error(
    {"loc": ["query", "n"], "message": "Input should be greater than or equal to 1", "type": "greater_than_equal"}
)


def error_response_doc(description: str, examples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """An error status of an endpoint in /docs: the ErrorResponse schema and named examples of the answer.

    Setting 422 here also replaces the validation error schema FastAPI would document, which is not what the service
    answers. FastAPI drops None values from the whole OpenAPI document, so a null details is not shown in the
    examples; the schema still shows the field.
    """
    return {"model": ErrorResponse, "description": description, "content": {"application/json": {"examples": examples}}}


DATASET_UNAVAILABLE_RESPONSE = error_response_doc(
    "The dataset file is missing or broken; the reason is in the server log",
    {"dataset_unavailable": {"summary": "Dataset is not available", "value": DATASET_UNAVAILABLE_ERROR}},
)
NO_LABELED_ROWS_EXAMPLE: dict[str, Any] = {
    "summary": "No labeled rows",
    "description": "The dataset has a header only or no row with a churn label.",
    "value": NO_LABELED_ROWS_ERROR,
}
DATASET_NOT_TRAINABLE_RESPONSE = error_response_doc(
    "The dataset is loaded but cannot be split and trained on", {"no_labeled_rows": NO_LABELED_ROWS_EXAMPLE}
)

# Error responses of /predict in /docs next to the 200 examples; the 200 schema is generated from
# PredictionResponseChurn.
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
    422: error_response_doc(
        "A customer fails validation or the list of customers is empty",
        {
            "wrong_type": {
                "summary": "Wrong type of a value",
                "description": '`monthly_fee` of one customer sent as `"abc"`.',
                "value": WRONG_TYPE_ERROR,
            },
            "wrong_features": {
                "summary": "Missing and unknown features",
                "description": "The second customer of a list has no `autopay_enabled` and an unknown `contract_type`.",
                "value": WRONG_FEATURES_ERROR,
            },
            "empty_batch": {
                "summary": "Empty list",
                "description": "`[]` sent as the body.",
                "value": EMPTY_BATCH_ERROR,
            },
        },
    ),
    500: error_response_doc(
        "The model fails to predict for valid customers; the reason is in the server log",
        {"prediction_failed": {"summary": "Model failed", "value": PREDICTION_FAILED_ERROR}},
    ),
    503: error_response_doc(
        "No churn model is trained yet",
        {"model_not_trained": {"summary": "No trained model", "value": MODEL_NOT_TRAINED_ERROR}},
    ),
}

# Error responses of /model/train in /docs.
TRAIN_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: DATASET_NOT_TRAINABLE_RESPONSE,
    422: error_response_doc(
        "The body fails validation, or the model refuses the hyperparameters",
        {
            "unknown_model_type": {
                "summary": "Unknown model type",
                "description": '`{"model_type": "xgboost"}` sent.',
                "value": UNKNOWN_MODEL_TYPE_ERROR,
            },
            "invalid_hyperparameter": {
                "summary": "Hyperparameter value the model refuses",
                "description": '`{"hyperparameters": {"C": 0}}` sent.',
                "value": INVALID_C_ERROR,
            },
            "unknown_hyperparameter": {
                "summary": "Hyperparameter the model does not know",
                "description": '`{"hyperparameters": {"n_estimators": 10}}` sent for the default `logreg`.',
                "value": UNKNOWN_HYPERPARAMETER_ERROR,
            },
        },
    ),
    503: DATASET_UNAVAILABLE_RESPONSE,
}

# Error responses of /model/tune in /docs.
TUNE_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: error_response_doc(
        "The dataset cannot be split and trained on, or a churn class of the train part is smaller than cv",
        {
            "no_labeled_rows": NO_LABELED_ROWS_EXAMPLE,
            "cv_too_large": {
                "summary": "Fewer customers of a class than folds",
                "description": '`{"cv": 5}` sent while the train part has 4 churners.',
                "value": CV_TOO_LARGE_ERROR,
            },
        },
    ),
    422: error_response_doc(
        "The body fails validation, or the model refuses a combination of the grid",
        {
            "too_many_fits": {
                "summary": "Grid too large",
                "description": '`{"param_grid": {"C": [1, 2, ..., 49]}}` sent with the default 5 folds.',
                "value": TOO_MANY_FITS_ERROR,
            },
            "unknown_hyperparameter": {
                "summary": "Hyperparameter the model does not know",
                "description": '`{"param_grid": {"n_estimators": [10]}}` sent for the default `logreg`.',
                "value": UNKNOWN_HYPERPARAMETER_ERROR,
            },
        },
    ),
    503: DATASET_UNAVAILABLE_RESPONSE,
}

# Error responses of the /dataset endpoints in /docs.
PREVIEW_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: error_response_doc(
        "n is out of range or not a number",
        {"invalid_n": {"summary": "n out of range", "description": "`?n=0` sent.", "value": INVALID_N_ERROR}},
    ),
    503: DATASET_UNAVAILABLE_RESPONSE,
}
DATASET_INFO_RESPONSES: dict[int | str, dict[str, Any]] = {503: DATASET_UNAVAILABLE_RESPONSE}
SPLIT_INFO_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: DATASET_NOT_TRAINABLE_RESPONSE,
    503: DATASET_UNAVAILABLE_RESPONSE,
}
