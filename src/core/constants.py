from pathlib import Path
from typing import Any, Literal

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

Region = Literal["europe", "asia", "america", "africa"]
DeviceType = Literal["mobile", "desktop", "tablet"]
PaymentMethod = Literal["card", "paypal", "crypto"]
ModelType = Literal["logreg", "random_forest"]
CLASSIFIERS: dict[ModelType, type[ClassifierMixin]] = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}
# Hyperparameters that shape each model the most: /model/status shows their values even when training left them
# at the scikit-learn defaults.
MAIN_HYPERPARAMETERS: dict[ModelType, tuple[str, ...]] = {
    "logreg": ("C", "l1_ratio", "solver", "max_iter", "class_weight"),
    "random_forest": ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"),
}

# Metrics /model/tune can pick the best combination by: the ones /model/train reports, plus roc_auc, which scores
# the predicted probabilities rather than the labels.
TuningScoring = Literal["accuracy", "f1", "roc_auc"]
# Grids /model/tune searches when the client sends none. A forest fit costs about 6 logreg ones, so its grid leaves
# n_estimators at the default: more trees only make the answer stabler and slower, they trade off no metric.
DEFAULT_PARAM_GRIDS: dict[ModelType, dict[str, list[Any]]] = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    "random_forest": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5], "class_weight": [None, "balanced"]},
}
# Largest number of combinations times folds one /model/tune call may fit: the search runs while the client waits,
# and 200 fits of a default forest take about a minute on the bundled dataset.
MAX_TUNING_FITS = 200
# Best combinations /model/tune returns next to the best one, to show how close the runners-up are.
TUNING_TOP_CANDIDATES = 5

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
# Columns of the training data in this order: the fitted pipeline remembers them as feature_names_in_.
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
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

MODEL_NOT_TRAINED_DETAIL = "Churn model is not trained yet, call POST /model/train first"
