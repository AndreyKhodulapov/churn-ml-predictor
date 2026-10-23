from pathlib import Path
from typing import Literal

Region = Literal["europe", "asia", "america", "africa"]
DeviceType = Literal["mobile", "desktop", "tablet"]
PaymentMethod = Literal["card", "paypal", "crypto"]

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "churn_dataset.csv"
