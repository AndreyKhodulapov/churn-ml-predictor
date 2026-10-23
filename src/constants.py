from pathlib import Path
from typing import Literal

Region = Literal["europe", "asia", "america", "africa"]
DeviceType = Literal["mobile", "desktop", "tablet"]
PaymentMethod = Literal["card", "paypal", "crypto"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "churn_dataset.csv"
