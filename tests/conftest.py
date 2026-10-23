from pathlib import Path

import pytest


@pytest.fixture
def customer_features() -> dict:
    return {
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


@pytest.fixture
def write_csv(tmp_path):
    def write(content: str) -> Path:
        path = tmp_path / "dataset.csv"
        path.write_text(content)
        return path

    return write


@pytest.fixture
def csv_header(customer_features) -> str:
    return ",".join([*customer_features, "churn"])


@pytest.fixture
def make_csv_row(customer_features):
    def make(churn: int = 1) -> str:
        return ",".join(str(value) for value in [*customer_features.values(), churn])

    return make


@pytest.fixture
def csv_row(make_csv_row) -> str:
    return make_csv_row()
