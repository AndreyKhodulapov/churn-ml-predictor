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
