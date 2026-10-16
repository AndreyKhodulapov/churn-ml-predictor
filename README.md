# churn-ml-predictor

A FastAPI service that predicts customer churn: whether a customer will leave
the service next month (`churn = 1`) or stay (`churn = 0`), based on their
billing, usage and support history.

Current state: the API skeleton and the input schemas are in place. `POST /predict`
validates a customer's features and echoes them back until a model is trained.

## Features

| Field | Type | Values |
|-------|------|--------|
| `monthly_fee` | float | >= 0 |
| `usage_hours` | float | >= 0, hours used last month |
| `support_requests` | int | >= 0 |
| `account_age_months` | int | >= 0 |
| `failed_payments` | int | >= 0 |
| `region` | str | `europe`, `asia`, `america`, `africa` |
| `device_type` | str | `mobile`, `desktop`, `tablet` |
| `payment_method` | str | `card`, `paypal`, `crypto` |
| `autopay_enabled` | int | `0` or `1` |

A training dataset row has the same fields plus the `churn` label (`0` or `1`).

## Run

Requires Python 3.11+.

```bash
make install   # create .venv and install dependencies
make run       # start the service with auto-reload
```

Without make:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Then open:

- http://127.0.0.1:8000/ - health message
- http://127.0.0.1:8000/docs - interactive API docs (Swagger UI)

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 29.99, "usage_hours": 42.5, "support_requests": 1,
       "account_age_months": 18, "failed_payments": 0, "region": "europe",
       "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1}'
```

## Tests

```bash
make test

# or manually
pip install -r requirements-dev.txt
pytest                      # all tests
pytest tests/unit           # schema validation
pytest tests/integration    # HTTP API via httpx.AsyncClient
ruff check .
```

## Project layout

```
src/
├── main.py            # FastAPI app, router registration
├── schemas.py         # Pydantic models of features and dataset rows
├── constants.py       # allowed values of categorical features
└── routers/           # HTTP endpoints
tests/
├── unit/
└── integration/
docs/                  # notes on the underlying concepts
```
