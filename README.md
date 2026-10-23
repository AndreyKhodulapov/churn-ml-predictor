# churn-ml-predictor

A FastAPI service that predicts customer churn: whether a customer will leave
the service next month (`churn = 1`) or stay (`churn = 0`), based on their
billing, usage and support history.

Current state: the API validates customer features and serves the training
dataset. `POST /predict` echoes the validated features back until a model is
trained.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | health message |
| POST | `/predict` | validate customer features (echoed back for now) |
| GET | `/dataset/preview?n=5` | first `n` rows of the training dataset, `1 <= n <= 100` |
| GET | `/dataset/info` | row and column counts, feature names, churn class distribution |

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

## Dataset

The training data lives in `data/churn_dataset.csv`. It is loaded and
validated once at startup: if the file is missing or a row does not match the
schema, the service fails to start and logs the reason.

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

Example requests:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 29.99, "usage_hours": 42.5, "support_requests": 1,
       "account_age_months": 18, "failed_payments": 0, "region": "europe",
       "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1}'

curl "http://127.0.0.1:8000/dataset/preview?n=3"
curl http://127.0.0.1:8000/dataset/info
```

`/dataset/info` response for the bundled dataset:

```json
{
  "row_count": 2000,
  "column_count": 10,
  "feature_names": ["monthly_fee", "usage_hours", "support_requests", "account_age_months",
                    "failed_payments", "region", "device_type", "payment_method", "autopay_enabled"],
  "target_distribution": {
    "0": {"count": 1597, "share": 0.7985},
    "1": {"count": 403, "share": 0.2015}
  },
  "target_missing": {"count": 0, "share": 0.0}
}
```

`target_missing` counts rows without a `churn` label. They are not a class, so `target_distribution` leaves them out
and its shares, taken over all rows, sum to less than 1 when such rows exist.

## Tests

```bash
make test

# or manually
pip install -r requirements-dev.txt
pytest                      # all tests
pytest tests/unit           # schemas and dataset service
pytest tests/integration    # HTTP API via httpx.AsyncClient
ruff check .
```

## Project layout

```
data/
└── churn_dataset.csv  # training dataset
src/
├── main.py            # FastAPI app, startup, router registration
├── core/              # constants, exceptions, error handlers
├── routers/           # HTTP endpoints
├── services/          # dataset loading and statistics
└── schemas/           # Pydantic models of features, dataset rows and responses
tests/
├── unit/
└── integration/
docs/                  # notes on the underlying concepts
```
