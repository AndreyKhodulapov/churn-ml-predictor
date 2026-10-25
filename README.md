# churn-ml-predictor

A FastAPI service that predicts customer churn: whether a customer will leave
the service next month (`churn = 1`) or stay (`churn = 0`), based on their
billing, usage and support history.

Current state: the API validates customer features, serves the training
dataset, trains a logistic regression on it and keeps the trained model on
disk across restarts. `POST /predict` echoes the validated features back until it is wired to
the trained model.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | health message |
| POST | `/predict` | validate customer features (echoed back for now) |
| GET | `/dataset/preview?n=5` | first `n` rows of the training dataset, `1 <= n <= 100` |
| GET | `/dataset/info` | row and column counts, feature names, churn class distribution |
| GET | `/dataset/split-info` | train and test sizes, churn class distribution in each part |
| POST | `/model/train?class_weight=balanced` | train the churn model, save it, return accuracy and churn F1 on the test part; `class_weight` is optional |
| GET | `/model/status` | whether a model is trained, when, with which class weight and its test metrics |

Endpoints that need the dataset answer `503` while it is missing or broken, and `409` with the reason while it
cannot be split and trained on (no labeled rows, a single churn class, too few rows); the service itself keeps running.

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

At startup the dataset is also prepared for training: rows without a `churn`
label are dropped, the features are split 80/20 into train and test with the
churn share kept the same in both parts (`stratify`, fixed `random_state`), and
feature gaps are filled with the train median (numeric) or the most frequent
train value (`autopay_enabled` and categorical). A warning is logged if a churn
class share differs between train and test by more than 1 percentage point.

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
curl http://127.0.0.1:8000/dataset/split-info
curl -X POST http://127.0.0.1:8000/model/train
curl -X POST "http://127.0.0.1:8000/model/train?class_weight=balanced"
curl http://127.0.0.1:8000/model/status
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

`/dataset/split-info` response for the bundled dataset:

```json
{
  "train_size": 1600,
  "train_distribution": {
    "0": {"count": 1278, "share": 0.79875},
    "1": {"count": 322, "share": 0.20125}
  },
  "test_size": 400,
  "test_distribution": {
    "0": {"count": 319, "share": 0.7975},
    "1": {"count": 81, "share": 0.2025}
  }
}
```

`/model/train` responses for the bundled dataset:

```json
{"accuracy": 0.79, "f1": 0.045454545454545456}
```

```json
{"accuracy": 0.5875, "f1": 0.34782608695652173}
```

The first one is trained without class weights: with 20% churners the model almost never predicts churn, so its
accuracy is close to the 0.7975 of always answering "stays" and its F1 is near zero. `class_weight=balanced`
weights churners up: the model catches about half of them at the cost of accuracy.

## Trained model

Every successful `/model/train` saves the model to `models/churn_model.joblib`, together with its test metrics,
training time, class weight and the scikit-learn version it was trained with. The file is written to a temporary name
and renamed, so a crash during the write keeps the previous model intact. A training that fails (`503`, `409`) leaves
the previous model in place.

At startup the service loads the saved model, so `/model/status` reports the same model after a restart. Without the
file the service starts untrained. A file that cannot be read does not stop the service either: the reason is logged
and the service starts untrained until the next `/model/train` writes a new model over it. A model saved with another
scikit-learn version is loaded with a warning, since its predictions may differ.

To check it by hand: train a model, restart the service and compare `/model/status` before and after.

```bash
curl -X POST http://127.0.0.1:8000/model/train
curl http://127.0.0.1:8000/model/status
# stop the service with Ctrl+C and start it again with make run
curl http://127.0.0.1:8000/model/status   # the same trained_at and metrics
```

`/model/status` before the first training:

```json
{"trained": false, "trained_at": null, "metrics": null, "class_weight": null}
```

and after `/model/train?class_weight=balanced` (`trained_at` is in UTC):

```json
{
  "trained": true,
  "trained_at": "2026-10-25T16:15:09.999161Z",
  "metrics": {"accuracy": 0.5875, "f1": 0.34782608695652173},
  "class_weight": "balanced"
}
```

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
models/                # trained model saved by /model/train, not tracked by git
src/
├── main.py            # FastAPI app, startup, router registration
├── core/              # constants, exceptions, error handlers
├── routers/           # HTTP endpoints
├── services/          # dataset loading, statistics, preprocessing, train/test split, model training and storage
└── schemas/           # Pydantic models of features, dataset rows and responses
tests/
├── unit/
└── integration/
docs/                  # notes on the underlying concepts
```
