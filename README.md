# churn-ml-predictor

A FastAPI service that predicts customer churn: whether a customer will leave the service next month (`churn = 1`) or
stay (`churn = 0`), based on their billing, usage and support history. It trains the model on a CSV dataset, keeps it
across restarts and serves predictions over HTTP, locally or in Docker.

[Purpose](#purpose) · [Dataset](#dataset-churn_datasetcsv) · [Run locally](#run-locally) ·
[Run in Docker](#run-in-docker) · [Train and predict](#train-and-predict) · [Endpoints](#endpoints) ·
[Health and logs](#health-and-logs) · [Tests](#tests) · [Project layout](#project-layout) ·
[Documentation](#documentation)

## Purpose

A customer who is about to leave can still be kept, with a discount, a call or a fix of what went wrong, but an offer
sent to every customer costs too much and mostly reaches those who would stay anyway. The service tells, for each
customer, how likely they are to leave next month, so that retention goes to those at risk first.

It covers the whole loop on one dataset:

1. load and validate the training data and show what is in it: `/dataset/*`;
2. train a logistic regression or a random forest with the requested hyperparameters and score it on a held-out test
   part, `/model/train`, or search a grid for the best hyperparameters, `/model/tune`;
3. keep the trained model across restarts and record every training to compare configs: `/model/status`,
   `/model/metrics`;
4. predict churn of one customer or a batch with the served model: `/predict`;
5. report what it can do right now, `/health`, and log the key events.

About 20% of the customers in the data leave, so a model that never predicts churn is already 80% accurate. That is
why every training reports F1 of the churn class and ROC-AUC next to accuracy, see
[docs/api.md](docs/api.md#training-config).

## Dataset: churn_dataset.csv

The training data lives in `data/churn_dataset.csv`: one row per customer, comma-separated, with a header row. Columns
are found by name, so their order does not matter.

| Column | Type | Accepted values | In the bundled dataset |
|--------|------|-----------------|------------------------|
| `monthly_fee` | float | >= 0, finite | 9.99, 19.99, 29.99 or 49.99 |
| `usage_hours` | float | >= 0, finite, hours used last month | 0 to 52.07 |
| `support_requests` | int | >= 0 | 0 to 4 |
| `account_age_months` | int | >= 0 | 1 to 39 |
| `failed_payments` | int | >= 0 | 0 to 2 |
| `region` | str | `europe`, `asia`, `america`, `africa` | all four |
| `device_type` | str | `mobile`, `desktop`, `tablet` | all three |
| `payment_method` | str | `card`, `paypal`, `crypto` | all three |
| `autopay_enabled` | int | `0` or `1` | both |
| `churn` | int | `1` - the customer left the next month, `0` - stayed | 403 of 2000 rows are `1` |

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
19.99,21.48,2,1,0,america,mobile,card,1,0
```

- Every column has to be present, but any cell may be empty. A row without `churn` cannot be trained on and is
  dropped; an empty feature is filled from the train part. The bundled dataset has no gaps.
- A value of a wrong type, a negative number, infinity or an unknown category makes the whole file invalid, and so
  does a missing column.
- A column not in the table is loaded and listed by `/dataset/info`, but the model is not trained on it.

The file is loaded and validated once at startup and split 80/20 into train and test, with the churn share kept the
same in both parts. If it is missing, empty or invalid, the service still starts and logs the reason; the endpoints
that need the dataset answer `503` and retry the load on every request until the file is fixed.

What the service has loaded:

```bash
curl "http://127.0.0.1:8000/dataset/preview?n=3"   # the first rows as the service validated them
curl http://127.0.0.1:8000/dataset/info
curl http://127.0.0.1:8000/dataset/split-info
```

The responses for the bundled dataset: [docs/api.md](docs/api.md#dataset).

## Run locally

Requires Python 3.11+.

```bash
make install           # create .venv and install dependencies
cp .env.example .env   # optional: settings of the service, see Health and logs
make run               # start the service with auto-reload
```

Without make:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Then open:

- http://127.0.0.1:8000/health - what the service can do right now
- http://127.0.0.1:8000/docs - interactive API docs (Swagger UI)

Next: [train a model and predict](#train-and-predict).

## Run in Docker

```bash
make docker-build   # build the churn-ml-predictor image
make docker-run     # start it on port 8000, keeping models/ in the churn-models volume
```

Without make:

```bash
docker build -t churn-ml-predictor .
docker run --rm --name churn-ml-predictor -p 8000:8000 -v churn-models:/app/models churn-ml-predictor
```

The service answers at the same addresses as above, http://127.0.0.1:8000/health and http://127.0.0.1:8000/docs.

- The trained model and the training history go to the named volume `churn-models` mounted on `/app/models`, so a
  new container serves the model the previous one trained. `docker volume rm churn-models` starts from scratch.
- Logs go to `docker logs churn-ml-predictor`; `LOG_LEVEL` in `.env` (`make docker-run` passes it) or
  `-e LOG_LEVEL=DEBUG` on `docker run` changes their level.
- A `HEALTHCHECK` calls `/health` every 30 seconds; `docker ps` shows the container as `healthy` once the service
  answers, with or without a trained model.

What goes into the image and why it is built this way: [docs/docker.md](docs/docker.md).

## Train and predict

With the service running, locally or in Docker. Before the first training it is up but cannot predict yet:

```bash
curl http://127.0.0.1:8000/health
# {"status": "degraded", "dataset_loaded": true, "dataset_trainable": true, "model_available": false}
```

Train the default model, a logistic regression with the scikit-learn defaults, and get its metrics on the test part:

```bash
curl -X POST http://127.0.0.1:8000/model/train
# {"accuracy": 0.79, "f1": 0.045454545454545456, "roc_auc": 0.6085374821007005}
```

Or choose the model and its hyperparameters, see [Training config](docs/api.md#training-config):

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 300, "max_depth": 5, "class_weight": "balanced"}}'
# {"accuracy": 0.62, "f1": 0.3090909090909091, "roc_auc": 0.6114013700220596}
```

The new model replaces the previous one at once and is saved to `models/`. Predict churn of one customer:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 29.99, "usage_hours": 42.5, "support_requests": 1,
       "account_age_months": 18, "failed_payments": 0, "region": "europe",
       "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1}'
```

or of a list of them:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"monthly_fee": 29.99, "usage_hours": 42.5, "support_requests": 1,
        "account_age_months": 18, "failed_payments": 0, "region": "europe",
        "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1},
       {"monthly_fee": 59.99, "usage_hours": 3.0, "support_requests": 6,
        "account_age_months": 2, "failed_payments": 3, "region": "asia",
        "device_type": "desktop", "payment_method": "crypto", "autopay_enabled": 0}]'
```

The answer is always a list, one prediction per customer in the order they were sent. From the default logistic
regression (probabilities rounded):

```json
[
  {"churn": 0, "probability_no_churn": 0.957, "probability_churn": 0.043},
  {"churn": 1, "probability_no_churn": 0.189, "probability_churn": 0.811}
]
```

What the answer means and which customers are refused: [Prediction](docs/api.md#prediction). Every endpoint with its
request and response examples is in the interactive docs at http://127.0.0.1:8000/docs.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | health message |
| GET | `/health` | whether the dataset is loaded and can be trained on and a model is served; always `200`, `status` is `ok` or `degraded` |
| POST | `/predict` | predict churn of one customer or a list of them: class and probabilities of both classes; `503` until a model is trained |
| GET | `/dataset/preview?n=5` | first `n` rows of the training dataset, `1 <= n <= 100` |
| GET | `/dataset/info` | row and column counts, feature names, churn class distribution |
| GET | `/dataset/split-info` | train and test sizes, churn class distribution in each part |
| POST | `/model/train` | train the churn model of the type and with the hyperparameters from the optional JSON body, save it, return accuracy, churn F1 and ROC-AUC on the test part |
| POST | `/model/tune` | score a grid of hyperparameters by cross-validation on the train part, return the best config ready for `/model/train` and the five best combinations; changes no model |
| GET | `/model/status` | whether a model is trained, when, its type, main hyperparameters and test metrics |
| GET | `/model/metrics?model_type=&limit=10&best_by=` | the latest training, the `limit` newest ones and the best one by a test metric, optionally of one model type |
| GET | `/model/schema` | features `/predict` expects, in the training order: how the model treats each one, its type and accepted values |

Every error has the same JSON body, `{"code", "message", "details"}`, whatever the status. What each endpoint takes
and answers, the responses for the bundled dataset and every error with its example:
[docs/api.md](docs/api.md).

## Health and logs

`GET /health` reports what the service can do right now:

```json
{"status": "degraded", "dataset_loaded": true, "dataset_trainable": true, "model_available": false}
```

`status` is `ok` when all three flags are `true` and `degraded` otherwise. The answer is `200` either way: a service
without a model is up and can train one, so a healthcheck must not restart it. What each flag means for the
endpoints: [docs/api.md](docs/api.md#health).

The service logs to stderr: loading and splitting the dataset, every training, tuning and prediction, and every
handled error. Feature values never reach the log. The level comes from the `LOG_LEVEL` environment variable, `INFO`
by default; `.env.example` lists it, and `make run` and `make docker-run` pass `.env` to the service when it exists.
A value that is not a level is reported with a warning and falls back to `INFO`.

```
INFO src.services.model: Trained logreg in 0.09s, serving the model trained at 2026-10-28T08:01:18.706309+00:00: accuracy=0.5875 f1=0.3478 roc_auc=0.6110
INFO src.services.prediction: Predicted churn of 1 customers in 0.026s, 1 of them churn; logreg model trained at 2026-10-28T08:01:18.706309+00:00
WARNING src.core.error_handlers: POST /predict failed validation with 9 errors, first at body.FeatureVectorChurn.monthly_fee: missing
```

What is logged and why: [docs/monitoring.md](docs/monitoring.md).

## Tests

```bash
make test   # all tests
make lint   # ruff check and ruff format --check

# or manually
pip install -r requirements-dev.txt
pytest                      # all tests
pytest tests/unit           # schemas, the ml pipeline and every service on its own
pytest tests/integration    # HTTP API via httpx.AsyncClient and the whole pipeline via TestClient
```

`tests/integration/test_churn_pipeline.py` runs the whole scenario through `fastapi.testclient.TestClient` on
`tests/data/churn_sample.csv`, a fixed 150-row sample of the bundled dataset: the app starts with its lifespan, trains
via `/model/train`, reports via `/model/status` and predicts via `/predict`, including the errors on the way. Models
and the training history trained by the tests go to a temporary directory, never to `models/`.

## Project layout

```
Dockerfile             # image of the service, see Run in Docker
.env.example           # settings read from the environment, copy to .env
data/
└── churn_dataset.csv  # training dataset
models/                # trained model and training history saved by /model/train, not tracked by git
src/
├── main.py            # FastAPI app, startup, router registration
├── api/               # HTTP endpoints
├── core/              # config, /docs examples, exceptions, error handlers
├── ml/                # churn pipeline: data preparation, train/test split, training, scoring, tuning
├── services/          # dataset loading, the cached split, model storage, training history, prediction and health
└── schemas/           # Pydantic models of features, dataset rows and responses
tests/
├── data/              # fixed 150-row sample of the dataset for the pipeline tests
├── unit/
└── integration/
docs/                  # endpoint reference and notes on the underlying concepts
```

## Documentation

- [docs/api.md](docs/api.md) - every endpoint: requests, responses for the bundled dataset, errors with examples.
- [docs/docker.md](docs/docker.md) - what goes into the image, how it runs and keeps the trained model.
- [docs/monitoring.md](docs/monitoring.md) - what is logged, what is kept out of the log, what `/health` reports.
- [docs/dataset_loading.md](docs/dataset_loading.md) - reading, validating and serving the dataset.
- [docs/preprocessing.md](docs/preprocessing.md) - features and target, gaps, the stratified split, leakage.
- [docs/model_training.md](docs/model_training.md) - the two models, the metrics, class weights, tuning.
- [docs/model_persistence.md](docs/model_persistence.md) - saving the model with joblib and loading it back.
- [docs/training_history.md](docs/training_history.md) - recording every training and comparing configs.
- [docs/prediction.md](docs/prediction.md) - from a request to model input, what the probability means.
- [docs/error_handling.md](docs/error_handling.md) - one error format, the status of each failure.
- [docs/fastapi_pydantic_basics.md](docs/fastapi_pydantic_basics.md) - FastAPI and Pydantic concepts the service uses.
