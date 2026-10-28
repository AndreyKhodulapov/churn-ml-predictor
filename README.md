# churn-ml-predictor

A FastAPI service that predicts customer churn: whether a customer will leave
the service next month (`churn = 1`) or stay (`churn = 0`), based on their
billing, usage and support history.

Current state: the API validates customer features, serves the training
dataset, trains a logistic regression or a random forest on it with the
requested hyperparameters, finds the best hyperparameters by grid search,
keeps the trained model on disk across restarts, records the test metrics of
every training to compare configs and predicts churn of new customers with
the model. It reports its readiness at `/health`, logs the key events and runs
in a Docker container.

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

Endpoints that need the dataset answer `503` while it is missing or broken, and `409` with the reason while it
cannot be split and trained on (no labeled rows, a single churn class, too few rows); the service itself keeps running.
Every error has the same JSON body, see [Errors](#errors).

## Features

| Field | Type | Values |
|-------|------|--------|
| `monthly_fee` | float | >= 0, finite |
| `usage_hours` | float | >= 0, finite, hours used last month |
| `support_requests` | int | >= 0 |
| `account_age_months` | int | >= 0 |
| `failed_payments` | int | >= 0 |
| `region` | str | `europe`, `asia`, `america`, `africa` |
| `device_type` | str | `mobile`, `desktop`, `tablet` |
| `payment_method` | str | `card`, `paypal`, `crypto` |
| `autopay_enabled` | int | `0` or `1` |

A training dataset row has the same fields plus the `churn` label (`0` or `1`). A `/predict` customer with a field
that is not in this table, `churn` included, gets `422`, so a client never believes the model uses a field it ignores.

`GET /model/schema` returns the same table for a client to build requests from. The features come in the order the
model is trained on them, each with its `kind` (`numeric`, `binary` or `categorical`), JSON Schema `type`,
`allowed_values` and `minimum`; it answers before the first training too.

```json
{"features": [
  {"name": "monthly_fee", "kind": "numeric", "type": "number", "allowed_values": null, "minimum": 0},
  ...
  {"name": "autopay_enabled", "kind": "binary", "type": "integer", "allowed_values": [0, 1], "minimum": null},
  {"name": "region", "kind": "categorical", "type": "string", "allowed_values": ["europe", "asia", "america", "africa"], "minimum": null},
  ...
]}
```

## Dataset

The training data lives in `data/churn_dataset.csv`. It is loaded and
validated once at startup. If the file is missing, empty or a row does not
match the schema, the service still starts and logs the reason; the endpoints
that need the dataset answer `503` with `dataset_unavailable` and retry the load
on every request until the file is fixed.

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
- http://127.0.0.1:8000/health - what the service can do right now
- http://127.0.0.1:8000/docs - interactive API docs (Swagger UI)

Example requests:

```bash
curl "http://127.0.0.1:8000/dataset/preview?n=3"
curl http://127.0.0.1:8000/dataset/info
curl http://127.0.0.1:8000/dataset/split-info
curl -X POST http://127.0.0.1:8000/model/train
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 300, "max_depth": 5, "class_weight": "balanced"}}'
curl -X POST http://127.0.0.1:8000/model/tune
curl -X POST http://127.0.0.1:8000/model/tune \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "param_grid": {"max_depth": [3, 5, null], "class_weight": ["balanced"]}}'
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

- The image holds the runtime dependencies, `src/` and `data/` only; dev dependencies, tests and `models/` stay out.
- The trained model and the training history go to the named volume `churn-models` mounted on `/app/models`, so a
  new container serves the model the previous one trained. Without `-v` they are lost with the container.
  `docker volume rm churn-models` starts from scratch.
- Logs go to `docker logs churn-ml-predictor`; `-e LOG_LEVEL=DEBUG` on `docker run` changes their level.
- A `HEALTHCHECK` calls `/health` every 30 seconds; `docker ps` shows the container as `healthy` once the service
  answers, with or without a trained model.
- The service runs as a non-root user, in one uvicorn process: the served model lives in the memory of the process,
  so a model trained by one of several workers would not reach the others.

Why the image is built this way: [docs/docker.md](docs/docker.md).

## Training config

`POST /model/train` takes an optional JSON body:

```json
{"model_type": "random_forest", "hyperparameters": {"n_estimators": 300, "max_depth": 5}}
```

- `model_type` is `logreg` (`LogisticRegression`) or `random_forest` (`RandomForestClassifier`), `logreg` by default.
- `hyperparameters` are passed to that model under their scikit-learn names and set over its defaults; `{}` by
  default. Without a body the service trains a logistic regression with the scikit-learn defaults.
- `random_state` is fixed to 42 unless sent, so training the same config twice gives the same model.
- A field other than these two, or an unknown `model_type`, is rejected with `422` by the schema. A hyperparameter
  the model does not know, a value it refuses or a combination it forbids gets `422` with the reason:

```json
{"code": "invalid_hyperparameters", "message": "Cannot train logreg with these hyperparameters: The 'C' parameter of LogisticRegression must be a float in the range (0.0, inf]. Got 0 instead.", "details": null}
```

- `class_weight` is `"balanced"` or a dict of weights by class, like `{"0": 1, "1": 4}`: JSON object keys are
  strings, so the service turns `"0"` and `"1"` back into the classes `0` and `1` the model expects.

`/model/train` responses for the bundled dataset:

| `model_type` | `hyperparameters` | accuracy | churn F1 | ROC-AUC |
|---|---|---|---|---|
| `logreg` | `{}` | 0.79 | 0.045 | 0.609 |
| `logreg` | `{"class_weight": "balanced"}` | 0.5875 | 0.348 | 0.611 |
| `random_forest` | `{}` | 0.7825 | 0.171 | 0.585 |
| `random_forest` | `{"class_weight": "balanced"}` | 0.695 | 0.238 | 0.585 |
| `random_forest` | `{"n_estimators": 300, "max_depth": 5, "class_weight": "balanced"}` | 0.62 | 0.309 | 0.611 |

Without class weights, with 20% churners, the logistic regression almost never predicts churn: its accuracy is close
to the 0.7975 of always answering "stays" and its F1 is near zero. `"class_weight": "balanced"` weights churners up,
and the model catches about half of them at the cost of accuracy. A random forest of fully grown trees fits the noise
of the train part; shallow trees do better, but on this dataset the balanced logistic regression is still ahead.

`roc_auc` scores the churn probability rather than the label: how often a churner gets a higher one than a customer
who stays, 0.5 for a random guess. Class weights move the 0.5 threshold, not this ranking, so they change F1 a lot
and ROC-AUC barely: the unweighted logistic regression ranks churners as well as the balanced one and only fails to
call them churners.

## Hyperparameter tuning

`POST /model/tune` scores every combination of a grid by cross-validation on the train part and returns the best one
as `best_config`, the body `POST /model/train` takes, so the tune-then-train loop is two requests:

```bash
curl -X POST http://127.0.0.1:8000/model/tune \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "param_grid": {"C": [0.1, 1.0], "class_weight": [null, "balanced"]}, "scoring": "f1"}'
# send best_config from the answer to /model/train as it is
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {"C": 0.1, "class_weight": "balanced"}}'
```

- `param_grid` maps hyperparameter names of `model_type` to lists of values to try. Without it the default grid of
  the model type is searched: `C` and `class_weight` for `logreg`, `max_depth`, `min_samples_leaf` and
  `class_weight` for `random_forest`. A body without any field searches the default `logreg` grid.
- `scoring` is `f1` (of the churn class, the default), `accuracy` or `roc_auc`; `cv` is the number of folds, 2 to 10,
  5 by default.
- The test part is not used, and no model is trained or replaced: `/model/train` trains the chosen config.
- The grid may need at most 200 fits (combinations times folds), as the search runs while the client waits. A bigger
  grid, an unknown field or `scoring` gets `422` from the schema. A hyperparameter the model does not know, or
  combinations it refuses, get `422` with the reason:

```json
{"code": "invalid_hyperparameters", "message": "Cannot train logreg with 1 of 2 combinations of the grid: {\"C\": 0}; the first one fails with: The 'C' parameter of LogisticRegression must be a float in the range (0.0, inf]. Got 0 instead.", "details": null}
```

- A `cv` larger than the number of customers of a churn class in the train part gets `409`.

Answer for the bundled dataset without a body (scores rounded; `candidates` holds five combinations, best first):

```json
{
  "best_config": {"model_type": "logreg", "hyperparameters": {"C": 0.1, "class_weight": "balanced"}},
  "scoring": "f1",
  "best_score": 0.405,
  "candidates": [
    {"hyperparameters": {"C": 0.1, "class_weight": "balanced"}, "mean_score": 0.405, "std_score": 0.0238},
    {"hyperparameters": {"C": 0.01, "class_weight": "balanced"}, "mean_score": 0.3977, "std_score": 0.0251},
    "..."
  ]
}
```

`best_score` is the mean over the folds of the train part. `/model/train` with this config gets test F1 0.344, about
the same as with `C=1`: the scores of different `C` differ by less than their `std_score`, and only `class_weight`
changes the result. How to read the results and why the metric matters:
[docs/model_training.md](docs/model_training.md#hyperparameter-tuning).

## Trained model

Every successful `/model/train` saves the model to `models/churn_model.joblib`, together with its test metrics,
training time, model type, requested hyperparameters and the scikit-learn version it was trained with. The file is
written to a temporary name and renamed, so a crash during the write keeps the previous model intact. A training that
fails (`503`, `409`, `422`) leaves the previous model in place.

At startup the service loads the saved model, so `/model/status` reports the same model after a restart. Without the
file the service starts untrained. A file that cannot be read does not stop the service either: the reason is logged
and the service starts untrained until the next `/model/train` writes a new model over it. A model saved with another
scikit-learn version is loaded with a warning, since its predictions may differ. A file saved before the model type
and hyperparameters were stored is rejected the same way as a broken one: retrain the model once.

To check it by hand: train a model, restart the service and compare `/model/status` before and after.

```bash
curl -X POST http://127.0.0.1:8000/model/train
curl http://127.0.0.1:8000/model/status
# stop the service with Ctrl+C and start it again with make run
curl http://127.0.0.1:8000/model/status   # the same trained_at and metrics
```

`/model/status` before the first training:

```json
{"trained": false, "trained_at": null, "metrics": null, "model_type": null, "hyperparameters": null}
```

and after training a random forest with `{"n_estimators": 300, "max_depth": 5, "class_weight": "balanced"}`
(`trained_at` is in UTC):

```json
{
  "trained": true,
  "trained_at": "2026-10-27T10:11:34.699584Z",
  "metrics": {"accuracy": 0.62, "f1": 0.3090909090909091, "roc_auc": 0.6114013700220596},
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced"
  }
}
```

`hyperparameters` lists the main ones of the model type and every requested one, with the values the model was
trained with: `min_samples_leaf` and `max_features` were not sent and show the scikit-learn defaults.

## Training history

`/model/status` describes the served model only; every training replaces it. To follow the quality of the model and
compare configs, every successful `/model/train` also adds a record to `models/training_history.json`: `trained_at`,
`model_type`, `hyperparameters` as `/model/status` shows them and the test `metrics`. `GET /model/metrics` returns
them:

```bash
curl http://127.0.0.1:8000/model/metrics                              # the latest and the 10 newest trainings
curl "http://127.0.0.1:8000/model/metrics?limit=2&best_by=f1"          # plus the best one by churn F1
curl "http://127.0.0.1:8000/model/metrics?model_type=logreg&best_by=roc_auc"
```

- `latest` is the newest training, `history` the `limit` newest ones, newest first, `1 <= limit <= 100`.
- `best_by` is `accuracy`, `f1` or `roc_auc`; `best` is then the training with the highest value of it among every
  record, not only the listed ones, the newer one on a tie. Without `best_by` it is `null`.
- `model_type` narrows `latest`, `history` and `best` to the trainings of that type.
- An unknown `model_type` or `best_by`, or `limit` out of range, gets `422` with `validation_error`.
- Before the first training `latest` and `best` are `null` and `history` is empty; the endpoint needs no dataset.

After training the five configs of the [Training config](#training-config) table in their order,
`?limit=2&best_by=f1` answers (metrics rounded, timestamps shortened):

```json
{
  "latest": {"trained_at": "2026-10-28T06:43:12Z", "model_type": "random_forest", "...": "..."},
  "history": [
    {
      "trained_at": "2026-10-28T06:43:12Z",
      "model_type": "random_forest",
      "hyperparameters": {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced"},
      "metrics": {"accuracy": 0.62, "f1": 0.309, "roc_auc": 0.611}
    },
    {
      "trained_at": "2026-10-28T06:43:11Z",
      "model_type": "random_forest",
      "hyperparameters": {"n_estimators": 100, "max_depth": null, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced"},
      "metrics": {"accuracy": 0.695, "f1": 0.2375, "roc_auc": 0.585}
    }
  ],
  "best": {
    "trained_at": "2026-10-28T06:43:10Z",
    "model_type": "logreg",
    "hyperparameters": {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 100, "class_weight": "balanced"},
    "metrics": {"accuracy": 0.5875, "f1": 0.348, "roc_auc": 0.611}
  }
}
```

`latest` is the same record as the first one of `history`. The metric decides what "best" means: by `accuracy` the
best is the logistic regression without class weights (0.79), which almost never predicts churn (F1 0.045).

`model_type` and `hyperparameters` of a record are a body `/model/train` takes as it is and trains the same model with
again, so the best config of the history is one request away from being served:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 100, "class_weight": "balanced"}}'
```

The history keeps the 100 newest trainings and survives restarts like the model. A failed training (`409`, `422`,
`503`) is not recorded. A history file that cannot be read is logged and the service starts with an empty history;
the next training writes a new file over it. If the record cannot be written, the training still answers `200`: the
new model is already served, and the reason goes to the log.

How the history is kept and why a config picked by its test metrics needs care:
[docs/training_history.md](docs/training_history.md).

## Prediction

`POST /predict` takes one customer object or a list of them and always answers with a list: one prediction per
customer, in the order they were sent. `/docs` shows both request examples and the answers to them.

```bash
curl -X POST http://127.0.0.1:8000/model/train   # once, if no model is trained yet

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 29.99, "usage_hours": 42.5, "support_requests": 1,
       "account_age_months": 18, "failed_payments": 0, "region": "europe",
       "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1}'

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"monthly_fee": 29.99, "usage_hours": 42.5, "support_requests": 1,
        "account_age_months": 18, "failed_payments": 0, "region": "europe",
        "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1},
       {"monthly_fee": 59.99, "usage_hours": 3.0, "support_requests": 6,
        "account_age_months": 2, "failed_payments": 3, "region": "asia",
        "device_type": "desktop", "payment_method": "crypto", "autopay_enabled": 0}]'
```

Answer to the list from the model trained without class weights (probabilities rounded):

```json
[
  {"churn": 0, "probability_no_churn": 0.957, "probability_churn": 0.043},
  {"churn": 1, "probability_no_churn": 0.189, "probability_churn": 0.811}
]
```

`churn` is `1` when `probability_churn` is above 0.5. A logistic regression trained with `"class_weight": "balanced"`
overstates it (`0.164` and `0.942` for the same customers), so read it as a risk score rather than the real chance of
leaving.

Before the first training the endpoint answers `503`; invalid customers and an empty list get `422`. The answers are
shown in [Errors](#errors).

How it works: [docs/prediction.md](docs/prediction.md). The models, their hyperparameters and tuning:
[docs/model_training.md](docs/model_training.md).

## Health and logs

`GET /health` reports what the service can do right now:

```json
{"status": "degraded", "dataset_loaded": true, "dataset_trainable": true, "model_available": false}
```

| Field | `false` means |
|-------|---------------|
| `dataset_loaded` | the dataset is missing or broken: the `/dataset` endpoints answer `503` |
| `dataset_trainable` | the dataset is not loaded or cannot be split for training: `/model/train` and `/model/tune` answer `503` or `409` |
| `model_available` | no model is trained yet: `/predict` answers `503` |

`status` is `ok` when all three are `true` and `degraded` otherwise. The answer is `200` either way: a service
without a model is up and can train one, so a healthcheck must not restart it. The check loads nothing, it reads the
state the service already holds; a dataset that failed to load is retried by the next request that needs it.

The service logs to stderr, at the level of the `LOG_LEVEL` environment variable (`INFO` by default):

- loading and splitting the dataset at startup, and why it failed;
- every training and tuning: the config, the duration and the metrics;
- every prediction: the batch size, the duration, how many customers churn and which model answered (its type and
  `trained_at`, as in `/model/status`);
- handled errors: validation errors (where and which check only), refused hyperparameters, predictions before
  training, model failures with their traceback and the request of every unhandled error.

```
INFO src.services.model: Trained logreg in 0.09s, serving the model trained at 2026-10-28T08:01:18.706309+00:00: accuracy=0.5875 f1=0.3478 roc_auc=0.6110
INFO src.services.prediction: Predicted churn of 1 customers in 0.026s, 1 of them churn; logreg model trained at 2026-10-28T08:01:18.706309+00:00
WARNING src.core.error_handlers: POST /predict failed validation with 9 errors, first at body.FeatureVectorChurn.monthly_fee: missing
```

Feature values never reach the log, neither of predicted customers nor of invalid requests. What is logged and why:
[docs/monitoring.md](docs/monitoring.md).

## Errors

Every error answers with the same JSON body, whatever the status: `code` names the error for a client to branch on,
`message` explains it and `details` holds structured context or `null`. Neither of them names server paths or holds
a traceback: the reason of a `5xx` goes to the server log only.

```json
{"code": "dataset_unavailable", "message": "Dataset is not available", "details": null}
```

| `code` | Status | When |
|--------|--------|------|
| `validation_error` | `422` | the request does not match the schema: a wrong type, a value out of range, a missing or unknown field, an empty list of customers; `details` lists every invalid value |
| `invalid_hyperparameters` | `422` | `/model/train`, `/model/tune`: the body is valid, but the model does not know a hyperparameter or refuses its value |
| `dataset_not_trainable` | `409` | the dataset is loaded but cannot be split and trained on, or a churn class of the train part is smaller than `cv` |
| `dataset_unavailable` | `503` | the dataset file is missing, empty or broken |
| `model_not_trained` | `503` | `/predict` before the first `/model/train` |
| `prediction_failed` | `500` | the model fails to predict for valid customers |
| `internal_error` | `500` | any other unexpected failure |
| `not_found`, `method_not_allowed`, ... | `404`, `405`, ... | an unknown path or method; `code` is the status phrase in snake case |

An item of `details` of a `validation_error` has `loc`, where the value is (the part of the request, then the keys
and indexes down to it), `message` and `type`, the pydantic error type that does not change with the wording. The
invalid value itself is not repeated.

Values are converted when nothing is lost: numbers sent as strings (`"29.99"`), whole floats for integer features
(`18.0`) and `true`/`false` for integers are accepted. A fractional value for an integer feature (`1.5`), an unknown
category, `NaN` and infinity are refused.

`/docs` shows the error answers of every endpoint as response examples.

### Errors of /model/train

A model type the service does not know, `{"model_type": "xgboost"}`:

```json
{
  "code": "validation_error",
  "message": "Request failed validation, see details for every error",
  "details": [
    {"loc": ["body", "model_type"], "message": "Input should be 'logreg' or 'random_forest'", "type": "literal_error"}
  ]
}
```

A hyperparameter the model does not know, `{"hyperparameters": {"n_estimators": 10}}` for the default `logreg`:

```json
{"code": "invalid_hyperparameters", "message": "Unknown hyperparameters of logreg: n_estimators; valid ones are C, class_weight, dual, fit_intercept, intercept_scaling, l1_ratio, max_iter, n_jobs, penalty, random_state, solver, tol, verbose, warm_start", "details": null}
```

A dataset with a header only or no labeled row, `409`:

```json
{"code": "dataset_not_trainable", "message": "Dataset cannot be used for training: Dataset has no labeled rows", "details": null}
```

A missing, empty or broken dataset file, `503`; the file path stays in the log:

```json
{"code": "dataset_unavailable", "message": "Dataset is not available", "details": null}
```

A failed training keeps the previous model.

### Errors of /predict

No model trained yet, `503`:

```json
{"code": "model_not_trained", "message": "Churn model is not trained yet, call POST /model/train first", "details": null}
```

A wrong type of a value, `"monthly_fee": "abc"`. The body is checked as a list when it is one and as one customer
otherwise, so `loc` starts with the type it was checked against:

```json
{
  "code": "validation_error",
  "message": "Request failed validation, see details for every error",
  "details": [
    {"loc": ["body", "FeatureVectorChurn", "monthly_fee"],
     "message": "Input should be a valid number, unable to parse string as a number", "type": "float_parsing"}
  ]
}
```

A wrong set of features: the second customer of a list has no `autopay_enabled` and an unknown `contract_type`. In a
list `loc` goes on with the index of the customer:

```json
{
  "code": "validation_error",
  "message": "Request failed validation, see details for every error",
  "details": [
    {"loc": ["body", "list[FeatureVectorChurn]", 1, "autopay_enabled"], "message": "Field required", "type": "missing"},
    {"loc": ["body", "list[FeatureVectorChurn]", 1, "contract_type"],
     "message": "Extra inputs are not permitted", "type": "extra_forbidden"}
  ]
}
```

An empty list, `[]`:

```json
{
  "code": "validation_error",
  "message": "Request failed validation, see details for every error",
  "details": [
    {"loc": ["body", "list[FeatureVectorChurn]"],
     "message": "List should have at least 1 item after validation, not 0", "type": "too_short"}
  ]
}
```

The model fails on valid customers, `500`; the reason and the traceback stay in the log:

```json
{"code": "prediction_failed", "message": "Churn model failed to make a prediction", "details": null}
```

Why these statuses and this format: [docs/error_handling.md](docs/error_handling.md).

## Tests

```bash
make test

# or manually
pip install -r requirements-dev.txt
pytest                      # all tests
pytest tests/unit           # schemas and dataset service
pytest tests/integration    # HTTP API via httpx.AsyncClient and the whole pipeline via TestClient
ruff check .
```

`tests/integration/test_churn_pipeline.py` runs the whole scenario through `fastapi.testclient.TestClient`: the app
starts with its lifespan, reads a CSV from disk, trains via `/model/train`, reports via `/model/status` and predicts via
`/predict`, including the errors on the way (prediction without a model, a failed training, a missing or single-class
dataset). It reads `tests/data/churn_sample.csv`, 150 rows of the bundled dataset stratified by churn
(`train_test_split(df, train_size=150, stratify=df["churn"], random_state=42)`), so it is fast and gives the same
result on every run. Models and the training history trained by the tests go to a temporary directory, never to
`models/`.

## Project layout

```
Dockerfile             # image of the service, see Run in Docker
data/
└── churn_dataset.csv  # training dataset
models/                # trained model and training history saved by /model/train, not tracked by git
src/
├── main.py            # FastAPI app, startup, router registration
├── core/              # constants, /docs examples, exceptions, error handlers
├── routers/           # HTTP endpoints
├── services/          # dataset loading, statistics, preprocessing, train/test split, model training, storage, training history, prediction and health
└── schemas/           # Pydantic models of features, dataset rows and responses
tests/
├── data/              # fixed 150-row sample of the dataset for the pipeline tests
├── unit/
└── integration/
docs/                  # notes on the underlying concepts
```
