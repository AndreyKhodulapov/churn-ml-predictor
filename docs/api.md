# Endpoint reference

What every endpoint of the churn service takes and answers, with the responses for the bundled dataset. How to start
the service and the first training and prediction are in the [README](../README.md#train-and-predict); why the
endpoints behave this way is in the other notes of this directory, linked from each section.

[Dataset](#dataset) · [Training config](#training-config) · [Hyperparameter tuning](#hyperparameter-tuning) ·
[Trained model](#trained-model) · [Training history](#training-history) · [Prediction](#prediction) ·
[Health](#health) · [Errors](#errors)

## Dataset

```bash
curl "http://127.0.0.1:8000/dataset/preview?n=3"   # the first rows as the service validated them, 1 <= n <= 100
curl http://127.0.0.1:8000/dataset/info
curl http://127.0.0.1:8000/dataset/split-info
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

The dataset is loaded and validated once at startup and split 80/20 into train and test with the churn share kept the
same in both parts (`stratify`, fixed `random_state`). Rows without a `churn` label are dropped before the split, and
feature gaps are filled by the churn pipeline with the train median (numeric) or the most frequent train value
(`autopay_enabled` and categorical). A warning is logged if a churn class share differs between train and test by more
than 1 percentage point.

These endpoints answer `503` while the dataset file is missing or broken and retry the load on every request;
`/dataset/split-info` answers `409` with the reason while the dataset cannot be split and trained on. How the data is
read and prepared: [dataset_loading.md](dataset_loading.md), [preprocessing.md](preprocessing.md).

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

The models, their hyperparameters and the metrics: [model_training.md](model_training.md).

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

Other grids:

```bash
curl -X POST http://127.0.0.1:8000/model/tune   # the default logreg grid by churn F1
curl -X POST http://127.0.0.1:8000/model/tune \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "param_grid": {"max_depth": [3, 5, null], "class_weight": ["balanced"]}}'
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
[model_training.md](model_training.md#hyperparameter-tuning).

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

How the model is saved and what can break when it is loaded back: [model_persistence.md](model_persistence.md).

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
[training_history.md](training_history.md).

## Prediction

`POST /predict` takes one customer object or a list of them and always answers with a list: one prediction per
customer, in the order they were sent. The requests of the [README](../README.md#train-and-predict) show both shapes,
and `/docs` shows them with the answers too.

A customer carries the nine features of the [dataset table](../README.md#dataset-churn_datasetcsv), with the same
types and accepted values, but no `churn` and no missing value. A field not in the table, `churn` included, gets
`422`, so a client never believes the model uses a field it ignores.

`churn` is `1` when `probability_churn` is above 0.5. A logistic regression trained with `"class_weight": "balanced"`
overstates it (`0.164` and `0.942` for the customers of the README instead of `0.043` and `0.811`), so read it as a
risk score rather than the real chance of leaving.

Before the first training the endpoint answers `503`; invalid customers and an empty list get `422`. The answers are
shown in [Errors](#errors).

`GET /model/schema` lists the features for a client to build requests from. They come in the order the model is
trained on them, each with its `kind` (`numeric`, `binary` or `categorical`), JSON Schema `type`, `allowed_values` and
`minimum`; it answers before the first training too.

```json
{"features": [
  {"name": "monthly_fee", "kind": "numeric", "type": "number", "allowed_values": null, "minimum": 0},
  ...
  {"name": "autopay_enabled", "kind": "binary", "type": "integer", "allowed_values": [0, 1], "minimum": null},
  {"name": "region", "kind": "categorical", "type": "string", "allowed_values": ["europe", "asia", "america", "africa"], "minimum": null},
  ...
]}
```

How a request becomes model input and what the probability means: [prediction.md](prediction.md).

## Health

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

What the service logs and why `/health` answers this way: [monitoring.md](monitoring.md).

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

Why these statuses and this format: [error_handling.md](error_handling.md).
