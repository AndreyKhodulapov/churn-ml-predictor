# Loading a training dataset in a FastAPI service

Short notes on reading, validating and serving a tabular dataset.

## Reading CSV with pandas

- `pd.read_csv(path)` returns a `DataFrame`: a table of named columns, each
  a typed `Series` that shares one row index.
- Column types are inferred: numbers become `int64` / `float64`, text becomes
  a string dtype. A column with a missing value turns from `int64` into
  `float64`, because `NaN` is a float.
- Typical failures: `FileNotFoundError`, `pd.errors.EmptyDataError` (empty
  file), `pd.errors.ParserError` (broken quoting, wrong number of fields).
  Wrap them into one domain error so callers do not depend on pandas.
- `df.shape` gives `(rows, columns)`, `df.columns` the column names,
  `df.head(n)` the first rows, `df.dtypes` the inferred types.

## DataFrame vs list of models

- A `DataFrame` is columnar and vectorized: aggregations (`value_counts`,
  `mean`, `groupby`) are fast and short. It is the natural input for
  scikit-learn.
- A list of Pydantic models is row-oriented: each row is validated against
  the same schema the API uses, and it serializes to JSON directly.
- `df.to_dict("records")` converts a frame to a list of dicts, one per row;
  `Model.model_validate(record)` turns each dict into a model.
- Validating every row at load time catches bad data (unknown category,
  negative value, missing column) before it reaches training or a response.
- pandas reads `inf` in a numeric column as infinity. It is not a gap, so no
  imputer fills it, and scikit-learn refuses it in `fit()` with a `ValueError`
  that the training would report as refused hyperparameters. The row schema
  sets `allow_inf_nan=False`, so such a row fails validation at load time with
  its line number. Gaps are `NaN` in the frame too, but they are turned into
  `None` before validation.

## Class distribution

- `df["churn"].value_counts()` counts rows per class;
  `value_counts(normalize=True)` gives shares instead of counts.
- Churn data is usually imbalanced (here about 80% stay, 20% leave). A model
  that always predicts "stays" gets 80% accuracy and is useless, so accuracy
  alone is a bad metric: look at precision, recall, F1 or ROC AUC.
- Imbalance is handled with stratified splits, class weights
  (`class_weight="balanced"`) or resampling.

## Loading data once

- Reading the file on every request wastes time and memory. Load it once and
  reuse the same object.
- `@lru_cache` on a dependency function makes it a lazy singleton:
  `Depends(get_dataset)` calls the function per request, but the body runs
  only on the first call. Exceptions are not cached, so a failed load is
  retried on the next call.
- Calling the dependency in `lifespan` loads the data at startup, so the
  first request does not wait for it. Two options for a missing or broken
  file: stop the service right away (fail fast), or start without data and
  answer `503 Service Unavailable` on the endpoints that need it. This
  service does the latter: `/health` and `/predict` keep working, and since
  exceptions are not cached, the first request after the file is fixed
  loads it without a restart.
- Keeping the dataset behind a dependency, not a module-level global, lets
  tests swap it with `app.dependency_overrides[get_dataset]`.

## Errors at the API boundary

- An unexpected exception should become a generic `500` JSON response;
  file paths and stack traces stay in the server log, not in the response.
- An expected failure gets its own handler and status: `DatasetLoadError`
  becomes `503` ("try again later"), not `500` ("bug in the service").
  A handler registered for a specific class wins over the `Exception` one.
- A dataset that loads but cannot be trained on (no labeled rows, a single
  class) is a different failure: `DatasetNotTrainableError` becomes `409
  Conflict` with the reason in `message`. Unlike the load error, the reason
  names only counts and classes, so it is safe to show.
- Invalid query parameters (`n=0`, `n=abc`) are rejected by `Query(ge=..., le=...)`
  with `422` before the handler runs.
