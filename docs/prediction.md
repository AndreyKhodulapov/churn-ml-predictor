# Serving churn predictions

Short notes on answering `/predict` with the trained model: how a request
becomes model input, what the answer means and how the endpoint is described
in `/docs`.

## From a request to model input

- FastAPI validates the body into `FeatureVectorChurn` objects, but
  scikit-learn knows nothing about Pydantic: it calls `np.asarray()` on its
  input. A single object becomes a 0-d array ("Expected 2D array, got scalar
  array"), a list of them a 1-d array of objects.
- A field outside `FeatureVectorChurn` gets `422` (`extra="forbid"`).
  Pydantic drops unknown fields by default, so a client sending, say,
  `contract_type` or the `churn` label of a dataset row would get a
  prediction and believe the model took that field into account.
- The pipeline was fitted on a `pd.DataFrame`, so the input is rebuilt the
  same way: one row per customer and one column per feature.
- The columns are taken from `pipeline.feature_names_in_`, which `fit()`
  remembers from the train part: the input has exactly the features of the
  served model, in the same order. The order of the `FeatureVectorChurn`
  fields differs (`autopay_enabled` is last there), and it does not leak in.
- The `ColumnTransformer` steps pick columns by name and are lenient: they
  accept another order, ignore an extra column, and a column missing from
  `pd.DataFrame(rows, columns=...)` would arrive as `NaN` and be filled like a
  gap. So the service compares the features of the model with the fields of
  `FeatureVectorChurn` itself and refuses to predict with a model trained on
  other features (`ModelFeaturesMismatchError`, a bare `500`).
- The fitted pipeline fills gaps, scales and encodes the row itself, so the
  endpoint passes raw features and cannot drift from what training did.
- An empty list is answered with an empty list before reaching the model:
  scikit-learn refuses a DataFrame with no rows.

## predict and predict_proba

- `predict_proba(X)` returns one row per customer with the probability of
  every class, in the order of `pipeline.classes_`, here `[0, 1]`. The two
  values of a row sum to 1.
- `predict(X)` returns the class itself: for logistic regression it is 1 when
  the churn probability is above 0.5.
- The endpoint returns both: the class for a quick decision and the
  probabilities to rank customers or apply a threshold other than 0.5.

## What the probability means

- Without class weights the model is trained on the real 80/20 class shares,
  so `probability_churn` is close to the real chance of leaving.
- With `class_weight=balanced` every churner counts 4 times more in training,
  as if half of the customers left. The model then overstates churn
  probabilities: they still rank customers correctly, but 0.56 no longer means
  "leaves with a 56% chance".
- To catch more churners and keep honest probabilities, the unweighted model
  can be used with a lower threshold (say 0.3) instead of balanced weights.

## One customer or a list

- The body is typed `FeatureVectorChurn | list[FeatureVectorChurn]`, so one
  endpoint serves both a single check and a batch. A batch is cheaper: the
  pipeline transforms all rows at once instead of once per request.
- The answer is always a list, one prediction per customer in the order they
  were sent, so a client reads it the same way in both cases.
- A plain union is checked against both types, so a 422 would list an error
  for each of them: a customer with a bad `region` also gets "Input should be
  a valid list". A callable `Discriminator` picks the type by the shape of the
  body instead (a list is a batch, anything else one customer), and only that
  type is checked.
- `loc` names the checked type by its `Tag`, then the index of the customer in
  a batch: `["body", "FeatureVectorChurn", "region"]` or
  `["body", "list[FeatureVectorChurn]", 1, "region"]`. The tags repeat the
  names pydantic gives the types of a plain union, so the locations stay the
  same.

## No trained model

- Before the first `/model/train` there is nothing to predict with. The
  service raises `ModelNotTrainedError`, and its handler answers `503` with a
  message that names the fix, instead of a bare `500`.
- `503` rather than `4xx`: the request is valid, the service is not ready
  yet. The same status is used while the dataset is unavailable.

## Describing the endpoint in /docs

- `summary` and the endpoint docstring become the title and description of
  the operation in Swagger UI.
- `Body(openapi_examples=...)` adds named request examples that Swagger UI
  offers in a dropdown and fills into "Try it out".
- `responses={200: ..., 503: ...}` adds response examples and documents the
  error status; the 200 schema itself is still generated from the return
  type `list[PredictionResponseChurn]`.
- `Field(examples=[...])` on schema fields shows sample values in the schema
  section.
- The examples live in `src/core/openapi_examples.py` and are reused by the
  tests, which check that they are in the OpenAPI schema and that the model
  trained on the bundled dataset really answers them as documented.
