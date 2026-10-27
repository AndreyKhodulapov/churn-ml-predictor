# Handling errors in the churn service

Short notes on how the service answers what it cannot do: one JSON format,
the status of each failure and what never leaves the server.

## One format for every error

- Every error answer has the body `{"code", "message", "details"}`
  (`ErrorResponse` in `src/schemas/errors.py`), whatever raised it: the
  service, request validation or the routing of Starlette.
- `code` is a stable name for a client to branch on, like `model_not_trained`.
  Parsing `message` would break as soon as its wording changes.
- `message` is written for a person and is safe to show: it never names a
  server path or holds a traceback.
- `details` holds structured context when the error has any, like the list
  of invalid values of a `validation_error`, and is `null` otherwise.
- The body is flat rather than wrapped in `{"error": {...}}`: there is
  nothing else to put next to it.

## Exceptions and handlers

- The services raise domain exceptions (`src/core/exceptions.py`) and know
  nothing of HTTP. Each exception has a handler (`src/core/error_handlers.py`)
  that picks the status and the code, so an endpoint has no `try` of its own.
- A handler registered for a class wins over the one for `Exception`, which
  answers everything unexpected with `500 internal_error`.
- `HTTPException` is registered from `starlette.exceptions`: routing raises
  the Starlette one for an unknown path (`404`) or method (`405`), and the
  FastAPI one is its subclass, so one handler covers both. The code is the
  status phrase in snake case (`not_found`, `method_not_allowed`), and the
  headers, like `Allow` of a `405`, are kept.

## Choosing the status

- `422`: the request itself is wrong, and sending it again will not help.
  Validation errors (`validation_error`) and hyperparameters a model refuses
  (`invalid_hyperparameters`): the body passes the schema, but only the model
  knows which values it accepts.
- `409`, not `422`, for a dataset that loads but cannot be trained on
  (`dataset_not_trainable`): the request is fine, the state of the data is
  not. The same request succeeds once the dataset is fixed.
- `503`, not `500`, for a missing dataset (`dataset_unavailable`) and for no
  trained model (`model_not_trained`): nothing is broken, the service is not
  ready yet. `503` tells a client to retry later; `500` reports a bug.
- `500` for a model that fails on valid customers. It gets its own code,
  `prediction_failed`, rather than the generic `internal_error`, so the
  client learns that the request was fine and the model is at fault.

## Cases worth checking

- A wrong number of features: a missing one fails with `missing`, an unknown
  one with `extra_forbidden` (`extra="forbid"`), so the client never believes
  the model uses a field it ignores.
- A wrong type: `"abc"` for a number, `1.5` for an integer, a category
  outside the allowed values. Each invalid value is a separate item of
  `details`.
- `NaN` and infinity (`1e400` is valid JSON and does not fit a float) are
  refused with `finite_number` (`allow_inf_nan=False`) instead of making
  scikit-learn fail with a `500`.
- An empty dataset: a zero-byte file cannot be read (`503
  dataset_unavailable`), a file with a header only has no labeled rows (`409
  dataset_not_trainable`).
- An empty list of customers gets `422`: it is most likely a client bug, and
  an empty answer would hide it.
- No trained model gets `503` with a message that names the fix.

## Validation errors

- FastAPI answers a validation error with the pydantic errors as they are.
  They hold `input`, the invalid value, which may be a whole batch of
  customers or something the client would rather not see in logs, and `ctx`,
  which repeats what the message says. The handler keeps `loc`, `message`
  and `type` only.
- `type` (`missing`, `literal_error`, `too_short`) does not change with the
  wording of the message, so a client branches on it.
- The `/predict` body is one customer or a list of them. A plain union is
  checked against both, so a customer with a bad `region` would also get
  "Input should be a valid list". A callable `Discriminator` picks the branch
  by the shape of the body, and only that branch reports errors.

## Lax types rather than strict

- Pydantic converts a value in lax mode when nothing is lost: `"29.99"` for a
  float, `18.0` or `"18"` for an integer, `true` for an integer. `1.5` for an
  integer, `"1"` for `autopay_enabled` (a `Literal[0, 1]`) and an unknown
  category are refused.
- Strict mode would refuse `18.0` for an integer, which clients built on
  pandas send all the time, since a column with a gap turns into floats.
  It would still accept `true` for `Literal[0, 1]`, since `True == 1`.
- Everything that loses information is refused already, and `NaN` and
  infinity are closed separately, so strict mode would add friction without
  closing a real hole.

## No tracebacks in answers

- An unhandled exception gets the generic `500 internal_error`. Starlette
  re-raises it after the answer is sent, so the traceback still reaches the
  server log.
- A handled exception is not re-raised, so a handler whose exception carries
  details logs them itself: the file path of `dataset_unavailable` as a
  warning, the cause of `prediction_failed` with the traceback.
- Messages that name only counts, classes, model types, hyperparameters or
  the fix (`dataset_not_trainable`, `invalid_hyperparameters`,
  `model_not_trained`) are returned as they are; the others are replaced by a
  generic text.

## Errors in /docs

- `responses={status: {"model": ErrorResponse, "content": ...}}` documents an
  error status with its schema and named examples; the examples live in
  `src/core/openapi_examples.py` next to the request examples.
- FastAPI documents its own `422` body (`HTTPValidationError`) on every
  operation with parameters, which is not what the service answers. Setting
  `422` in `responses` replaces it, so every operation with parameters sets
  it.
- FastAPI drops `None` values from the whole OpenAPI document, so
  `details: null` is not shown in the examples; the schema still shows the
  field.
- The tests send the request each example describes and check that the
  service answers exactly the example, so `/docs` cannot drift from the code.
