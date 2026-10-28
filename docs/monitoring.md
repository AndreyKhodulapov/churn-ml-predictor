# Monitoring the churn service

Short notes on what the service tells about itself while it runs: which
events it logs, what it keeps out of the log and what `/health` reports.

## Why logs and a health check

- Once the service runs in a container nobody watches, the log is the only
  record of what it did: which model was trained with which metrics, which
  model answered a prediction, why a request failed.
- A health check answers a narrower question for a machine rather than a
  person: is the service up, and what can it do right now. Docker, a load
  balancer or an orchestrator ask it on a schedule and act on the answer.

## What is logged

- The dataset: where it is loaded from, how many rows, the train and test
  sizes and churn shares, and why a load or a split failed.
- Every training: the model type and hyperparameters at the start; the
  duration, the test metrics and the `trained_at` of the new model at the end.
  A training that the model refuses is logged with the reason.
- Every tuning: the grid, the metric and the folds at the start; the duration
  and the best combination at the end.
- Every prediction: the batch size, the duration, how many customers are
  predicted to churn and the type and `trained_at` of the model that answered.
  `trained_at` ties a prediction to the training that made its model, so a
  wrong answer can be traced back to its config and data.
- Handled errors: a validation error with the number of errors and the first
  of them, refused hyperparameters, a prediction before any training, a model
  that fails on valid customers with its traceback, and the method and path of
  every unhandled error.
- Every request also gets a line in the access log of uvicorn, with its
  method, path and status: the service does not repeat it.

## What is not logged

- Feature values: neither of predicted customers nor of invalid requests. They
  describe real customers, a batch may hold thousands of them, and a log is
  kept longer and read by more people than the data it came from.
- So a validation error is logged by where it is and which check failed,
  `body.FeatureVectorChurn.region: literal_error`, never by the value sent;
  the answer to the client follows the same rule.
- Only the first validation error is logged with the count of the others: a
  batch of a thousand invalid customers would otherwise write a thousand
  lines for one request.
- The traceback of an unhandled error is not logged by the handler: Starlette
  re-raises the exception after the answer and uvicorn logs the traceback.
  Logging it in the handler too would print it twice; the handler adds the
  request it belongs to.

## Levels and format

- `INFO` for what the service did, `WARNING` for what a client or the data
  did wrong and the service survived, `ERROR` for what the service did wrong.
  An alert on `ERROR` lines then fires on bugs, not on bad requests.
- The level comes from the `LOG_LEVEL` environment variable, `INFO` by
  default, so a container is started at `DEBUG` without rebuilding the image.
  A name that is not a level is reported with a warning and falls back to
  `INFO`: a typo in the environment must not stop the container.
- Every line holds the time, the level and the logger, which is the module:
  `src.services.prediction` names where the line came from.
- The log goes to stderr rather than to a file. In a container, `docker logs`
  collects it and a log collector ships it on; a file inside the container
  would be lost with it.

## What /health reports

- `dataset_loaded`, `dataset_trainable` and `model_available`, the three
  things the endpoints depend on, and `status`: `ok` when all of them hold,
  `degraded` otherwise.
- The answer is `200` even when `degraded`. A health check answers two
  different questions: liveness (is the process alive, or should it be
  restarted) and readiness (can it serve requests now). A service without a
  model is alive and can train one; a `503` would make Docker mark it
  unhealthy and an orchestrator restart it, and a restart brings no model.
  The fields answer readiness for whoever needs it.
- The check loads nothing: it reads whether the dataset and the split are in
  their cache and whether the store holds a model. A health check is called
  every few seconds, and a load reads and validates every row of the dataset.
  `lru_cache` keeps only a successful load, so after a failed one the cache is
  empty, `/health` reports `false`, and the next request that needs the
  dataset retries the load.
- `/health` is not logged by the service: Docker calls it every 30 seconds,
  and the access log of uvicorn already has a line for every call.

## Beyond logs and a health check

- Structured logs: every line a JSON object with named fields, so a log
  collector filters by model type or batch size instead of parsing text.
- A request id, set by a middleware and added to every line of the request,
  ties the access log, the service lines and a traceback together.
- Metrics instead of log lines for what is counted: requests, latency
  percentiles, errors by code, predicted churn share. They are scraped by
  Prometheus and drawn in Grafana; a log line per prediction does not scale to
  thousands of requests a second.
- Model monitoring: the distribution of incoming features and of predicted
  probabilities compared with the training data. A drift there warns that the
  model is getting stale long before anybody measures its accuracy on new
  labels.
- In Kubernetes liveness and readiness are separate probes, so `/health`
  would split into one that always answers `200` while the process runs and
  one that answers `503` until the service can predict, keeping traffic away
  from it without restarting it.
