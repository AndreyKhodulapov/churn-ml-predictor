# Keeping the history of trainings

Short notes on recording every training of the churn model: what a record
holds, how the history is stored and what it takes to compare configs by it.

## Why keep a history

- The served model and `/model/status` describe the last training only: every
  `/model/train` replaces them, and the metrics of earlier configs are lost.
- With a history, configs are compared side by side, a drop of quality after
  a change of the data or the code shows up, and a config that did better
  before can be trained again.
- The same idea at a larger scale is experiment tracking: every run is
  recorded with its parameters and metrics to be searched and compared later.

## What a record holds

- `trained_at`, `model_type`, `hyperparameters` and the test `metrics`:
  accuracy, churn F1 and ROC-AUC.
- `hyperparameters` are read back from the fitted classifier, like in
  `/model/status`: the main ones of the model type and every requested one,
  with their actual values. A main hyperparameter left out and the same one
  sent at its default give equal values, so records compare by what the
  model was trained with rather than by what the client happened to send.
- These values are the whole config: `model_type` and `hyperparameters` of a
  record are a body `/model/train` trains the same model with again.
  `random_state` is fixed by the service, so the forest comes out the same
  too.
- The record holds no model: the shallow forest of 300 trees takes 1.5 MB on
  disk, a record about 400 bytes. A past model is retrained from its record
  instead.

## Storing it

- A JSON file, `models/training_history.json`, next to the model. Unlike a
  pickle it runs no code when read, can be opened by any tool, and does not
  break when a class of the service is renamed.
- It is a separate file rather than a part of the model file: the model file
  is replaced by every training, while the history grows; and reading the
  history should not need unpickling a model.
- JSON object keys are strings, so `class_weight` `{0: 1, 1: 4}` comes back
  as `{"0": 1, "1": 4}`. `TrainingRecord` turns the digit keys back into
  ints, as `TrainingConfigChurn` does for a request.
- The file is rewritten as a whole on every training, to a temporary file
  that is then renamed over it, like the model: a crash halfway leaves the
  previous history intact. Appending a line to a JSON Lines file would be
  cheaper, but a crash could leave a broken last line.
- Rewriting the whole file stays cheap because the history is capped: the 100
  newest records are kept and the oldest dropped.
- A lock serializes appends: two trainings finishing at once would otherwise
  both start from the same records and the second write would drop the first
  record. The records in memory are replaced as a whole, so a reader sees
  either the old list or the new one.
- The file is loaded once at startup. A missing file means no training yet;
  a damaged one is logged and the service starts with an empty history, and
  the next training writes a new file over it.

## Recording a training

- The record is added after the model is saved and served. A training that
  fails is not recorded: it has no metrics to compare.
- Saving and recording run under the lock of the model store, as one step.
  With the lock of the history alone, two trainings finishing at once could
  be served in one order and recorded in the other, and the latest record
  would not be the served model.
- If the record cannot be written, the training still answers `200` and the
  reason goes to the log. The new model is already served, and an error would
  make the client believe the training failed and retrain for nothing.

## Comparing configs

- `GET /model/metrics?best_by=f1` returns the training with the highest churn
  F1 among every record, `model_type` narrows the comparison to one type.
- The metric decides what "best" means. On the bundled dataset the best by
  accuracy is the logistic regression without class weights (0.79), which
  almost never predicts churn (F1 0.045); the best by F1 is the balanced one
  (accuracy 0.5875, F1 0.348).
- Small differences are noise. The test part has 400 customers, 81 of them
  churners: one more churner found moves recall by more than a point. By
  ROC-AUC the shallow balanced forest (0.6114) is ahead of the balanced
  logistic regression (0.6110), but a difference in the fourth decimal says
  nothing about new customers.
- Picking the best of many configs by their test metrics makes the test part
  a part of the choice: the winner is partly the config that got lucky on
  these 400 rows, and its test score overstates its quality on new
  customers. The more configs are compared, the larger the overstatement.
- So the history is for following the model and narrowing the options down.
  To choose among many configs, `/model/tune` scores them by
  cross-validation on the train part; the test metrics of the chosen one then
  stay an honest estimate. A strict setup keeps a third, holdout part that is
  looked at once, for the final model only.

## Beyond a JSON file

- Several instances of the service would each keep their own file: a shared
  history goes to a database, one row per training, with indexes by model
  type and time instead of filtering in memory.
- Experiment trackers (MLflow, Weights & Biases, Neptune) record every run
  with its parameters, metrics and artifacts, the model file and the data
  version included, and compare runs in a UI.
- A model registry links a tracked run to the model in production, so the
  served model can always be traced back to how it was trained and on which
  data.
