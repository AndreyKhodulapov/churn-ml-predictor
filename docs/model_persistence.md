# Saving and loading a trained model

Short notes on keeping a trained model across restarts: how it is written to
disk, what is saved with it and what can break when it is loaded back.

## Why persist a model

- A fitted estimator lives in process memory: a restart, a deploy or a crash
  loses it, and retraining on every start is slow and may give a different
  model if the data has changed.
- Saving it splits training from serving: the model is trained once, and the
  same file is loaded by every restart or by several service instances.
- The saved file is also a record of what is in production: which model
  answered a request and how it scored when it was trained.

## pickle

- Python's built-in serialization. It does not store code: it stores a
  recipe to rebuild an object, the import path of its class
  (`src.services.model_storage.TrainedChurnModel`) and its attributes.
- Loading imports that class by its path and restores the attributes, so the
  file is tied to the code: renaming or moving the class breaks every file
  saved before.
- Adding a field to the class does not break loading, but an old file comes
  back without the new attribute and fails only once it is read. The service
  checks at load that every field of `TrainedChurnModel` is present and
  rejects an older file like a broken one.
- Loading can run arbitrary code: an object can tell pickle to call any
  function when it is rebuilt, `os.system` included. Only files the service
  wrote itself may be loaded, never one uploaded by a user.
- It works for Python only; another language cannot read the file.

## joblib

- The library scikit-learn recommends for saving its estimators:
  `joblib.dump(model, path)`, `joblib.load(path)`.
- It is pickle underneath, so the same security and code coupling apply. It
  adds efficient storage of large numpy arrays (written as raw buffers
  instead of going through pickle), optional compression (`compress=3` or a
  `.gz` / `.xz` extension) and memory mapping on load (`mmap_mode="r"`) for
  models of hundreds of megabytes.
- For a small model such as a logistic regression it makes little
  difference against plain pickle; the gain is for forests and large
  vocabularies.
- joblib also runs work in parallel (`Parallel`, behind `n_jobs` in
  scikit-learn) and caches function results on disk (`Memory`).

## What to save with the model

- The fitted pipeline, not the bare classifier: the imputer, scaler and
  encoder are part of the model, and without them it cannot read raw rows.
- Metadata needed to describe it after a restart: test metrics, training
  time, model type and hyperparameters.
- The versions of the libraries it was trained with.
- One file holding the pipeline and its metadata cannot get out of sync. A
  separate JSON next to the model can be read without unpickling, but the two
  files have to be written and replaced together.

## Library versions

- scikit-learn does not guarantee that a model saved by one version loads in
  another: the internal attributes of an estimator may change. The file may
  fail to load or, worse, load and predict differently.
- On a version mismatch scikit-learn emits `InconsistentVersionWarning`; the
  service also stores the version with the model and logs a warning at load.
- The usual practice is to pin library versions in requirements and retrain
  the model after an upgrade instead of loading an old file.

## Writing safely

- Writing straight into the model file leaves a broken file if the process
  dies halfway through, and the next start cannot load anything.
- Writing to a temporary file in the same directory and renaming it over the
  target fixes that: a rename within one file system is atomic, so the target
  holds either the old model or the new one.
- The file is written before the in-memory model is replaced: if the write
  fails, the service keeps serving the previous model that is still on disk.
- A lock around "write, then replace in memory" keeps two trainings that
  finish at once from leaving one model on disk and the other in memory.

## Loading at startup

- The model is loaded once when the service starts, not on every request.
- A missing file is normal before the first training: the service starts
  without a model and reports it in its status.
- A file that cannot be loaded is logged and the service still starts, so a
  broken model does not take down the endpoints that do not need it; a new
  training writes a good file over it.

## Beyond a local file

- `skops.io` saves scikit-learn models in a format that only rebuilds
  allowed types, so loading an untrusted file does not run its code.
- ONNX exports a model into a language-neutral format for serving outside
  Python.
- A model registry (MLflow and similar) keeps every trained version with its
  metrics and parameters, and promotes one of them to production; the file
  itself goes to shared storage such as S3 instead of a local disk.
