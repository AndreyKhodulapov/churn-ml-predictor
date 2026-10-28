# Training a churn classifier

Short notes on the two churn models: what they compute, why their features
are prepared the way they are, how they are configured, how their quality
is measured and how their hyperparameters are tuned.

## Logistic regression

- A linear model for binary classification: it computes a weighted sum of the
  features `z = w·x + b` and turns it into a probability of churn with the
  sigmoid `p = 1 / (1 + e^(-z))`. `predict` answers 1 when `p >= 0.5`,
  `predict_proba` returns `p` itself.
- Training finds the weights that minimize log loss: confident wrong answers
  cost the most.
- Each weight shows how a feature moves the odds of churn, so the model is easy
  to explain. It also trains in milliseconds, which makes it a good baseline to
  compare stronger models (gradient boosting) against.
- Its limit is the linear boundary: interactions such as "many failed
  payments only matter without autopay" have to be added as features by hand.
- scikit-learn applies L2 regularization by default (`C=1.0`, smaller means
  stronger): large weights are penalized, which reduces overfitting.

## Random forest

- An ensemble of decision trees. Each tree is grown on a bootstrap sample of
  the train rows (drawn with replacement) and looks at a random subset of the
  features at every split (`max_features="sqrt"`). The forest averages the
  class probabilities of its trees.
- A single deep tree memorizes the train part. Trees grown on different
  samples and features make different mistakes, so their average keeps the
  flexibility of a tree and loses most of its variance.
- Trees find thresholds and interactions on their own, such as "failed
  payments only matter without autopay", which a linear model needs as
  hand-made features. The price is a model that is harder to explain.
- Main hyperparameters: `n_estimators` (more trees give a stabler answer and
  slower training, but do not overfit), `max_depth` and `min_samples_leaf`
  (limit the size of a tree: smaller trees overfit less and give smoother
  probabilities), `max_features` and `class_weight`.
- The bootstrap samples and feature subsets depend on `random_state`. The
  service fixes it to 42 unless the client sends its own, so the same config
  trained twice gives the same forest.
- On the bundled dataset fully grown trees fit the noise of 1600 rows: the
  balanced forest reaches churn F1 0.24. Limiting the depth to 5 raises it to
  0.31, still below 0.35 of the balanced logistic regression.

## Why numeric features are scaled

- Regularization penalizes all weights by the same rule. A fee in tens and a
  counter in units would need weights of different size to have the same
  effect, so the penalty would hit them unequally. `StandardScaler` brings
  every feature to mean 0 and standard deviation 1, which removes that bias.
- Scaled features also make the solver converge in fewer iterations.
- A 0/1 flag such as `autopay_enabled` is already on that scale and is passed
  through as it is.
- Tree models split on thresholds and do not need scaling. The pipeline
  scales for the random forest too: a threshold moves with the scale, but the
  splits stay the same, and both models share one pipeline.

## Why categorical features are one-hot encoded

- A linear model multiplies every input by a weight, so it needs numbers.
  Encoding `region` as `0, 1, 2, 3` would invent an order and a distance
  between regions that do not exist.
- `OneHotEncoder` turns a column into one 0/1 column per category, and each
  category gets its own weight.
- `handle_unknown="ignore"` encodes a category the model has not seen in
  training as all zeros instead of failing the prediction.

## Pipeline

- `Pipeline` chains the steps into one estimator: here gap filling, then
  scaling and encoding (`ColumnTransformer`), then the classifier.
- `pipeline.fit(X_train, y_train)` fits every step on the train part only, so
  no statistic of the test part leaks into training.
- `pipeline.predict(X)` applies the same fitted steps to raw rows, so training
  and inference cannot drift apart, and the whole chain is saved and loaded as
  one object.
- Cross-validation and hyperparameter search take the pipeline as is and refit
  the preprocessing inside every fold.

## Training config

- `POST /model/train` takes `{"model_type": ..., "hyperparameters": {...}}`.
  The model class is looked up by type, and the hyperparameters are set over
  its scikit-learn defaults with `set_params`.
- Names are checked before training against `get_params()`, the list of
  everything the model accepts. Values are checked by scikit-learn only in
  `fit()`: a single value out of range raises `InvalidParameterError`, a
  forbidden combination (`l1_ratio=1`, an l1 penalty, with the default
  `lbfgs` solver) a plain `ValueError`. Both become `422` with the reason.
- `/model/status` reads the hyperparameters back from the fitted classifier
  with `get_params()`, so it shows the defaults the client did not send,
  next to the metrics they produced.
- Without a body the config defaults to a logistic regression with the
  scikit-learn defaults.

## Metrics

- Accuracy is the share of right answers. With 80% of customers staying, a
  model that never predicts churn already gets about 0.8, so accuracy alone
  says little here.
- Precision: of the customers predicted to churn, how many really left.
  Recall: of the customers who left, how many the model found.
- F1 is their harmonic mean, `2 · precision · recall / (precision + recall)`.
  It is low when either of them is low, so it exposes a model that ignores the
  rare class. Here it is computed for the churn class.
- A model that never predicts churn has no precision to compute;
  `zero_division=0` makes its F1 0 instead of a warning.
- ROC-AUC is computed from the churn probabilities (`predict_proba`) rather
  than the labels: it is the chance that a random churner gets a higher
  probability than a random customer who stays. 0.5 is a random guess, 1 a
  perfect ranking. It does not depend on the 0.5 threshold, so a model with
  F1 0 can still rank churners well. It needs both classes in the test part,
  which the stratified split keeps.
- On the bundled dataset every model gets ROC-AUC about 0.6, and class weights
  barely change it (0.609 without them, 0.611 balanced): they move the
  threshold, not the ranking. F1 shows how useful the labels are, ROC-AUC
  how useful the probabilities are as a risk score.
- Metrics are taken on the test part, which the model has not seen, so they
  estimate quality on new customers.

## Class weights

- Without weights the baseline here reaches accuracy 0.79, close to the 0.7975
  of always answering "stays", and churn F1 0.05: it almost never predicts
  churn.
- `class_weight="balanced"` weights every class by
  `n_samples / (n_classes · class_count)`, so both classes carry the same total
  weight in the loss: with 20% churners one churner counts about four times as
  much as a customer who stays (about 2.5 against 0.63).
- The model then finds more churners: F1 grows to 0.35 while accuracy drops to
  0.59. Which trade-off is better depends on the cost of a missed churner
  versus a false alarm.
- A random forest takes the same `class_weight`: the weights count in the
  split criterion of every tree. `"balanced_subsample"` recomputes them on
  every bootstrap sample.
- Class weights are set as a hyperparameter, `{"class_weight": "balanced"}`
  or a dict of weights by class, `{"class_weight": {"0": 1, "1": 4}}`.
  JSON object keys are always strings, while the model looks the classes `0`
  and `1` up as ints, so `TrainingConfigChurn` turns digit keys back into ints;
  without that sklearn answers "The classes, [0, 1], are not in class_weight".
- Other ways to handle imbalance: moving the decision threshold below 0.5 or
  resampling the train part.

## Hyperparameter tuning

- `POST /model/tune` takes `{"model_type", "param_grid", "scoring", "cv"}`
  and runs `GridSearchCV`: every combination of the listed values is scored
  by cross-validation, and the best one comes back as `best_config`, the body
  `POST /model/train` takes.
- Cross-validation runs on the train part only. `StratifiedKFold` cuts it
  into `cv` folds with the same churn share; each combination is fitted on
  `cv - 1` folds and scored on the remaining one, and the scores are averaged.
  The test part stays unseen: a combination picked by its test score would
  make the test metrics a result of the choice rather than an estimate of
  quality on new customers.
- The whole pipeline is searched, so gap filling and scaling are refitted
  inside every fold and the scored fold leaks into nothing. A hyperparameter
  of a pipeline step is addressed as `<step>__<name>` (`model__C`); the
  response drops the prefix, so its names are the ones `/model/train` takes.
- `refit=False`: by default `GridSearchCV` refits the best combination on the
  whole train part. Here the search only suggests a config, and the client
  trains it with `/model/train`, which scores it on the test part and saves
  it. Tuning cannot replace the served model.
- The classifier and the folds use the fixed `random_state`, so the same
  request gives the same answer, and the forest scored in the search is the
  one `/model/train` fits for the same config.
- The grid grows multiplicatively: 4 values of `C` times 2 of `class_weight`
  is 8 combinations, 40 fits with 5 folds. The search runs while the client
  waits, so a request may need at most 200 fits; a forest of 100 trees takes
  about 0.25 s to fit here, so 200 fits are about a minute.
- Without `param_grid` the default grid of the model type is searched: `C`
  and `class_weight` for the logistic regression; `max_depth`,
  `min_samples_leaf` and `class_weight` for the forest. `n_estimators` stays
  at its default: more trees make the answer stabler and training slower, but
  trade no metric off.
- A combination the model refuses does not stop the search
  (`error_score=nan`): every refused one is collected and the request answers
  `422` with their list and the reason of the first one, found by training it
  again as `/model/train` would. When every fit fails, sklearn raises a
  `ValueError` whose message holds tracebacks with server paths, so it is not
  returned.

## Reading tuning results

- On the bundled dataset the default logistic regression grid picks
  `{"C": 0.1, "class_weight": "balanced"}` with cv F1 0.405 ± 0.024. The other
  values of `C` with balanced weights score 0.395-0.398: closer than one
  standard deviation, so they are as good. The choice that matters is
  `class_weight`: without it F1 drops to 0.09.
- `/model/train` with that config gets test F1 0.344, below the cv score and
  about the 0.348 of `C=1`. The best of many noisy scores is usually an
  overestimate, as luck helped it win, and the test part of 400 rows is noisy
  too. Treat a difference smaller than `std_score` as noise.
- The forest grid picks `max_depth=5` with balanced weights: cv F1 0.402,
  test F1 0.314, still behind the logistic regression.
- The metric decides what "best" means. ROC-AUC scores how the predicted
  probabilities rank churners above the others and ignores the 0.5
  threshold, so class weights barely move it: tuned by `roc_auc`, the best
  logistic regression has no class weights (cv ROC-AUC 0.669) and test F1
  0.02. Tune by F1 when the churn label is used, by ROC-AUC when customers are
  ranked by risk.

## Tuning in production

- Here tuning is a synchronous request, as the dataset fits in memory and a
  search takes seconds. Production services run it as a separate job: a
  request starts it and gets a job id, the job runs on its own machines, and
  its trials go to an experiment tracker such as MLflow. Examples: SageMaker
  hyperparameter tuning jobs, Katib in Kubeflow, Optuna or Ray Tune in a
  training pipeline.
- Grid search is easy to read but spends most fits on poor regions of the
  grid. Random search and Bayesian optimization (Optuna's TPE) usually find a
  better config within the same number of fits.
