# Training a churn classifier

Short notes on the baseline model: what it computes, why its features are
prepared the way they are, and how its quality is measured.

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

## Why numeric features are scaled

- Regularization penalizes all weights by the same rule. A fee in tens and a
  counter in units would need weights of different size to have the same
  effect, so the penalty would hit them unequally. `StandardScaler` brings
  every feature to mean 0 and standard deviation 1, which removes that bias.
- Scaled features also make the solver converge in fewer iterations.
- A 0/1 flag such as `autopay_enabled` is already on that scale and is passed
  through as it is.
- Tree models split on thresholds and do not need scaling.

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
- Other ways to handle imbalance: moving the decision threshold below 0.5 or
  resampling the train part.
