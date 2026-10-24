# Preparing churn data for training

Short notes on turning a raw table into train and test sets a model can learn
from.

## Features and target

- `X` is the feature matrix (one row per customer, one column per feature),
  `y` is the target vector (`churn`: 1 - left, 0 - stayed). scikit-learn
  estimators take them separately: `model.fit(X, y)`, `model.predict(X)`.
- The target must never be in `X`: the model would learn to copy the answer
  and fail on real data, where the answer is unknown.
- A row without a label cannot be used for supervised training, so it is
  dropped. Filling a missing label would invent the answer.

## Numeric and categorical features

- Numeric features carry order and distance (`monthly_fee`, `usage_hours`);
  categorical ones are labels from a fixed set (`region`, `device_type`).
- Decide by meaning, not by dtype: a region code stored as `1, 2, 3` is still
  categorical, and a 0/1 flag such as `autopay_enabled` can stay numeric.
- The split matters because each type gets its own treatment: numeric
  features are scaled for linear models, categorical ones are encoded
  (`OneHotEncoder`), and missing values are filled differently.
- Listing the columns explicitly instead of using `select_dtypes` keeps the
  feature set stable when a column's dtype changes, for example `int64`
  turning into `float64` after a gap appears.

## Missing values

- Options: drop rows (loses data, may bias the sample), fill with a statistic
  (imputation), or add a "missing" category / indicator column when the gap
  itself is informative.
- Median is robust to outliers: one tariff of 1000 moves the mean a lot and
  the median barely. Mean suits roughly symmetric data without outliers.
- Most frequent value suits categorical and binary features: the filled
  value is a real category, and a 0/1 flag stays 0 or 1 instead of 0.67.
- `SimpleImputer` (`strategy="median"`, `"most_frequent"`, `"constant"`)
  does this in scikit-learn. A column with no observed values in the fitting
  data is dropped from the output with a warning.

## Data leakage

- Leakage is any information from the test set or from the future reaching
  training. Metrics then look better than the model will be in production.
- Every statistic a transformer learns (median, mean, scale, category list)
  is a fit on data. Fit it on train only: `fit_transform(X_train)`, then
  `transform(X_test)` with the same fitted object.
- Imputing before the split computes the median over test rows too - a
  small but real leak. The same rule applies to scaling, encoding and
  feature selection.
- Putting the transformers into a `Pipeline` makes this automatic, including
  inside cross-validation, and the same fitted steps are reused at inference.

## Train and test split

- The test set is held out to estimate how the model works on unseen data;
  it is used once, for the final evaluation, not for tuning.
- `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`:
  `test_size` is the held-out share (20-30% is typical), `random_state` makes
  the shuffle reproducible, `stratify=y` keeps the class shares equal in both
  parts.
- Stratification matters for imbalanced data: with 20% churn a random split
  can give a test set with noticeably fewer churners, and the metrics become
  noisy. It needs at least 2 rows of every class.
- Hyperparameters are tuned on a separate validation set or with
  cross-validation on train (`StratifiedKFold`), never on test.
- When the data has a time order, a random split leaks the future into
  training; split by time instead (train on earlier months, test on later).

## Checking class distribution

- After the split, compare the share of every class in train and test
  (`y.value_counts(normalize=True)`). With stratification the difference is
  at most one row; a bigger gap means the split or the data is wrong.
- Here 2000 rows give 1600 for train and 400 for test, both with about 20%
  churn.

## ColumnTransformer

- `ColumnTransformer` applies a different transformer to each group of
  columns and concatenates the results, which is how numeric and categorical
  features get separate preprocessing in one object.
- By default it returns a numpy array; `set_output(transform="pandas")` keeps
  a `DataFrame`, and `verbose_feature_names_out=False` keeps the original
  column names instead of prefixes like `numeric__monthly_fee`.
