import json
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.config import (
    CATEGORICAL_FEATURES,
    CLASSIFIERS,
    MAIN_HYPERPARAMETERS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TUNING_TOP_CANDIDATES,
    ModelType,
)
from src.core.exceptions import DatasetNotTrainableError, InvalidHyperparametersError
from src.ml.preprocessing import build_imputer, split_numeric_features
from src.schemas.learning import (
    TrainingConfigChurn,
    TrainMetrics,
    TuningCandidate,
    TuningConfigChurn,
    TuningResultChurn,
)


def build_feature_transformer(
    numeric_features: tuple[str, ...], categorical_features: tuple[str, ...]
) -> ColumnTransformer:
    """Scale numeric features, pass 0/1 flags as they are and one-hot encode categorical ones.

    Scaling puts fees, hours and counters on one scale, so the regularization of LogisticRegression
    penalizes their coefficients equally. A 0/1 flag is already on that scale.
    """
    scaled, binary = split_numeric_features(numeric_features)
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), scaled),
            ("binary", "passthrough", binary),
            # A category the train part has not seen gets all zeros instead of failing the prediction.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        verbose_feature_names_out=False,
    )


def check_hyperparameter_names(model_type: ModelType, names: Iterable[str]) -> None:
    """Raise InvalidHyperparametersError if model_type does not know some of the names."""
    valid = CLASSIFIERS[model_type]().get_params()
    unknown = sorted(set(names) - set(valid))
    if unknown:
        raise InvalidHyperparametersError(
            f"Unknown hyperparameters of {model_type}: {', '.join(unknown)}; valid ones are {', '.join(sorted(valid))}"
        )


def build_classifier(model_type: ModelType, hyperparameters: Mapping[str, Any]) -> ClassifierMixin:
    """Unfitted classifier of model_type with hyperparameters set over the scikit-learn defaults.

    Only the names are checked here: scikit-learn checks the values when the classifier is fitted.
    """
    check_hyperparameter_names(model_type, hyperparameters)
    classifier = CLASSIFIERS[model_type]()
    # A random forest draws its bootstrap samples at random: a fixed seed makes training on the same split
    # give the same model. A client may still pass its own random_state.
    return classifier.set_params(**{"random_state": RANDOM_STATE, **hyperparameters})


def build_churn_pipeline(
    numeric_features: tuple[str, ...], categorical_features: tuple[str, ...], classifier: ClassifierMixin
) -> Pipeline:
    """Unfitted pipeline from raw features to churn: fill gaps, transform features, classify with classifier.

    Gap filling is a step of the pipeline, so fit() learns its statistics from the train part only
    and the fitted pipeline accepts the same raw rows as the dataset.
    """
    return Pipeline(
        [
            # The imputer returns a DataFrame with the original names, so the next step picks columns by name.
            ("impute", build_imputer(numeric_features, categorical_features)),
            ("transform", build_feature_transformer(numeric_features, categorical_features)),
            ("model", classifier),
        ]
    )


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: ModelType = "logreg",
    hyperparameters: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Fit a new churn pipeline with a model_type classifier on the train part; every call returns a separate model."""
    classifier = build_classifier(model_type, hyperparameters or {})
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, classifier)
    try:
        return pipeline.fit(X_train, y_train)
    # scikit-learn checks hyperparameter values only in fit(): a value out of range raises InvalidParameterError,
    # a subclass of ValueError, and a forbidden combination, like l1_ratio=1 (an l1 penalty) with the lbfgs
    # solver, a plain ValueError. The train part is already validated, so a ValueError here means the
    # hyperparameters do not fit the model.
    except ValueError as e:
        raise InvalidHyperparametersError(f"Cannot train {model_type} with these hyperparameters: {e}") from e


def describe_hyperparameters(pipeline: Pipeline, model_type: ModelType, requested: Mapping[str, Any]) -> dict[str, Any]:
    """Values the classifier of the pipeline was trained with: the main ones of model_type and every requested one.

    They are read from the classifier itself, so a hyperparameter left at its default shows the actual value.
    """
    params = pipeline.named_steps["model"].get_params()
    return {name: params[name] for name in (*MAIN_HYPERPARAMETERS[model_type], *requested)}


def evaluate_churn_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> TrainMetrics:
    """Accuracy, F1 of the churn class and ROC-AUC of the churn probability on the test part.

    ROC-AUC needs both churn classes in the test part: split_data() refuses a split without them
    (check_split_classes()), since a stratified split of a tiny dataset may leave the rarer class out.
    """
    predictions = pipeline.predict(X_test)
    # Classes are sorted, so the second column is the probability of churn = 1.
    churn_probabilities = pipeline.predict_proba(X_test)[:, 1]
    # sklearn returns numpy scalars, and f1_score is annotated as an array too for average=None.
    return TrainMetrics(
        accuracy=float(accuracy_score(y_test, predictions)),
        # A model that never predicts churn has no precision to take; its F1 is 0, not a warning.
        f1=float(f1_score(y_test, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, churn_probabilities)),
    )


def tune_churn_model(X_train: pd.DataFrame, y_train: pd.Series, config: TuningConfigChurn) -> TuningResultChurn:
    """Score every combination of the grid by cross-validation on the train part and return the best ones.

    The test part stays unseen, so /model/train metrics of the best config remain an honest estimate. Nothing is
    refitted on the whole train part: the client trains the best config with /model/train.
    """
    grid = config.grid
    check_hyperparameter_names(config.model_type, grid)
    # Every fold has to hold customers of both classes, otherwise StratifiedKFold refuses to split.
    rarest_class = int(y_train.value_counts().min())
    if config.cv > rarest_class:
        raise DatasetNotTrainableError(
            f"{config.cv}-fold cross-validation needs at least {config.cv} customers of each churn class "
            f"in the train part, the rarest class has {rarest_class}"
        )

    # The same pipeline as /model/train builds, so gap filling and scaling are refitted inside every fold and
    # the fixed random_state gives the same forest that /model/train fits for the best config.
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_classifier(config.model_type, {}))
    search = GridSearchCV(
        pipeline,
        # Hyperparameters of a pipeline step are addressed as <step>__<name>.
        {f"model__{name}": values for name, values in grid.items()},
        scoring=config.scoring,
        cv=StratifiedKFold(config.cv, shuffle=True, random_state=RANDOM_STATE),
        refit=False,
        # A combination the model refuses is scored NaN instead of stopping the search, so all of them are reported.
        error_score=np.nan,
        # The fits run one by one in this request's thread and leave the other cores to /predict.
    )
    # Refused combinations are reported below with a 422, so the warnings about them are not repeated in the log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FitFailedWarning)
        warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite")
        try:
            search.fit(X_train, y_train)
        # The search raises only when every fit fails, with tracebacks holding server paths in the message.
        except ValueError:
            refused = list(ParameterGrid(grid))
        else:
            refused = [
                strip_step_prefix(params)
                for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
                if np.isnan(score)
            ]
    if refused:
        raise refused_combinations_error(X_train, y_train, config.model_type, refused, len(ParameterGrid(grid)))

    results = search.cv_results_
    # A stable sort keeps tied combinations in grid order, so the first one is the best_index_ of the search.
    best = np.argsort(results["rank_test_score"], kind="stable")[:TUNING_TOP_CANDIDATES]
    candidates = [
        TuningCandidate(
            hyperparameters=strip_step_prefix(results["params"][i]),
            mean_score=float(results["mean_test_score"][i]),
            std_score=float(results["std_test_score"][i]),
        )
        for i in best
    ]
    return TuningResultChurn(
        best_config=TrainingConfigChurn(model_type=config.model_type, hyperparameters=candidates[0].hyperparameters),
        scoring=config.scoring,
        best_score=candidates[0].mean_score,
        candidates=candidates,
    )


def strip_step_prefix(params: Mapping[str, Any]) -> dict[str, Any]:
    """Hyperparameters of the classifier under the names TrainingConfigChurn takes: model__C becomes C."""
    return {name.removeprefix("model__"): value for name, value in params.items()}


def refused_combinations_error(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: ModelType,
    refused: list[dict[str, Any]],
    combinations: int,
) -> InvalidHyperparametersError:
    """Error naming the combinations model_type refuses and why it refuses the first of them.

    The search scores a refused combination NaN and drops the reason, so the first one is trained again
    as /model/train would; scikit-learn checks the values at the start of fit(), so it fails at once.
    """
    listed = ", ".join(json.dumps(params) for params in refused)
    message = f"Cannot train {model_type} with {len(refused)} of {combinations} combinations of the grid: {listed}"
    try:
        train_churn_model(X_train, y_train, model_type, refused[0])
    except InvalidHyperparametersError as e:
        return InvalidHyperparametersError(f"{message}; the first one fails with: {e.__cause__}")
    # It failed only on some folds of the train part.
    return InvalidHyperparametersError(message)
