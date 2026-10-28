import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.core.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TUNING_TOP_CANDIDATES
from src.core.exceptions import DatasetNotTrainableError, InvalidHyperparametersError
from src.schemas.learning import TrainingConfigChurn, TrainMetrics, TuningConfigChurn, TuningResultChurn
from src.services.learning import (
    build_churn_pipeline,
    build_classifier,
    build_feature_transformer,
    describe_hyperparameters,
    evaluate_churn_model,
    train_churn_model,
    tune_churn_model,
)


@pytest.fixture
def make_X(customer_features):
    def make(rows: int, **overrides: list) -> pd.DataFrame:
        X = pd.DataFrame([customer_features] * rows)
        for column, values in overrides.items():
            X[column] = values
        return X

    return make


@pytest.fixture
def X_train(make_X) -> pd.DataFrame:
    return make_X(
        4,
        monthly_fee=[10.0, 20.0, 30.0, None],
        usage_hours=[5.0, 15.0, 25.0, 35.0],
        autopay_enabled=[0, 1, 1, None],
        region=["asia", "europe", None, "europe"],
    )


y_train = pd.Series([0, 1, 0, 1])


@pytest.mark.parametrize(
    ("model_type", "hyperparameters", "expected_class"),
    [
        ("logreg", {"C": 0.5, "class_weight": "balanced"}, LogisticRegression),
        ("random_forest", {"n_estimators": 10, "max_depth": 3}, RandomForestClassifier),
    ],
)
def test_build_classifier_creates_model_of_type_with_hyperparameters(model_type, hyperparameters, expected_class):
    classifier = build_classifier(model_type, hyperparameters)

    assert type(classifier) is expected_class
    assert classifier.get_params().items() >= hyperparameters.items()


@pytest.mark.parametrize(
    ("model_type", "model_class"), [("logreg", LogisticRegression), ("random_forest", RandomForestClassifier)]
)
def test_build_classifier_keeps_sklearn_defaults_except_fixed_random_state(model_type, model_class):
    classifier = build_classifier(model_type, {})

    assert classifier.get_params() == {**model_class().get_params(), "random_state": RANDOM_STATE}


def test_build_classifier_lets_client_override_random_state():
    assert build_classifier("random_forest", {"random_state": 7}).random_state == 7


def test_build_classifier_returns_separate_model_on_every_call():
    assert build_classifier("logreg", {}) is not build_classifier("logreg", {})


def test_random_forest_trains_reproducibly_by_default(X_train):
    def fit_forest() -> Pipeline:
        classifier = build_classifier("random_forest", {"n_estimators": 5})
        return build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, classifier).fit(X_train, y_train)

    np.testing.assert_array_equal(fit_forest().predict_proba(X_train), fit_forest().predict_proba(X_train))


def test_build_classifier_rejects_unknown_hyperparameters():
    with pytest.raises(InvalidHyperparametersError) as error:
        build_classifier("logreg", {"n_estimators": 10, "foo": 1, "C": 0.5})

    assert str(error.value).startswith("Unknown hyperparameters of logreg: foo, n_estimators; valid ones are C,")


def test_churn_pipeline_fills_gaps_transforms_and_classifies_with_given_classifier():
    classifier = LogisticRegression()
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, classifier)

    assert list(pipeline.named_steps) == ["impute", "transform", "model"]
    assert pipeline.named_steps["model"] is classifier


def test_fitted_churn_pipeline_predicts_raw_rows_with_gaps(X_train):
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, LogisticRegression()).fit(X_train, y_train)

    predictions = pipeline.predict(X_train)

    check_is_fitted(pipeline)
    assert set(predictions) <= {0, 1}
    assert len(predictions) == len(X_train)


def test_churn_pipeline_learns_gap_filling_from_train_part(X_train, make_X):
    pipeline = build_churn_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, LogisticRegression()).fit(X_train, y_train)
    X_test = make_X(1, monthly_fee=[None])

    filled = pipeline.named_steps["impute"].transform(X_test)

    # The median of the train fees [10, 20, 30].
    assert filled.loc[0, "monthly_fee"] == 20.0


def test_feature_transformer_scales_numeric_passes_binary_and_one_hot_encodes_categorical(make_X):
    X = make_X(2, monthly_fee=[10.0, 30.0], autopay_enabled=[0, 1], region=["asia", "europe"])
    transformer = build_feature_transformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(X)

    transformed = pd.DataFrame(transformer.transform(X), columns=transformer.get_feature_names_out())

    assert transformed["monthly_fee"].tolist() == [-1.0, 1.0]
    assert transformed["autopay_enabled"].tolist() == [0, 1]
    assert transformed[["region_asia", "region_europe"]].values.tolist() == [[1, 0], [0, 1]]
    assert "region" not in transformed.columns


def test_feature_transformer_ignores_category_unseen_in_train_part(make_X):
    transformer = build_feature_transformer(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(
        make_X(2, region=["asia", "europe"])
    )

    transformed = pd.DataFrame(
        transformer.transform(make_X(1, region=["africa"])), columns=transformer.get_feature_names_out()
    )

    assert np.all(transformed[["region_asia", "region_europe"]] == 0)


def test_train_churn_model_fits_logreg_with_sklearn_defaults_by_default(X_train):
    pipeline = train_churn_model(X_train, y_train)

    check_is_fitted(pipeline)
    model = pipeline.named_steps["model"]
    assert type(model) is LogisticRegression
    assert model.class_weight is None


def test_train_churn_model_fits_classifier_of_model_type_with_hyperparameters(X_train):
    pipeline = train_churn_model(X_train, y_train, "random_forest", {"n_estimators": 5, "class_weight": "balanced"})

    check_is_fitted(pipeline)
    model = pipeline.named_steps["model"]
    assert type(model) is RandomForestClassifier
    assert (model.n_estimators, model.class_weight) == (5, "balanced")


@pytest.mark.parametrize(
    ("model_type", "hyperparameters", "reason"),
    [
        ("logreg", {"C": -1}, "The 'C' parameter of LogisticRegression must be a float in the range (0.0, inf]"),
        ("logreg", {"l1_ratio": 1}, "Solver lbfgs supports only 'l2' or None penalties, got l1 penalty"),
        ("random_forest", {"n_estimators": 0}, "The 'n_estimators' parameter of RandomForestClassifier must be"),
        ("random_forest", {"oob_score": True, "bootstrap": False}, "Out of bag estimation only available if bootstrap"),
    ],
)
def test_train_churn_model_rejects_hyperparameters_the_model_refuses(X_train, model_type, hyperparameters, reason):
    with pytest.raises(InvalidHyperparametersError) as error:
        train_churn_model(X_train, y_train, model_type, hyperparameters)

    assert str(error.value).startswith(f"Cannot train {model_type} with these hyperparameters: {reason}")


def test_train_churn_model_returns_separate_model_on_every_call(X_train):
    first, second = train_churn_model(X_train, y_train), train_churn_model(X_train, y_train)

    assert first is not second


def test_describe_hyperparameters_shows_main_ones_with_sklearn_defaults(X_train):
    pipeline = train_churn_model(X_train, y_train, "random_forest", {"n_estimators": 5})

    assert describe_hyperparameters(pipeline, "random_forest", {"n_estimators": 5}) == {
        "n_estimators": 5,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    }


def test_describe_hyperparameters_adds_requested_ones_beyond_the_main(X_train):
    requested = {"tol": 0.001, "C": 0.5}
    pipeline = train_churn_model(X_train, y_train, "logreg", requested)

    described = describe_hyperparameters(pipeline, "logreg", requested)

    assert list(described) == ["C", "l1_ratio", "solver", "max_iter", "class_weight", "tol"]
    assert (described["C"], described["tol"]) == (0.5, 0.001)


def fixed_predictions(churn_probabilities: list[float]) -> Mock:
    """Pipeline predicting these churn probabilities and the labels the 0.5 threshold makes of them."""
    pipeline = Mock(spec=Pipeline)
    probabilities = np.array(churn_probabilities)
    pipeline.predict_proba.return_value = np.column_stack([1 - probabilities, probabilities])
    pipeline.predict.return_value = (probabilities > 0.5).astype(int)
    return pipeline


def test_evaluate_churn_model_scores_test_predictions():
    pipeline = fixed_predictions([0.1, 0.9, 0.2, 0.3])
    X_test = pd.DataFrame(index=range(4))

    metrics = evaluate_churn_model(pipeline, X_test, pd.Series([0, 1, 1, 0]))

    pipeline.predict.assert_called_once_with(X_test)
    pipeline.predict_proba.assert_called_once_with(X_test)
    # Labels 0, 1, 0, 0: 3 of 4 rows are right; churn precision 1/1 and recall 1/2 give F1 = 2/3.
    assert metrics.accuracy == 0.75
    assert metrics.f1 == pytest.approx(2 / 3)
    # Of the 4 churner and non-churner pairs, only the churner at 0.2 is ranked below a non-churner, at 0.3.
    assert metrics.roc_auc == 0.75


def test_evaluate_churn_model_gives_zero_f1_to_model_that_never_predicts_churn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = evaluate_churn_model(
            fixed_predictions([0.1, 0.4, 0.2, 0.3]), pd.DataFrame(index=range(4)), pd.Series([0, 1, 0, 0])
        )

    # No probability crosses 0.5, yet the churner gets the highest one: the ranking is perfect.
    assert metrics == TrainMetrics(accuracy=0.75, f1=0.0, roc_auc=1.0)


@pytest.fixture
def tuning_X(make_X) -> pd.DataFrame:
    # Failed payments mostly decide churn, with one customer of each class against the rule, so scores differ by C.
    return make_X(
        12,
        failed_payments=[0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0],
        usage_hours=[float(hours) for hours in range(12)],
    )


tuning_y = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


def tune(tuning_X, **config) -> TuningResultChurn:
    return tune_churn_model(tuning_X, tuning_y, TuningConfigChurn(cv=3, **config))


def test_tune_churn_model_returns_best_combination_as_train_config(tuning_X):
    result = tune(tuning_X, param_grid={"C": [0.001, 1.0], "class_weight": [None, "balanced"]})

    best = result.candidates[0]
    assert result.best_config == TrainingConfigChurn(model_type="logreg", hyperparameters=best.hyperparameters)
    assert result.best_score == best.mean_score
    assert result.scoring == "f1"
    # The names are the classifier's ones, without the model__ prefix of the pipeline step.
    assert best.hyperparameters.keys() == {"C", "class_weight"}


def test_tune_churn_model_returns_top_candidates_best_first(tuning_X):
    result = tune(tuning_X, param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]})

    assert len(result.candidates) == TUNING_TOP_CANDIDATES
    scores = [candidate.mean_score for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(candidate.std_score >= 0 for candidate in result.candidates)


def test_tune_churn_model_returns_every_combination_of_a_small_grid(tuning_X):
    result = tune(tuning_X, param_grid={"C": [0.1, 1.0]})

    assert sorted(candidate.hyperparameters["C"] for candidate in result.candidates) == [0.1, 1.0]


def test_tune_churn_model_searches_default_grid_without_param_grid(tuning_X):
    # A forest grid costs seconds here; that a model type picks its own default grid is checked by the schema tests.
    result = tune(tuning_X, scoring="roc_auc")

    assert result.best_config.hyperparameters.keys() == {"C", "class_weight"}
    assert 0 <= result.best_score <= 1


def test_tune_churn_model_is_reproducible(tuning_X):
    config = {"model_type": "random_forest", "param_grid": {"n_estimators": [5, 10], "max_depth": [2, None]}}

    assert tune(tuning_X, **config) == tune(tuning_X, **config)


def test_tune_churn_model_rejects_unknown_hyperparameters(tuning_X):
    with pytest.raises(InvalidHyperparametersError, match="Unknown hyperparameters of logreg: n_estimators;"):
        tune(tuning_X, param_grid={"C": [1.0], "n_estimators": [10]})


@pytest.mark.parametrize(
    ("param_grid", "detail"),
    [
        (
            {"C": [0, 1.0]},
            (
                'Cannot train logreg with 1 of 2 combinations of the grid: {"C": 0}; the first one fails with: '
                "The 'C' parameter of LogisticRegression must be a float in the range (0.0, inf]"
            ),
        ),
        (
            {"l1_ratio": [1], "C": [0.1, 1.0]},
            (
                'Cannot train logreg with 2 of 2 combinations of the grid: {"C": 0.1, "l1_ratio": 1}, '
                '{"C": 1.0, "l1_ratio": 1}; the first one fails with: Solver lbfgs supports only'
            ),
        ),
    ],
)
def test_tune_churn_model_names_combinations_the_model_refuses(tuning_X, param_grid, detail):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(InvalidHyperparametersError) as error:
            tune(tuning_X, param_grid=param_grid)

    assert str(error.value).startswith(detail)
    # The refused combinations are reported by the error, not by sklearn warnings in the server log.
    assert caught == []


def test_tune_churn_model_needs_every_class_in_every_fold(tuning_X):
    with pytest.raises(DatasetNotTrainableError, match="7-fold cross-validation needs at least 7 customers"):
        tune_churn_model(tuning_X, tuning_y, TuningConfigChurn(cv=7))
