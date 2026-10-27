import pandas as pd
import pytest

from src.core.exceptions import ModelNotTrainedError
from src.schemas.learning import TrainMetrics
from src.schemas.prediction import FeatureVectorChurn
from src.services.learning import train_churn_model
from src.services.model_storage import TrainedChurnModel
from src.services.prediction import predict_churn


@pytest.fixture
def make_customer(customer_features):
    def make(failed_payments: int) -> FeatureVectorChurn:
        return FeatureVectorChurn(**{**customer_features, "failed_payments": failed_payments})

    return make


@pytest.fixture
def model(make_customer) -> TrainedChurnModel:
    # Churn is decided by failed payments alone, so even a tiny model separates the customers.
    X = pd.DataFrame([make_customer(value).model_dump() for value in [0, 0, 0, 5, 5, 5]])
    pipeline = train_churn_model(X, pd.Series([0, 0, 0, 1, 1, 1]))
    return TrainedChurnModel(pipeline, TrainMetrics(accuracy=1.0, f1=1.0), "logreg", {})


def test_predicts_one_customer(model, make_customer):
    [prediction] = predict_churn(model, make_customer(5))

    assert prediction.churn == 1
    assert prediction.probability_churn > 0.5
    assert prediction.probability_no_churn + prediction.probability_churn == pytest.approx(1)


def test_predicts_several_customers_in_order(model, make_customer):
    predictions = predict_churn(model, [make_customer(0), make_customer(5)])

    assert [prediction.churn for prediction in predictions] == [0, 1]


def test_matches_the_pipeline_on_the_same_rows(model, make_customer):
    customers = [make_customer(0), make_customer(5)]

    predictions = predict_churn(model, customers)

    X = pd.DataFrame([customer.model_dump() for customer in customers])
    assert [prediction.probability_churn for prediction in predictions] == list(model.pipeline.predict_proba(X)[:, 1])


def test_returns_no_predictions_for_no_customers(model):
    assert predict_churn(model, []) == []


def test_refuses_without_trained_model(make_customer):
    with pytest.raises(ModelNotTrainedError):
        predict_churn(None, make_customer(0))
