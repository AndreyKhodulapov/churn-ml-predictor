import logging

import pytest

from src.core.openapi_examples import (
    LOYAL_CUSTOMER,
    LOYAL_CUSTOMER_PREDICTION,
    MODEL_NOT_TRAINED_ERROR,
    RISKY_CUSTOMER,
    RISKY_CUSTOMER_PREDICTION,
)


@pytest.fixture
async def trained_client(client):
    response = await client.post("/model/train")
    assert response.status_code == 200
    return client


async def test_predict_answers_503_without_trained_model(client, customer_features):
    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 503
    assert response.json() == MODEL_NOT_TRAINED_ERROR


@pytest.fixture
def broken_model(trained_client, model_store, monkeypatch):
    def broken_predict_proba(X):
        raise ValueError("Model internals in /secret/model.joblib")

    monkeypatch.setattr(model_store.model.pipeline, "predict_proba", broken_predict_proba)
    return trained_client


async def test_predict_answers_500_when_model_fails(broken_model, customer_features):
    response = await broken_model.post("/predict", json=customer_features)

    assert response.status_code == 500
    assert response.json() == {
        "code": "prediction_failed",
        "message": "Churn model failed to make a prediction",
        "details": None,
    }


async def test_predict_logs_the_cause_of_a_model_failure_and_hides_it(broken_model, customer_features, caplog):
    with caplog.at_level(logging.ERROR):
        response = await broken_model.post("/predict", json=customer_features)

    assert "/secret/model.joblib" not in response.text
    assert "Traceback" not in response.text
    assert "Churn model failed to predict, batch size 1: Model internals in /secret/model.joblib" in caplog.text
    # The traceback reaches the log, down to the exception the pipeline raised.
    assert "ValueError" in caplog.text


async def test_predict_one_customer_returns_one_prediction(trained_client, customer_features):
    response = await trained_client.post("/predict", json=customer_features)

    assert response.status_code == 200
    [prediction] = response.json()
    assert prediction.keys() == {"churn", "probability_no_churn", "probability_churn"}
    assert prediction["churn"] in (0, 1)
    assert prediction["probability_no_churn"] + prediction["probability_churn"] == pytest.approx(1)
    # The predicted class is the more probable one.
    assert prediction["churn"] == int(prediction["probability_churn"] > 0.5)


async def test_predict_several_customers_keeps_their_order(trained_client):
    response = await trained_client.post("/predict", json=[LOYAL_CUSTOMER, RISKY_CUSTOMER, LOYAL_CUSTOMER])

    assert response.status_code == 200
    loyal, risky, loyal_again = response.json()
    assert loyal == loyal_again
    assert risky["probability_churn"] > loyal["probability_churn"]


async def test_predict_rejects_empty_list(trained_client):
    response = await trained_client.post("/predict", json=[])

    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_error",
        "message": "Request failed validation, see details for every error",
        "details": [
            {
                "loc": ["body", "list[FeatureVectorChurn]"],
                "message": "List should have at least 1 item after validation, not 0",
                "type": "too_short",
            }
        ],
    }


async def test_predict_rejects_empty_list_before_looking_for_model(client):
    response = await client.post("/predict", json=[])

    assert response.status_code == 422


async def test_predict_rejects_invalid_customer_in_list(trained_client):
    response = await trained_client.post("/predict", json=[LOYAL_CUSTOMER, {**RISKY_CUSTOMER, "region": "antarctica"}])

    assert response.status_code == 422


async def test_openapi_shows_predict_examples(client):
    response = await client.get("/openapi.json")

    operation = response.json()["paths"]["/predict"]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]
    assert examples["one_customer"]["value"] == LOYAL_CUSTOMER
    assert examples["several_customers"]["value"] == [LOYAL_CUSTOMER, RISKY_CUSTOMER]
    # Picking the branch by the shape of the body leaves the schema a choice of one customer or a non-empty list.
    assert operation["requestBody"]["content"]["application/json"]["schema"]["oneOf"] == [
        {"$ref": "#/components/schemas/FeatureVectorChurn"},
        {"type": "array", "items": {"$ref": "#/components/schemas/FeatureVectorChurn"}, "minItems": 1},
    ]
    response_examples = operation["responses"]["200"]["content"]["application/json"]["examples"]
    assert response_examples["several_customers"]["value"] == [LOYAL_CUSTOMER_PREDICTION, RISKY_CUSTOMER_PREDICTION]
    # FastAPI drops None values from the whole OpenAPI document, the null details of the example included.
    assert operation["responses"]["503"]["content"]["application/json"]["example"] == {
        "code": MODEL_NOT_TRAINED_ERROR["code"],
        "message": MODEL_NOT_TRAINED_ERROR["message"],
    }
    assert "Answers `503` until a model is trained" in operation["description"]


async def test_openapi_response_examples_match_the_model_trained_on_bundled_dataset(trained_client):
    response = await trained_client.post("/predict", json=[LOYAL_CUSTOMER, RISKY_CUSTOMER])

    for prediction, example in zip(
        response.json(), [LOYAL_CUSTOMER_PREDICTION, RISKY_CUSTOMER_PREDICTION], strict=True
    ):
        assert prediction == pytest.approx(example, abs=1e-3)
