import pytest

from src.core.constants import LOYAL_CUSTOMER, RISKY_CUSTOMER


@pytest.fixture
async def trained_client(client):
    response = await client.post("/model/train")
    assert response.status_code == 200
    return client


async def test_predict_answers_503_without_trained_model(client, customer_features):
    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 503
    assert response.json() == {"detail": "Churn model is not trained yet, call POST /model/train first"}


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


async def test_predict_empty_list_returns_no_predictions(trained_client):
    response = await trained_client.post("/predict", json=[])

    assert response.status_code == 200
    assert response.json() == []


async def test_predict_rejects_invalid_customer_in_list(trained_client):
    response = await trained_client.post("/predict", json=[LOYAL_CUSTOMER, {**RISKY_CUSTOMER, "region": "antarctica"}])

    assert response.status_code == 422


async def test_openapi_shows_predict_examples(client):
    response = await client.get("/openapi.json")

    operation = response.json()["paths"]["/predict"]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]
    assert examples["one_customer"]["value"] == LOYAL_CUSTOMER
    assert examples["several_customers"]["value"] == [LOYAL_CUSTOMER, RISKY_CUSTOMER]
    assert "503" in operation["responses"]
