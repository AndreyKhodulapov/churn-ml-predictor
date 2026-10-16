from src.schemas import FeatureVectorChurn


async def test_root_reports_service_is_running(client):
    response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}


async def test_predict_echoes_features(client, customer_features):
    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 200
    assert response.json() == customer_features


async def test_predict_rejects_unknown_region(client, customer_features):
    customer_features["region"] = "antarctica"

    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "region"]


async def test_predict_rejects_missing_feature(client, customer_features):
    del customer_features["failed_payments"]

    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "failed_payments"]


async def test_openapi_describes_feature_schema(client):
    response = await client.get("/openapi.json")

    feature_schema = response.json()["components"]["schemas"]["FeatureVectorChurn"]
    assert set(feature_schema["required"]) == set(FeatureVectorChurn.model_fields)
