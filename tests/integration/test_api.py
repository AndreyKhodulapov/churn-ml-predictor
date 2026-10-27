from src.schemas.prediction import FeatureVectorChurn


async def test_root_reports_service_is_running(client):
    response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}


async def test_predict_rejects_unknown_region(client, customer_features):
    customer_features["region"] = "antarctica"

    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 422
    # The body is one customer or a list of them, so the error names the branch the input was checked against.
    assert response.json()["details"][0]["loc"] == ["body", "FeatureVectorChurn", "region"]
    # An object is checked as one customer only, so the list branch adds no error of its own.
    assert len(response.json()["details"]) == 1


async def test_predict_rejects_missing_feature(client, customer_features):
    del customer_features["failed_payments"]

    response = await client.post("/predict", json=customer_features)

    assert response.status_code == 422
    # The body is one customer or a list of them, so the error names the branch the input was checked against.
    assert response.json()["details"][0]["loc"] == ["body", "FeatureVectorChurn", "failed_payments"]
    assert len(response.json()["details"]) == 1


async def test_predict_names_the_invalid_customer_of_a_list(client, customer_features):
    invalid = {**customer_features, "monthly_fee": "abc"}

    response = await client.post("/predict", json=[customer_features, invalid])

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["details"]] == [
        ["body", "list[FeatureVectorChurn]", 1, "monthly_fee"]
    ]


async def test_predict_checks_a_body_of_another_type_as_one_customer(client):
    response = await client.post("/predict", json="customer")

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["details"]] == [["body", "FeatureVectorChurn"]]


async def test_openapi_describes_feature_schema(client):
    response = await client.get("/openapi.json")

    feature_schema = response.json()["components"]["schemas"]["FeatureVectorChurn"]
    assert set(feature_schema["required"]) == set(FeatureVectorChurn.model_fields)
