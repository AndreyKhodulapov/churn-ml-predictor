from src.main import app, lifespan
from src.services.dataset import get_dataset
from src.services.preprocessing import get_split


async def test_health_reports_ok_when_the_dataset_is_ready_and_a_model_is_served(client):
    get_split()
    await client.post("/model/train")

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "dataset_loaded": True,
        "dataset_trainable": True,
        "model_available": True,
    }


async def test_health_reports_service_without_a_model_as_degraded_but_up(client):
    get_split()

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "dataset_loaded": True,
        "dataset_trainable": True,
        "model_available": False,
    }


async def test_health_reports_dataset_that_is_not_loaded_without_loading_it(client):
    get_dataset.cache_clear()
    get_split.cache_clear()

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "dataset_loaded": False,
        "dataset_trainable": False,
        "model_available": False,
    }
    assert get_dataset.cache_info().currsize == 0


async def test_health_reports_dataset_that_cannot_be_trained_on(client, serve_csv, csv_header):
    get_dataset()
    serve_csv(f"{csv_header}\n")
    assert (await client.post("/model/train")).status_code == 409

    response = await client.get("/health")

    assert response.json()["dataset_loaded"] is True
    assert response.json()["dataset_trainable"] is False


async def test_health_reports_what_the_startup_prepared(client):
    get_dataset.cache_clear()
    get_split.cache_clear()

    async with lifespan(app):
        response = await client.get("/health")

    # The test store starts empty, so the startup has no model to load.
    assert response.json() == {
        "status": "degraded",
        "dataset_loaded": True,
        "dataset_trainable": True,
        "model_available": False,
    }
