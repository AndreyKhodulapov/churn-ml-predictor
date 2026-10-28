import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.services.dataset import DatasetProcessor, get_dataset
from src.services.model_storage import ChurnModelStore, get_model_store
from src.services.preprocessing import get_split
from src.services.training_history import TrainingHistory, get_training_history


@pytest.fixture(autouse=True)
def model_store(tmp_path, monkeypatch) -> ChurnModelStore:
    """Point the service store at a temporary file, so models trained by tests never land in the project models/.

    The path is patched rather than the dependency overridden: tests clear dependency_overrides on their own.
    """
    monkeypatch.setattr("src.services.model_storage.MODEL_PATH", tmp_path / "models" / "churn_model.joblib")
    get_model_store.cache_clear()
    yield get_model_store()
    get_model_store.cache_clear()


@pytest.fixture(autouse=True)
def training_history(tmp_path, monkeypatch) -> TrainingHistory:
    """Point the service history at a temporary file, for the same reason as model_store."""
    monkeypatch.setattr("src.services.training_history.HISTORY_PATH", tmp_path / "models" / "training_history.json")
    get_training_history.cache_clear()
    yield get_training_history()
    get_training_history.cache_clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def documented_errors(client):
    """Error answers an operation shows in /docs, by example name, for a method, a path and a status."""
    paths = (await client.get("/openapi.json")).json()["paths"]

    def examples(method: str, path: str, status: int) -> dict[str, dict]:
        content = paths[path][method]["responses"][str(status)]["content"]["application/json"]
        assert content["schema"] == {"$ref": "#/components/schemas/ErrorResponse"}
        # FastAPI drops None values from the whole OpenAPI document; null is the default of details, so it is put back
        # to compare the examples with real answers.
        return {name: {"details": None, **example["value"]} for name, example in content["examples"].items()}

    return examples


@pytest.fixture
def serve_csv(write_csv, monkeypatch):
    """Serve a dataset written from CSV text to every endpoint, the cached split included."""

    def serve(content: str) -> DatasetProcessor:
        processor = DatasetProcessor(write_csv(content))
        processor.load()
        app.dependency_overrides[get_dataset] = lambda: processor
        # get_split() calls get_dataset() directly rather than through Depends, so the override does not reach it.
        monkeypatch.setattr("src.services.preprocessing.get_dataset", lambda: processor)
        get_split.cache_clear()
        return processor

    yield serve
    app.dependency_overrides.clear()
    # Drop the split of the served dataset so later tests get the default one again.
    get_split.cache_clear()
