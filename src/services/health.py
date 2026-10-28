from src.schemas.health import HealthStatus
from src.services.dataset import is_dataset_loaded
from src.services.model_storage import ChurnModelStore
from src.services.split import is_dataset_trainable


def check_health(store: ChurnModelStore) -> HealthStatus:
    """What the service can do right now, from the state it already holds.

    Nothing is loaded here: /health is polled often, and a load of the dataset reads and validates every row.
    A dataset that failed to load is retried by the next request that needs it, and /health reports the result.
    """
    dataset_loaded = is_dataset_loaded()
    dataset_trainable = is_dataset_trainable()
    model_available = store.model is not None
    return HealthStatus(
        status="ok" if dataset_loaded and dataset_trainable and model_available else "degraded",
        dataset_loaded=dataset_loaded,
        dataset_trainable=dataset_trainable,
        model_available=model_available,
    )
