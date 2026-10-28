from typing import Annotated

from fastapi import APIRouter, Body, Depends

from src.core.openapi_examples import TRAIN_EXAMPLES, TRAIN_RESPONSES, TUNE_EXAMPLES, TUNE_RESPONSES
from src.schemas.learning import ModelStatus, TrainingConfigChurn, TrainMetrics, TuningConfigChurn, TuningResultChurn
from src.schemas.prediction import FeatureSchemaChurn
from src.services.model import train_and_save_churn_model, tune_churn_model_on_split
from src.services.model_storage import ChurnModelStore, get_model_store
from src.services.prediction import describe_features
from src.services.preprocessing import SplitData, get_split
from src.services.training_history import TrainingHistory, get_training_history

router = APIRouter(tags=["Model"], prefix="/model")


# A plain def: fitting is CPU-bound, FastAPI runs it in a thread pool instead of blocking the event loop.
@router.post(
    "/train",
    summary="Train the churn model and serve it",
    response_description="Accuracy, churn F1 and ROC-AUC of the new model on the test part",
    responses=TRAIN_RESPONSES,
)
def train_model(
    split: Annotated[SplitData, Depends(get_split)],
    store: Annotated[ChurnModelStore, Depends(get_model_store)],
    history: Annotated[TrainingHistory, Depends(get_training_history)],
    # Without a body the service trains logreg with the scikit-learn defaults.
    config: Annotated[TrainingConfigChurn, Body(default_factory=TrainingConfigChurn, openapi_examples=TRAIN_EXAMPLES)],
) -> TrainMetrics:
    """Train a churn model on the train part of the dataset, score it on the test part, save it and serve it.

    The body is optional. `model_type` is `logreg` (logistic regression) or `random_forest`, and `hyperparameters`
    are set over the scikit-learn defaults of that model under their scikit-learn names, for example
    `{"class_weight": "balanced"}`. Without a body the service trains a logistic regression with the defaults.
    `GET /model/status` shows the main hyperparameters the model was trained with, the defaults included.

    Answers `422` when the model does not know a hyperparameter or refuses its value, `409` when the dataset cannot be
    trained on and `503` while it is not available. A failed training keeps the previous model.
    """
    return train_and_save_churn_model(split, config, store, history)


# A plain def for the same reason as /train: the grid search fits one model per combination and fold.
@router.post(
    "/tune",
    summary="Find the best churn model hyperparameters by grid search",
    response_description="The best config, ready to be sent to POST /model/train, and the best combinations",
    responses=TUNE_RESPONSES,
)
def tune_model(
    split: Annotated[SplitData, Depends(get_split)],
    # Without a body the service searches the default logreg grid by churn F1.
    config: Annotated[TuningConfigChurn, Body(default_factory=TuningConfigChurn, openapi_examples=TUNE_EXAMPLES)],
) -> TuningResultChurn:
    """Score every combination of `param_grid` by cross-validation on the train part and return the best ones.

    `param_grid` maps scikit-learn hyperparameter names of `model_type` to the values to try, for example
    `{"C": [0.1, 1.0], "class_weight": [null, "balanced"]}`; without it the default grid of `model_type` is searched.
    `scoring` is `f1` (of the churn class), `accuracy` or `roc_auc`, and `cv` the number of folds.

    `best_config` is the body to send to `POST /model/train` as it is. `best_score` is the mean over the folds of the
    train part, so it differs from the test metrics `/model/train` reports. Tuning neither trains nor changes the
    served model.

    Answers `422` when the grid needs more fits (combinations times folds) than the service allows, the model does not
    know a hyperparameter or refuses some combination, `409` when the train part has fewer customers of a churn class
    than `cv` and `503` while the dataset is not available.
    """
    return tune_churn_model_on_split(split, config)


@router.get("/status")
def get_model_status(store: Annotated[ChurnModelStore, Depends(get_model_store)]) -> ModelStatus:
    return store.get_status()


@router.get(
    "/schema",
    summary="Describe the features a prediction request carries",
    response_description="Features in the order the model is trained on them, with their types and accepted values",
)
def get_model_schema() -> FeatureSchemaChurn:
    """List the features `POST /predict` expects for every customer, in the order the churn model is trained on them.

    For each feature: `kind` is how the model treats it (`numeric` ones are scaled, `binary` 0/1 flags are passed as
    they are, `categorical` ones are one-hot encoded), `type` is its JSON Schema type (`number`, `integer` or
    `string`), `allowed_values` the only values accepted, if there are such, and `minimum` the smallest one.

    The features are set by the service rather than learned, so the answer is the same before the first training.
    """
    return describe_features()
