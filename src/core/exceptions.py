class DatasetLoadError(Exception):
    """The dataset file cannot be read or contains rows that fail validation."""


class DatasetNotTrainableError(Exception):
    """The dataset is loaded but has too few labeled rows or churn classes to split and train on.

    The message names row counts and classes only, so it is safe to return to the client.
    """


class InvalidHyperparametersError(Exception):
    """The hyperparameters requested for training are not accepted by the chosen model.

    The message names the model type and hyperparameters only, so it is safe to return to the client.
    """


class ModelLoadError(Exception):
    """The saved churn model file cannot be read or does not hold a churn model."""


class HistoryLoadError(Exception):
    """The saved training history file cannot be read or does not hold training records."""


class ModelFeaturesMismatchError(Exception):
    """The served churn model was trained on other features than a prediction request carries.

    It is a fault of the service, not of the request: predict_churn() reports it as PredictionError.
    """


class PredictionError(Exception):
    """The served churn model fails to predict for a request that has passed validation.

    It is a fault of the service, not of the request. The message and its cause name model internals, so they go to
    the log and the client gets a generic message.
    """


class ModelNotTrainedError(Exception):
    """A prediction is requested before any churn model has been trained or loaded.

    The message names the fix and nothing of the server, so it is safe to return to the client.
    """

    def __init__(self, message: str = "Churn model is not trained yet, call POST /model/train first") -> None:
        super().__init__(message)
