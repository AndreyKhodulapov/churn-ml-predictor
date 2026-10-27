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


class ModelFeaturesMismatchError(Exception):
    """The served churn model was trained on other features than a prediction request carries.

    It is a fault of the service, not of the request, so the client gets a bare 500 and the message goes to the log.
    """


class ModelNotTrainedError(Exception):
    """A prediction is requested before any churn model has been trained or loaded."""
