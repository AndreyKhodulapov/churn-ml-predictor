class DatasetLoadError(Exception):
    """The dataset file cannot be read or contains rows that fail validation."""


class DatasetNotTrainableError(Exception):
    """The dataset is loaded but has too few labeled rows or churn classes to split and train on.

    The message names row counts and classes only, so it is safe to return to the client.
    """


class ModelLoadError(Exception):
    """The saved churn model file cannot be read or does not hold a churn model."""
