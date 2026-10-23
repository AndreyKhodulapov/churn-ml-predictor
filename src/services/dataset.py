import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.core.constants import DATASET_PATH
from src.core.exceptions import DatasetLoadError
from src.schemas.dataset import DatasetRowChurn

logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(self, dataset_path: Path = DATASET_PATH):
        self.dataset_path = dataset_path

    def read_csv(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.dataset_path)
        except FileNotFoundError as e:
            raise DatasetLoadError(f"Dataset file not found: {self.dataset_path}") from e
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise DatasetLoadError(f"Cannot read dataset {self.dataset_path}: {e}") from e

    def make_dataset(self) -> list[DatasetRowChurn]:
        logger.info("Loading dataset from %s", self.dataset_path)
        rows = []
        for index, record in enumerate(self.read_csv().to_dict("records")):
            try:
                rows.append(DatasetRowChurn.model_validate(record))
            except ValidationError as e:
                # +2: line numbers start at 1 and the first line is the header.
                raise DatasetLoadError(f"Invalid row at line {index + 2} of {self.dataset_path}: {e}") from e
        logger.info("Loaded %d rows from %s", len(rows), self.dataset_path)
        return rows


@lru_cache
def get_dataset() -> list[DatasetRowChurn]:
    """Load the default dataset once and reuse it for every request."""
    return DatasetProcessor().make_dataset()
