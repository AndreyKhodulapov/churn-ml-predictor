import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.core.constants import DATASET_PATH
from src.core.exceptions import DatasetLoadError
from src.schemas.dataset import DatasetInfo, DatasetRowChurn
from src.services.utils import class_distribution

logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(self, dataset_path: Path = DATASET_PATH, target_column: str = "churn") -> None:
        self.dataset_path = dataset_path
        self.target_column = target_column
        self._df: pd.DataFrame | None = None
        self._rows: list[DatasetRowChurn] = []

    def read_csv(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.dataset_path)
        except FileNotFoundError as e:
            raise DatasetLoadError(f"Dataset file not found: {self.dataset_path}") from e
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise DatasetLoadError(f"Cannot read dataset {self.dataset_path}: {e}") from e

    def load(self) -> None:
        """Read and validate the dataset once; the other methods serve the loaded data."""
        logger.info("Loading dataset from %s", self.dataset_path)
        df = self.read_csv()
        if self.target_column not in df.columns:
            raise DatasetLoadError(f"Target column {self.target_column!r} not found in {self.dataset_path}")
        self._rows = self._validate_rows(df)
        self._df = df
        logger.info("Loaded %d rows from %s", len(self._rows), self.dataset_path)

    def preview(self, n: int) -> list[DatasetRowChurn]:
        self._ensure_loaded()
        return self._rows[:n]

    def get_info(self) -> DatasetInfo:
        df = self._ensure_loaded()
        return DatasetInfo(
            row_count=len(df),
            column_count=len(df.columns),
            feature_names=[str(column) for column in df.columns if column != self.target_column],
            target_distribution=class_distribution(df[self.target_column]),
        )

    def _validate_rows(self, df: pd.DataFrame) -> list[DatasetRowChurn]:
        rows = []
        for index, record in enumerate(df.to_dict("records")):
            try:
                rows.append(DatasetRowChurn.model_validate(record))
            except ValidationError as e:
                # +2: line numbers start at 1 and the first line is the header.
                raise DatasetLoadError(f"Invalid row at line {index + 2} of {self.dataset_path}: {e}") from e
        return rows

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Dataset is not loaded, call load() first")
        return self._df


@lru_cache
def get_dataset() -> DatasetProcessor:
    """Load the default dataset once and reuse the processor for every request."""
    processor = DatasetProcessor()
    processor.load()
    return processor
