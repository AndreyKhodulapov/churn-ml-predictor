import logging
import pandas as pd

from src.constants import DATASET_PATH
from src.schemas import DatasetRowChurn


logger = logging.getLogger(__name__)


class DatasetProcessor:
    def __init__(self):
        self.dataset_path = DATASET_PATH
        logger.info(f"Loading dataset from {self.dataset_path}")

    def read_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path)

    def make_dataset(self) -> list[DatasetRowChurn]:
        dataset = self.read_csv()
        return [DatasetRowChurn(**row) for row in dataset.to_dict("records")]   # type: ignore


processor = DatasetProcessor()
DATASET = processor.make_dataset()
DF = processor.read_csv()
