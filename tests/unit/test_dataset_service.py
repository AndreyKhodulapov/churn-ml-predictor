from pathlib import Path

import pytest

from src.core.constants import DATASET_PATH
from src.core.exceptions import DatasetLoadError
from src.schemas.dataset import DatasetRowChurn
from src.services.dataset import DatasetProcessor


@pytest.fixture
def write_csv(tmp_path):
    def write(content: str) -> Path:
        path = tmp_path / "dataset.csv"
        path.write_text(content)
        return path

    return write


@pytest.fixture
def csv_header(customer_features) -> str:
    return ",".join([*customer_features, "churn"])


@pytest.fixture
def csv_row(customer_features) -> str:
    return ",".join(str(value) for value in [*customer_features.values(), 1])


def test_processor_uses_default_dataset_path():
    assert DatasetProcessor().dataset_path == DATASET_PATH


def test_make_dataset_converts_rows_to_models(write_csv, csv_header, csv_row, customer_features):
    path = write_csv(f"{csv_header}\n{csv_row}\n{csv_row}\n")

    rows = DatasetProcessor(path).make_dataset()

    assert len(rows) == 2
    assert rows[0] == DatasetRowChurn(**customer_features, churn=1)


def test_make_dataset_loads_default_dataset():
    rows = DatasetProcessor().make_dataset()

    assert len(rows) == 2000
    assert all(isinstance(row, DatasetRowChurn) for row in rows)


def test_read_csv_reports_missing_file(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(DatasetLoadError, match="not found"):
        DatasetProcessor(path).read_csv()


def test_read_csv_reports_empty_file(write_csv):
    path = write_csv("")

    with pytest.raises(DatasetLoadError, match="Cannot read"):
        DatasetProcessor(path).read_csv()


def test_read_csv_reports_directory(tmp_path):
    with pytest.raises(DatasetLoadError, match="Cannot read"):
        DatasetProcessor(tmp_path).read_csv()


def test_make_dataset_reports_line_of_invalid_row(write_csv, csv_header, csv_row):
    invalid_row = csv_row.replace("europe", "antarctica")
    path = write_csv(f"{csv_header}\n{csv_row}\n{invalid_row}\n")

    with pytest.raises(DatasetLoadError, match="line 3"):
        DatasetProcessor(path).make_dataset()


def test_make_dataset_reports_missing_column(write_csv, csv_header, csv_row):
    path = write_csv(f"{csv_header.removesuffix(',churn')}\n{csv_row.removesuffix(',1')}\n")

    with pytest.raises(DatasetLoadError, match="line 2"):
        DatasetProcessor(path).make_dataset()
