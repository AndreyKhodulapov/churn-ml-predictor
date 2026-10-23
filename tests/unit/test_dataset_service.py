import pytest

from src.core.constants import DATASET_PATH
from src.core.exceptions import DatasetLoadError
from src.schemas.dataset import ClassDistribution, DatasetInfo, DatasetRowChurn
from src.services.dataset import DatasetProcessor

FEATURE_NAMES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled",
]


@pytest.fixture
def loaded_processor(write_csv, csv_header, make_csv_row) -> DatasetProcessor:
    rows = [make_csv_row(churn) for churn in (0, 0, 0, 1)]
    processor = DatasetProcessor(write_csv("\n".join([csv_header, *rows]) + "\n"))
    processor.load()
    return processor


def test_processor_uses_default_dataset_path():
    assert DatasetProcessor().dataset_path == DATASET_PATH


def test_preview_returns_rows_as_models(loaded_processor, customer_features):
    rows = loaded_processor.preview(2)

    assert rows == [DatasetRowChurn(**customer_features, churn=0)] * 2


def test_preview_returns_all_rows_when_n_exceeds_dataset_size(loaded_processor):
    assert len(loaded_processor.preview(10)) == 4


def test_get_info_describes_dataset(loaded_processor):
    assert loaded_processor.get_info() == DatasetInfo(
        row_count=4,
        column_count=10,
        feature_names=FEATURE_NAMES,
        target_distribution={
            0: ClassDistribution(count=3, share=0.75),
            1: ClassDistribution(count=1, share=0.25),
        },
    )


def test_get_info_omits_absent_class(write_csv, csv_header, make_csv_row):
    processor = DatasetProcessor(write_csv(f"{csv_header}\n{make_csv_row(1)}\n"))
    processor.load()

    assert processor.get_info().target_distribution == {1: ClassDistribution(count=1, share=1.0)}


def test_load_reads_default_dataset():
    processor = DatasetProcessor()
    processor.load()

    info = processor.get_info()
    assert info.row_count == 2000
    assert info.feature_names == FEATURE_NAMES
    assert info.target_distribution == {
        0: ClassDistribution(count=1597, share=0.7985),
        1: ClassDistribution(count=403, share=0.2015),
    }


@pytest.mark.parametrize("method", [lambda p: p.preview(5), lambda p: p.get_info()])
def test_processor_requires_load_before_serving_data(method):
    with pytest.raises(RuntimeError, match="not loaded"):
        method(DatasetProcessor())


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


def test_load_reports_line_of_invalid_row(write_csv, csv_header, csv_row):
    invalid_row = csv_row.replace("europe", "antarctica")
    path = write_csv(f"{csv_header}\n{csv_row}\n{invalid_row}\n")

    with pytest.raises(DatasetLoadError, match="line 3"):
        DatasetProcessor(path).load()


def test_load_reports_missing_target_column(write_csv, csv_header, csv_row):
    path = write_csv(f"{csv_header.removesuffix(',churn')}\n{csv_row.removesuffix(',1')}\n")

    with pytest.raises(DatasetLoadError, match="Target column 'churn' not found"):
        DatasetProcessor(path).load()


def test_load_reports_missing_feature_column(write_csv, csv_header, csv_row):
    path = write_csv(f"{csv_header.replace('region,', '')}\n{csv_row.replace('europe,', '')}\n")

    with pytest.raises(DatasetLoadError, match="line 2"):
        DatasetProcessor(path).load()


def test_failed_load_keeps_processor_unloaded(write_csv, csv_header, csv_row):
    processor = DatasetProcessor(write_csv(f"{csv_header}\n{csv_row.replace('europe', 'antarctica')}\n"))

    with pytest.raises(DatasetLoadError):
        processor.load()
    with pytest.raises(RuntimeError, match="not loaded"):
        processor.get_info()
