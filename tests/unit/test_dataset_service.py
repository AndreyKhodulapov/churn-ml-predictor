import pytest

from src.core.constants import DATASET_PATH
from src.core.exceptions import DatasetLoadError
from src.schemas.dataset import ClassDistribution, DatasetInfo, DatasetRowChurn
from src.services.dataset import DatasetProcessor, get_dataset, is_dataset_loaded

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
        target_missing=ClassDistribution(count=0, share=0.0),
    )


def test_get_info_omits_absent_class(write_csv, csv_header, make_csv_row):
    processor = DatasetProcessor(write_csv(f"{csv_header}\n{make_csv_row(1)}\n"))
    processor.load()

    assert processor.get_info().target_distribution == {1: ClassDistribution(count=1, share=1.0)}


def test_load_keeps_missing_values(write_csv, csv_header, csv_row):
    row_with_gaps = csv_row.replace("29.99", "").replace("europe", "").removesuffix("1")
    processor = DatasetProcessor(write_csv(f"{csv_header}\n{csv_row}\n{row_with_gaps}\n"))
    processor.load()

    row = processor.preview(2)[1]
    assert (row.monthly_fee, row.region, row.churn) == (None, None, None)
    assert processor.get_dataframe().isna().sum().sum() == 3


def test_get_info_reports_unlabeled_rows_apart_from_classes(write_csv, csv_header, make_csv_row):
    rows = [make_csv_row(0), make_csv_row(1), make_csv_row(1).removesuffix("1")]
    processor = DatasetProcessor(write_csv("\n".join([csv_header, *rows]) + "\n"))
    processor.load()

    info = processor.get_info()
    assert info.row_count == 3
    assert info.target_distribution == {
        0: ClassDistribution(count=1, share=1 / 3),
        1: ClassDistribution(count=1, share=1 / 3),
    }
    assert info.target_missing == ClassDistribution(count=1, share=1 / 3)


def test_get_info_of_header_only_dataset(write_csv, csv_header):
    processor = DatasetProcessor(write_csv(f"{csv_header}\n"))
    processor.load()

    info = processor.get_info()
    assert (info.row_count, info.target_distribution) == (0, {})
    assert info.target_missing == ClassDistribution(count=0, share=0.0)


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


def test_load_reports_row_with_infinity(write_csv, csv_header, csv_row):
    # pandas reads inf as infinity, while an empty cell is a gap the pipeline fills.
    path = write_csv(f"{csv_header}\n{csv_row}\n{csv_row.replace('29.99', 'inf')}\n")

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


@pytest.fixture
def default_dataset_path(tmp_path, monkeypatch):
    """Point get_dataset() at a temporary file with an empty cache; the cache is emptied again for later tests."""
    path = tmp_path / "default.csv"
    monkeypatch.setattr("src.services.dataset.DATASET_PATH", path)
    get_dataset.cache_clear()
    yield path
    get_dataset.cache_clear()


def test_dataset_is_reported_loaded_once_a_retry_succeeds(default_dataset_path, csv_header, csv_row):
    with pytest.raises(DatasetLoadError):
        get_dataset()
    assert not is_dataset_loaded()

    default_dataset_path.write_text(f"{csv_header}\n{csv_row}\n")
    get_dataset()

    assert is_dataset_loaded()
