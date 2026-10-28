import logging
import re

import pandas as pd
import pytest

from src.core.config import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from src.core.exceptions import DatasetNotTrainableError
from src.ml.preprocessing import PreparedData, build_imputer, prepare_data, split_data
from src.schemas.dataset import ClassDistribution, DatasetRowChurn, SplitDataInfo
from src.schemas.prediction import FeatureVectorChurn
from src.services.dataset import DatasetProcessor
from src.services.split import check_class_shares, describe_split, get_split, is_dataset_trainable


@pytest.fixture
def make_df(customer_features):
    def make(*labels: float | None, **overrides: list) -> pd.DataFrame:
        df = pd.DataFrame([{**customer_features, TARGET_COLUMN: label} for label in labels])
        for column, values in overrides.items():
            df[column] = values
        return df

    return make


def test_feature_lists_match_dataset_and_request_schemas():
    assert len(FEATURES) == len(set(FEATURES))
    assert set(FEATURES) == DatasetRowChurn.model_fields.keys() - {TARGET_COLUMN}
    # /predict builds its rows from FeatureVectorChurn, so it has to carry every feature the model is trained on.
    assert set(FEATURES) == FeatureVectorChurn.model_fields.keys()
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert set(BINARY_FEATURES) <= set(NUMERIC_FEATURES)


def test_prepare_data_separates_features_and_target(make_df):
    prepared = prepare_data(make_df(0, 1, 1))

    assert list(prepared.X.columns) == list(FEATURES)
    assert prepared.y.tolist() == [0, 1, 1]
    assert (prepared.numeric_features, prepared.categorical_features) == (NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def test_prepare_data_drops_rows_without_label(make_df, caplog):
    with caplog.at_level(logging.WARNING):
        prepared = prepare_data(make_df(0, None, 1))

    assert prepared.y.tolist() == [0, 1]
    assert prepared.y.dtype == "int64"
    assert prepared.X.index.equals(prepared.y.index)
    assert "Dropped 1 rows" in caplog.text


def test_prepare_data_keeps_feature_gaps_and_reports_them(make_df, caplog):
    df = make_df(0, 1, monthly_fee=[None, 10.0], region=["asia", None])

    with caplog.at_level(logging.WARNING):
        prepared = prepare_data(df)

    assert prepared.X.isna().sum().sum() == 2
    assert "'monthly_fee': 1" in caplog.text
    assert "'region': 1" in caplog.text


def test_prepare_data_does_not_modify_input(make_df):
    df = make_df(0, None)
    original = df.copy()

    prepare_data(df)

    pd.testing.assert_frame_equal(df, original)


def test_imputer_fills_gaps_with_train_statistics(make_df):
    labels = [0, 1, 0, 1, 0]
    df = make_df(
        *labels, monthly_fee=[10.0, 20.0, 90.0, None, 1000.0], region=["asia", "asia", "europe", None, "europe"]
    )
    X = prepare_data(df).X
    # Values of the test part (1000.0, "europe") must not affect what the gaps are filled with.
    train, test = X.iloc[:3], X.iloc[3:]
    imputer = build_imputer(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(train)

    filled = imputer.transform(test)

    assert list(filled.columns) == list(FEATURES)
    # The train median; the mean would be 40.0.
    assert filled.loc[3, "monthly_fee"] == 20.0
    assert filled.loc[3, "region"] == "asia"


def test_imputer_keeps_binary_feature_binary(make_df):
    X = prepare_data(make_df(0, 1, 0, 1, autopay_enabled=[1, 1, 0, None])).X
    imputer = build_imputer(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(X.iloc[:3])

    filled = imputer.transform(X.iloc[3:])

    # The mean would be 0.67; the most frequent value keeps the flag a 0/1 number.
    assert filled.loc[3, "autopay_enabled"] == 1


@pytest.fixture
def prepared(make_df) -> PreparedData:
    # 15 stayed and 5 left: the 20% test part gets 3 and 1, and every class has the 2 rows stratification needs.
    labels = [0] * 15 + [1] * 5
    return prepare_data(make_df(*labels, monthly_fee=[*range(19), None]))


def test_split_data_holds_out_test_share(prepared):
    split = split_data(prepared)

    assert (len(split.X_train), len(split.X_test)) == (16, 4)
    assert (len(split.y_train), len(split.y_test)) == (16, 4)


def test_split_data_parts_do_not_overlap_and_cover_all_rows(prepared):
    split = split_data(prepared)

    assert not set(split.X_train.index) & set(split.X_test.index)
    assert set(split.X_train.index) | set(split.X_test.index) == set(prepared.X.index)


def test_split_data_keeps_features_and_labels_aligned(prepared):
    split = split_data(prepared)

    assert split.X_train.index.equals(split.y_train.index)
    assert split.X_test.index.equals(split.y_test.index)


def test_split_data_keeps_churn_share_in_both_parts(prepared):
    split = split_data(prepared)

    assert split.y_train.mean() == split.y_test.mean() == prepared.y.mean() == 0.25


def test_split_data_is_reproducible(prepared):
    first, second = split_data(prepared), split_data(prepared)

    assert first.X_train.index.equals(second.X_train.index)
    assert first.X_test.index.equals(second.X_test.index)


def test_split_data_keeps_feature_gaps(prepared):
    split = split_data(prepared)

    assert split.X_train["monthly_fee"].isna().sum() + split.X_test["monthly_fee"].isna().sum() == 1


@pytest.mark.parametrize(
    ("labels", "reason"),
    [
        ([None, None], "Dataset has no labeled rows"),
        ([0] * 10, "Dataset has only churn class 0, a classifier needs both"),
        ([1] * 10, "Dataset has only churn class 1, a classifier needs both"),
        # Stratification needs 2 rows of every class.
        ([0] * 9 + [1], "least populated classes in y have only 1 member"),
        # 20% of 4 rows is a 1-row test part, which cannot hold both classes.
        ([0, 0, 1, 1], "test_size = 1 should be greater or equal to the number of classes = 2"),
    ],
)
def test_split_data_rejects_dataset_it_cannot_split_for_training(make_df, labels, reason):
    prepared = prepare_data(make_df(*labels))

    with pytest.raises(DatasetNotTrainableError, match=re.escape(reason)):
        split_data(prepared)


def test_split_data_accepts_smallest_dataset_with_both_classes_in_both_parts(make_df):
    split = split_data(prepare_data(make_df(0, 0, 0, 1, 1, 1)))

    assert sorted(split.y_test) == [0, 1]
    assert sorted(split.y_train) == [0, 0, 1, 1]


def test_get_split_splits_default_dataset_once():
    split = get_split()

    assert (len(split.X_train), len(split.X_test)) == (1600, 400)
    assert list(split.X_train.columns) == list(split.X_test.columns) == list(FEATURES)
    assert get_split() is split


def test_get_split_logs_sizes_of_both_parts(caplog):
    # Cleared to split again: an earlier test may have cached the split, and only splitting logs.
    get_split.cache_clear()

    with caplog.at_level(logging.INFO):
        get_split()

    assert "Split the dataset into 1600 train and 400 test rows" in caplog.text


def test_dataset_is_reported_trainable_only_after_a_split_succeeds(write_csv, csv_header, monkeypatch):
    header_only = DatasetProcessor(write_csv(f"{csv_header}\n"))
    header_only.load()
    monkeypatch.setattr("src.services.split.get_dataset", lambda: header_only)
    get_split.cache_clear()

    with pytest.raises(DatasetNotTrainableError):
        get_split()
    assert not is_dataset_trainable()

    monkeypatch.undo()
    get_split()

    assert is_dataset_trainable()


def test_describe_split_reports_sizes_and_distributions(prepared):
    info = describe_split(split_data(prepared))

    assert info == SplitDataInfo(
        train_size=16,
        train_distribution={0: ClassDistribution(count=12, share=0.75), 1: ClassDistribution(count=4, share=0.25)},
        test_size=4,
        test_distribution={0: ClassDistribution(count=3, share=0.75), 1: ClassDistribution(count=1, share=0.25)},
    )


def make_info(train_shares: dict[int, float], test_shares: dict[int, float]) -> SplitDataInfo:
    def distribution(shares: dict[int, float], size: int) -> dict[int, ClassDistribution]:
        return {label: ClassDistribution(count=round(share * size), share=share) for label, share in shares.items()}

    return SplitDataInfo(
        train_size=80,
        train_distribution=distribution(train_shares, 80),
        test_size=20,
        test_distribution=distribution(test_shares, 20),
    )


def test_check_class_shares_logs_matching_shares_without_warning(caplog):
    with caplog.at_level(logging.INFO):
        check_class_shares(make_info({0: 0.8, 1: 0.2}, {0: 0.805, 1: 0.195}))

    assert "Churn shares in train: {0: 0.8, 1: 0.2}, in test: {0: 0.805, 1: 0.195}" in caplog.text
    assert not [record for record in caplog.records if record.levelno == logging.WARNING]


def test_check_class_shares_warns_about_differing_class(caplog):
    with caplog.at_level(logging.WARNING):
        check_class_shares(make_info({0: 0.8, 1: 0.2}, {0: 0.7, 1: 0.3}))

    assert "Churn class 0 share differs by 0.1000" in caplog.text
    assert "Churn class 1 share differs by 0.1000" in caplog.text


def test_check_class_shares_treats_absent_class_as_zero_share(caplog):
    with caplog.at_level(logging.WARNING):
        check_class_shares(make_info({0: 0.8, 1: 0.2}, {0: 1.0}))

    assert "Churn class 1 share differs by 0.2000" in caplog.text


def test_check_class_shares_respects_tolerance(caplog):
    with caplog.at_level(logging.WARNING):
        check_class_shares(make_info({0: 0.8, 1: 0.2}, {0: 0.7, 1: 0.3}), tolerance=0.2)

    assert not caplog.records
