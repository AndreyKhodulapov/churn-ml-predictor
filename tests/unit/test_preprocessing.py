import logging

import pandas as pd
import pytest

from src.core.constants import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from src.schemas.dataset import DatasetRowChurn
from src.services.preprocessing import build_imputer, prepare_data

FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


@pytest.fixture
def make_df(customer_features):
    def make(*labels: float | None, **overrides: list) -> pd.DataFrame:
        df = pd.DataFrame([{**customer_features, TARGET_COLUMN: label} for label in labels])
        for column, values in overrides.items():
            df[column] = values
        return df

    return make


def test_feature_lists_match_dataset_schema():
    assert set(FEATURES) == DatasetRowChurn.model_fields.keys() - {TARGET_COLUMN}
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert set(BINARY_FEATURES) <= set(NUMERIC_FEATURES)


def test_prepare_data_separates_features_and_target(make_df):
    prepared = prepare_data(make_df(0, 1, 1))

    assert list(prepared.X.columns) == FEATURES
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

    assert list(filled.columns) == FEATURES
    # The train median; the mean would be 40.0.
    assert filled.loc[3, "monthly_fee"] == 20.0
    assert filled.loc[3, "region"] == "asia"


def test_imputer_keeps_binary_feature_binary(make_df):
    X = prepare_data(make_df(0, 1, 0, 1, autopay_enabled=[1, 1, 0, None])).X
    imputer = build_imputer(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(X.iloc[:3])

    filled = imputer.transform(X.iloc[3:])

    # The mean would be 0.67; the most frequent value keeps the flag a 0/1 number.
    assert filled.loc[3, "autopay_enabled"] == 1
