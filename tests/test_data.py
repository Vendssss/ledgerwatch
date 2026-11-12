import numpy as np
import pandas as pd
import pytest

from ledgerwatch.data import (
    EXPECTED_COLUMNS,
    FEATURES,
    TARGET,
    drop_duplicates,
    load_raw,
    split_data,
)


@pytest.fixture
def transactions() -> pd.DataFrame:
    """1,000 synthetic transactions with the real schema, 20 of them fraud (2%)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(1000, len(FEATURES))), columns=FEATURES)
    df[TARGET] = 0
    df.loc[:19, TARGET] = 1
    return df[EXPECTED_COLUMNS]


def test_split_preserves_fraud_rate(transactions):
    train, test = split_data(transactions)

    assert train[TARGET].mean() == pytest.approx(test[TARGET].mean(), abs=0.005)
    assert test[TARGET].sum() > 0


def test_split_sizes_add_up(transactions):
    train, test = split_data(transactions)

    assert len(train) + len(test) == len(transactions)
    assert len(test) == pytest.approx(0.2 * len(transactions), abs=1)


def test_split_has_no_overlap(transactions):
    train, test = split_data(transactions)

    assert train.merge(test, how="inner").empty


def test_split_is_reproducible(transactions):
    train_a, _ = split_data(transactions)
    train_b, _ = split_data(transactions)

    pd.testing.assert_frame_equal(train_a, train_b)


def test_drop_duplicates_removes_exact_copies(transactions):
    with_dupes = pd.concat([transactions, transactions.head(5)])

    deduped = drop_duplicates(with_dupes)

    assert len(deduped) == len(transactions)
    assert not deduped.duplicated().any()


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download creditcard.csv"):
        load_raw(tmp_path / "creditcard.csv")


def test_load_raw_missing_columns_raises(tmp_path):
    path = tmp_path / "creditcard.csv"
    pd.DataFrame({"Time": [0], "Amount": [1.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_raw(path)
