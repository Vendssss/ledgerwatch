"""Loading and splitting the Kaggle Credit Card Fraud Detection dataset.

Every later phase (models, evaluation) reads the splits written here, so the
train/test boundary is decided once, in one place, and never re-drawn.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", *PCA_FEATURES, "Amount"]
EXPECTED_COLUMNS = [*FEATURES, TARGET]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read creditcard.csv and check it has the columns we expect."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it there."
        )
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df[EXPECTED_COLUMNS]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows.

    If the same row lands in both train and test, the model is tested on data
    it has already seen, which inflates metrics (a form of leakage).
    """
    return df.drop_duplicates().reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split so both sides keep the same fraud rate."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET],
        random_state=random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def file_sha256(path: Path) -> str:
    """Fingerprint of the raw file, so a split can be traced to the exact data used."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_path: Path = RAW_PATH,
    out_dir: Path = PROCESSED_DIR,
) -> dict:
    """Write train.csv, test.csv and split_meta.json; return the metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    meta = {
        "raw_sha256": file_sha256(raw_path),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_fraud": int(train[TARGET].sum()),
        "test_fraud": int(test[TARGET].sum()),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_splits(out_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the saved splits. Later phases use this instead of re-splitting."""
    train_path, test_path = out_dir / "train.csv", out_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Splits not found in {out_dir}. Run: python scripts/make_splits.py"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)
