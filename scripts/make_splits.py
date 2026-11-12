"""Create the train/test splits used by every model in this project.

Usage:
    python scripts/make_splits.py
"""

from ledgerwatch.data import TARGET, drop_duplicates, load_raw, save_splits, split_data


def main() -> None:
    df = load_raw()
    print(f"Loaded {len(df):,} rows, fraud rate {df[TARGET].mean():.4%}")

    deduped = drop_duplicates(df)
    print(f"Dropped {len(df) - len(deduped):,} duplicate rows")

    train, test = split_data(deduped)
    meta = save_splits(train, test)

    print(f"Train: {meta['train_rows']:,} rows, {meta['train_fraud']} fraud "
          f"({train[TARGET].mean():.4%})")
    print(f"Test:  {meta['test_rows']:,} rows, {meta['test_fraud']} fraud "
          f"({test[TARGET].mean():.4%})")
    print("Saved to data/processed/")


if __name__ == "__main__":
    main()
