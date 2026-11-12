"""Train the Isolation Forest, tune its threshold, evaluate on test, and save it.

Usage:
    python scripts/train_isolation_forest.py

Data flow (the test set is used exactly once, at the end):
    train  -> fit (75%)  : train the model (labels ignored)
           -> val (25%)  : choose the threshold that maximizes F1
    test                 : final, untouched evaluation
"""

import json

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

from ledgerwatch.anomaly import (
    anomaly_scores,
    make_bundle,
    save_bundle,
    train_isolation_forest,
)
from ledgerwatch.data import PROJECT_ROOT, TARGET, load_splits, split_data
from ledgerwatch.metrics import best_f1_threshold, evaluate

REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    train, test = load_splits()
    fit, val = split_data(train, test_size=0.25)
    print(f"fit: {len(fit):,} rows | val: {len(val):,} rows | test: {len(test):,} rows")

    model = train_isolation_forest(fit)

    threshold = best_f1_threshold(val[TARGET], anomaly_scores(model, val))
    test_scores = anomaly_scores(model, test)
    metrics = evaluate(test[TARGET], test_scores, threshold)

    digest = save_bundle(make_bundle(model, threshold))

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "isolation_forest_metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(test[TARGET], test_scores, ax=ax,
                                            name="Isolation Forest")
    ax.axhline(metrics["pr_auc_baseline"], linestyle="--", color="gray", label="random")
    ax.legend()
    ax.set_title("Isolation Forest: precision-recall on test set")
    fig.savefig(REPORTS_DIR / "isolation_forest_pr_curve.png", dpi=120, bbox_inches="tight")

    print("\nTest set metrics")
    for key in ["precision", "recall", "f1", "pr_auc", "pr_auc_baseline", "roc_auc"]:
        print(f"  {key:<16} {metrics[key]:.4f}")
    print(f"  alerts           {metrics['alerts']} "
          f"(tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']})")
    print(f"\nSaved models/isolation_forest.joblib (sha256 {digest[:12]}...)")
    print("Saved reports/isolation_forest_metrics.json and isolation_forest_pr_curve.png")


if __name__ == "__main__":
    main()
