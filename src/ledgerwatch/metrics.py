"""Evaluation helpers shared by every model (Isolation Forest, supervised baseline).

All models produce a score where higher means "more likely fraud". A threshold
turns scores into flag / don't flag decisions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, scores) -> float:
    """Return the score threshold that maximizes F1.

    Call this on a validation set, never on the test set: picking the threshold
    on test data would make the test metrics optimistic.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold, so drop it.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[np.argmax(f1)])


def evaluate(y_true, scores, threshold: float) -> dict:
    """Threshold-dependent and threshold-free metrics for one model."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        # PR-AUC of a model that guesses randomly equals the fraud rate.
        "pr_auc_baseline": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "alerts": int(tp + fp),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
