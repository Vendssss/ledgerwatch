import numpy as np
import pytest

from ledgerwatch.metrics import best_f1_threshold, evaluate


def test_best_f1_threshold_separates_perfectly_split_scores():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    threshold = best_f1_threshold(y_true, scores)

    assert 0.4 < threshold <= 0.8


def test_evaluate_counts_confusion_matrix():
    y_true = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.9, 0.3])

    m = evaluate(y_true, scores, threshold=0.5)

    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["alerts"] == 2
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["pr_auc_baseline"] == pytest.approx(0.4)


def test_evaluate_no_alerts_gives_zero_precision_not_error():
    y_true = np.array([0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3])

    m = evaluate(y_true, scores, threshold=1.0)

    assert m["alerts"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
