"""Isolation Forest anomaly model: training, scoring, and safe saving/loading."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from ledgerwatch.data import PCA_FEATURES, PROJECT_ROOT, RANDOM_STATE, file_sha256

# `Time` is left out on purpose: it counts seconds since the dataset began, so in
# production it would only ever grow and never match what the model saw in training.
MODEL_FEATURES = [*PCA_FEATURES, "Amount"]
MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest.joblib"


def train_isolation_forest(
    X: pd.DataFrame,
    n_estimators: int = 200,
    max_samples: int | str = 256,
    random_state: int = RANDOM_STATE,
) -> IsolationForest:
    """Fit an Isolation Forest without using any labels.

    No scaling is needed: each split picks a random value between a feature's
    min and max, so the result does not depend on the feature's units.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination="auto",  # we pick our own threshold, see metrics.best_f1_threshold
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X[MODEL_FEATURES])
    return model


def anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous. (scikit-learn's score_samples is the opposite.)"""
    return -model.score_samples(X[MODEL_FEATURES])


def make_bundle(model: IsolationForest, threshold: float) -> dict:
    """Everything needed to score a transaction later, saved together."""
    return {
        "model": model,
        "threshold": threshold,
        "features": MODEL_FEATURES,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _hash_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def save_bundle(bundle: dict, path: Path = MODEL_PATH) -> str:
    """Save the bundle and a SHA-256 checksum file next to it; return the hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    digest = file_sha256(path)
    _hash_path(path).write_text(digest)
    return digest


def load_bundle(path: Path = MODEL_PATH) -> dict:
    """Load a saved bundle after checking its checksum.

    joblib files are pickles, and unpickling can run arbitrary code. The checksum
    makes sure we only load the exact file our own training script produced.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python scripts/train_isolation_forest.py"
        )
    hash_file = _hash_path(path)
    if not hash_file.exists():
        raise ValueError(f"Checksum file {hash_file} missing; refusing to load model.")
    if file_sha256(path) != hash_file.read_text().strip():
        raise ValueError(f"Checksum mismatch for {path}; refusing to load model.")

    bundle = joblib.load(path)  # noqa: S301 - integrity verified above
    if bundle["sklearn_version"] != sklearn.__version__:
        raise ValueError(
            f"Model trained with scikit-learn {bundle['sklearn_version']}, "
            f"but {sklearn.__version__} is installed. Retrain the model."
        )
    return bundle
