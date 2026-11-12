import numpy as np
import pandas as pd
import pytest

from ledgerwatch.anomaly import (
    MODEL_FEATURES,
    anomaly_scores,
    load_bundle,
    make_bundle,
    save_bundle,
    train_isolation_forest,
)


@pytest.fixture
def X() -> pd.DataFrame:
    """500 normal points around 0 plus 10 obvious outliers around 8."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(500, len(MODEL_FEATURES)))
    outliers = rng.normal(8, 1, size=(10, len(MODEL_FEATURES)))
    return pd.DataFrame(np.vstack([normal, outliers]), columns=MODEL_FEATURES)


@pytest.fixture
def model(X):
    return train_isolation_forest(X, n_estimators=50)


def test_outliers_score_higher_than_normal_points(model, X):
    scores = anomaly_scores(model, X)

    assert scores[500:].min() > np.median(scores[:500])


def test_time_column_is_not_a_model_feature():
    assert "Time" not in MODEL_FEATURES


def test_save_and_load_roundtrip(model, X, tmp_path):
    path = tmp_path / "model.joblib"
    save_bundle(make_bundle(model, threshold=0.6), path)

    bundle = load_bundle(path)

    assert bundle["threshold"] == 0.6
    np.testing.assert_allclose(anomaly_scores(bundle["model"], X), anomaly_scores(model, X))


def test_load_refuses_tampered_model(model, tmp_path):
    path = tmp_path / "model.joblib"
    save_bundle(make_bundle(model, threshold=0.6), path)

    with path.open("ab") as f:
        f.write(b"tampered")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_bundle(path)


def test_load_refuses_model_without_checksum(model, tmp_path):
    path = tmp_path / "model.joblib"
    save_bundle(make_bundle(model, threshold=0.6), path)
    path.with_suffix(".joblib.sha256").unlink()

    with pytest.raises(ValueError, match="Checksum file"):
        load_bundle(path)


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.joblib")
