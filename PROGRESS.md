# LedgerWatch progress

Resume point for working across machines. Update at the end of every phase.

**Last updated:** 2026-11-13
**Current phase:** Phases 2 and 3 code written and tested on synthetic data.
**Blocked on:** `creditcard.csv` not downloaded yet, so no real metrics exist yet.
Next: Phase 4 (supervised baseline).

## Done

### Phase 1: Repository setup
- `src/` layout package (`src/ledgerwatch/`) with `pyproject.toml`
  (setuptools, pytest config with `pythonpath = ["src"]`, Ruff lint rules incl. `S` security checks)
- Pinned dependencies, verified to install on Python 3.14.7 (Windows):
  - `requirements.txt`: FastAPI, Uvicorn, Pydantic, pydantic-settings, scikit-learn,
    pandas, NumPy, joblib, httpx
  - `requirements-dev.txt`: pytest, Ruff, ipykernel, matplotlib
- Folder skeleton: `data/raw`, `data/processed`, `models`, `notebooks`, `scripts`, `tests`
  (kept in git with `.gitkeep`)
- `.gitignore` extended: datasets (`data/raw/*`, `*.csv`, `*.parquet`), all model files,
  Jupyter checkpoints
- `.gitattributes`: LF line endings so diffs match across machines
- README stub with dataset instructions, setup steps, and roadmap

### Phase 2: Data exploration and split (code done, real-data run pending)
- `src/ledgerwatch/data.py`: `load_raw` (with column check), `drop_duplicates`,
  `split_data` (stratified, 80/20, `random_state=42`), `save_splits`, `load_splits`
- `scripts/make_splits.py`: one command that writes `data/processed/train.csv`, `test.csv`
  and `split_meta.json` (row/fraud counts + SHA-256 of the raw file)
- `notebooks/01_data_exploration.ipynb`: class balance, Amount, hour of day, top PCA
  features, duplicates, split checks (stratification preserved, zero train/test overlap)
- Leakage decisions: exact duplicates dropped *before* splitting; no scaling in this phase
  (scalers will be fit on train only inside a scikit-learn `Pipeline`)
- Smoke-tested on a synthetic dataset with the same schema; not yet run on the real data
- `tests/test_data.py`: 7 unit tests on synthetic data (stratification, split sizes,
  no overlap, reproducibility, duplicate removal, missing file, missing columns), all passing

### Phase 3: Isolation Forest (code done, real-data run pending)
- `src/ledgerwatch/metrics.py` (shared with Phase 4): `best_f1_threshold` (tuned on
  validation only) and `evaluate` (precision, recall, F1, PR-AUC + random baseline,
  ROC-AUC, confusion counts, alert count)
- `src/ledgerwatch/anomaly.py`: `train_isolation_forest` (200 trees, max_samples=256,
  labels never used), `anomaly_scores` (higher = more anomalous), model bundle with
  threshold + feature list + sklearn version + timestamp
- Features: `V1`-`V28` + `Amount`. `Time` excluded (seconds since dataset start, would not
  generalize to production). No scaling (Isolation Forest does not depend on feature units).
- Data flow: train -> fit 75% / val 25% (stratified); threshold picked on val; test used once
- Model integrity: `save_bundle` writes `models/isolation_forest.joblib` + `.sha256`;
  `load_bundle` refuses a missing/mismatched checksum or a different scikit-learn version
  (joblib = pickle = arbitrary code execution if the file is swapped)
- `scripts/train_isolation_forest.py` writes `reports/isolation_forest_metrics.json` and
  `reports/isolation_forest_pr_curve.png` (`reports/` is tracked in git)
- Tests: `tests/test_metrics.py` (3), `tests/test_anomaly.py` (6). Full suite: 16 passing
- End-to-end run verified on synthetic data only

## To finish Phases 2 and 3 (needs the real dataset)
- [ ] Download `creditcard.csv` into `data/raw/`
- [ ] `python scripts/make_splits.py` and record the numbers below
- [ ] Run the notebook and write the "Findings" cell in your own words
- [ ] `python scripts/train_isolation_forest.py` and record the metrics below
- [x] Tests written and passing (`pytest -v`)

Real-data split (fill in): rows = ?, duplicates dropped = ?, train = ? (? fraud),
test = ? (? fraud)

Isolation Forest test metrics (fill in): precision = ?, recall = ?, F1 = ?, PR-AUC = ?

## Next: Phase 4 (supervised baseline)
- Logistic regression (scaled inside a `Pipeline`, `class_weight="balanced"`) and/or
  random forest on the same splits, same `best_f1_threshold` on val, same `evaluate` on test
- README section comparing supervised vs unsupervised detection

## Open issues / notes
- Split is random-stratified as planned. A time-based split (train on earlier `Time`, test on
  later) would better mimic production; worth mentioning in the README as a limitation.
- Dataset is not in git. Download it again on each machine.
- Model files are not in git. Run `scripts/make_splits.py` then
  `scripts/train_isolation_forest.py` on each machine.
- Possible Phase 3 improvement: small search over `max_samples` / `max_features`,
  scored by PR-AUC on val.
- Python 3.14 is new; if a package fails to install on another machine, use Python 3.12 or 3.13.
- Incident Triage link in README points to the GitHub profile; replace with the repo URL.
- License not chosen yet.

## Setting up on another machine
```powershell
git pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
# then place creditcard.csv in data\raw\
```
