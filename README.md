# LedgerWatch

Fraud risk scoring service for financial transactions, built with Python and FastAPI.

LedgerWatch scores each transaction with two independent layers:

1. **Rules engine**: explainable checks such as unusually large amounts, rapid repeat
   transactions, and activity at odd hours.
2. **Isolation Forest**: an unsupervised anomaly detection model (scikit-learn) that flags
   transactions that look statistically unusual.

Transactions flagged by either layer are sent as alerts to
[Incident Triage](https://github.com/Vendssss) (Java / Spring Boot) through a webhook.

> **Status:** early development. See [PROGRESS.md](PROGRESS.md) for the current phase.

## Tech stack

| Area      | Tools                                   |
|-----------|-----------------------------------------|
| API       | FastAPI, Pydantic, Uvicorn              |
| ML        | scikit-learn, pandas, NumPy, joblib     |
| Alerting  | httpx (webhook client)                  |
| Testing   | pytest, Ruff                            |
| Packaging | Docker, docker-compose (planned)        |

## Dataset

This project uses the Kaggle
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset
(284,807 transactions, about 0.17% fraud). It is **not** included in this repository.

1. Download `creditcard.csv` from Kaggle.
2. Place it at `data/raw/creditcard.csv`.
3. Create the train/test split used by every model:
   ```powershell
   python scripts/make_splits.py
   ```
   Exact duplicate rows are removed first, then the data is split 80/20, stratified by class.
   Exploration is in [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb).
4. Train the Isolation Forest and evaluate it on the test set:
   ```powershell
   python scripts/train_isolation_forest.py
   ```
   This saves `models/isolation_forest.joblib` (with a SHA-256 checksum checked on load)
   and writes metrics and a precision-recall curve to `reports/`.

Because the classes are extremely imbalanced, accuracy is not a useful metric. Models are
evaluated with precision, recall, F1, and PR-AUC.

## Project structure

```
ledgerwatch/
├── src/ledgerwatch/    # application package (rules, model, API, webhook client)
├── tests/              # pytest suite
├── notebooks/          # data exploration
├── scripts/            # training and evaluation scripts
├── data/raw/           # creditcard.csv goes here (git-ignored)
├── data/processed/     # train/test splits (git-ignored)
├── models/             # trained model files (git-ignored)
├── requirements.txt    # runtime dependencies
└── requirements-dev.txt
```

## Getting started (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
```

On macOS / Linux, activate with `source .venv/bin/activate` instead.

## Roadmap

- [x] Repository setup
- [x] Data exploration and stratified train/test split
- [x] Isolation Forest model with threshold tuning (real-data metrics pending)
- [ ] Supervised baseline and comparison
- [ ] Rules engine
- [ ] FastAPI scoring endpoint
- [ ] Incident Triage webhook client
- [ ] Test suite
- [ ] Docker and docker-compose
- [ ] Architecture diagram and results

## License

To be decided.
