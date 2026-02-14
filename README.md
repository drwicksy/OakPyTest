# Transaction Anomaly Spotter (PoC)

## Overview
A small web app that flags unusual transactions in a CSV and explains why each transaction was flagged.

I chose this build because it’s a compact risk based system, the types of which I already work on day-to-day.

The app is intentionally scoped as a triage tool: it highlights statistically unusual activity and provides reasons/metrics to help a human reviewer, keeping a human in the loop.

## What it does
- Upload a CSV of transactions (or generate a sample dataset for testing).
- Computes a few anomaly signals per account.
- Produces:
  - an anomaly score (0–100)
  - a severity bucket (Low/Medium/High)
  - a list of reasons per flagged transaction
  - quick summary metrics and light charts
  - downloadable CSV outputs

## Future enhancements
If the build were to progress the obvious next step would be to automate ingestion of data and output to a more complex UI with.

## Data format
### Required columns
Your CSV must include:
- `timestamp` (ISO-8601 recommended)
- `account_id` (string or number)
- `amount` (number, positive/negative allowed)

### Optional columns (improves analysis)
- `transaction_id` (duplicate detection)
- `counterparty` (new counterparty detection)
- `currency`, `direction`, `channel`, `country`

## How to run
### Setup
python -m pip install -r requirements.txt
### Run
python -m streamlit run app.py
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Open the URL printed in the terminal (usually http://localhost:8501).
