# ClaimGuard — Auto Insurance Claims Risk, Severity & Tail-Loss Intelligence

Auto insurance risk platform benchmarking actuarial GLMs against machine learning for claim frequency, severity, pure premium, and extreme-loss risk. Built on real French motor TPL claims with calibration, EVT tail modeling, risk segmentation, and portfolio-level validation.

## Setup and data acquisition

ClaimGuard uses Python 3.11+ and the pinned freMTPL2 OpenML snapshots. Create an
isolated environment, install the dependencies, and download the raw data:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m preprocessing.acquire_data
```

The acquisition command validates each table's schema and writes Parquet files
plus a checksum manifest under `data/raw/`. It is idempotent: existing files are
validated and reused. Pass `--force` to download fresh copies.

Verify the setup with:

```bash
pytest -q
```

## Data audit and reconciliation

Profile data quality and reconcile policy claim counts against observed claim
records with:

```bash
python -m preprocessing.audit_data
```

This produces JSON and Markdown reports plus row-level flags under `data/audit/`.
Every rule has an explicit valid, correctable, ambiguous, or excluded handling
decision. Raw source files remain unchanged.

## Actuarial modeling tables

Construct policy-frequency, claim-severity, and policy-loss datasets with:

```bash
python -m preprocessing.build_tables
```

The generated tables retain audit lineage, join policy characteristics onto
individual claims, and reconcile individual claim amounts to policy total loss.
