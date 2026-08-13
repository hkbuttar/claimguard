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
