# ClaimGuard data

Run `python -m preprocessing.acquire_data` from the repository root to fetch
the pinned freMTPL2 OpenML snapshots. The command creates:

- `raw/freMTPL2freq.parquet` — one row per policy (OpenML data ID 41214)
- `raw/freMTPL2sev.parquet` — one row per claim (OpenML data ID 41215)
- `raw/manifest.json` — row counts, schemas, file sizes, and SHA-256 checksums

Downloaded files, generated data, and the OpenML cache are intentionally not
tracked by Git. `raw/` must remain an unmodified source layer; cleaning and
reconciliation outputs belong in `processed/`.

