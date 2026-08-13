# ClaimGuard data

Run `python -m preprocessing.acquire_data` from the repository root to fetch
the pinned freMTPL2 OpenML snapshots. The command creates:

- `raw/freMTPL2freq.parquet` — one row per policy (OpenML data ID 41214)
- `raw/freMTPL2sev.parquet` — one row per claim (OpenML data ID 41215)
- `raw/manifest.json` — row counts, schemas, file sizes, and SHA-256 checksums

Downloaded files, generated data, and the OpenML cache are intentionally not
tracked by Git. `raw/` must remain an unmodified source layer; cleaning and
reconciliation outputs belong in `processed/`.

## Data audit

Generate the validation report and row-level issue flags with:

```bash
python -m preprocessing.audit_data
```

Outputs are written to `audit/`. The report records rule counts and explicit
handling decisions. Flags preserve ambiguous observations for investigation,
while exclusions apply only to downstream analytical uses; raw data are never
changed.
