"""Read-only access to precomputed portfolio and model reports."""

import json
from pathlib import Path

import pandas as pd


class ArtifactRepository:
    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir

    def json_report(self, directory: str, filename: str = "metrics.json") -> dict:
        path = self.reports_dir / directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Report artifact is unavailable: {path}")
        return json.loads(path.read_text())

    def csv_records(self, directory: str, filename: str) -> list[dict]:
        path = self.reports_dir / directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Report artifact is unavailable: {path}")
        frame = pd.read_csv(path)
        return frame.where(frame.notna(), None).to_dict(orient="records")
