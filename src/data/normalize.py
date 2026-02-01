"""Normalize parsed JSON data into tabular files."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import PARSED_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize(parsed_dir: Path = PARSED_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)

    events = _read_jsonl(parsed_dir / "events.jsonl")
    fights = _read_jsonl(parsed_dir / "fights.jsonl")
    fighters = _read_jsonl(parsed_dir / "fighters.jsonl")
    fight_details = _read_jsonl(parsed_dir / "fight_details.jsonl")

    outputs: dict[str, Path] = {}
    if events:
        df = pd.DataFrame(events)
        outputs["events"] = _write_table(df, processed_dir / "events")
    if fights:
        df = pd.DataFrame(fights)
        outputs["fights"] = _write_table(df, processed_dir / "fights")
    if fighters:
        df = pd.DataFrame(fighters)
        outputs["fighters"] = _write_table(df, processed_dir / "fighters")
    if fight_details:
        totals_rows = []
        rounds_rows = []
        for row in fight_details:
            totals_rows.extend(row.get("totals", []))
            rounds_rows.extend(row.get("rounds", []))
        if totals_rows:
            outputs["fight_totals"] = _write_table(
                pd.DataFrame(totals_rows), processed_dir / "fight_totals"
            )
        if rounds_rows:
            outputs["fight_rounds"] = _write_table(
                pd.DataFrame(rounds_rows), processed_dir / "fight_rounds"
            )

    return outputs


def _write_table(df: pd.DataFrame, stem: Path) -> Path:
    try:
        import pyarrow  # noqa: F401

        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = stem.with_suffix(".csv")
        df.to_csv(path, index=False)
        return path


