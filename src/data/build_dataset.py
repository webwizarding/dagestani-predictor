"""Build supervised dataset with strict as-of features."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.config import DatasetConfig, PROCESSED_DIR
from src.data.features import add_diff_features, build_feature_rows, _fighter_id_from_url, _parse_date

logger = logging.getLogger(__name__)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_dataset(
    events_path: Path,
    fights_path: Path,
    fighters_path: Path,
    fight_totals_path: Path,
    output_dir: Path = PROCESSED_DIR,
    config: DatasetConfig | None = None,
) -> dict[str, Path]:
    config = config or DatasetConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = _read_table(events_path)
    fights = _read_table(fights_path)
    fighters = _read_table(fighters_path)
    totals = _read_table(fight_totals_path)

    events["event_date"] = events["date"].map(_parse_date)
    event_dates = events.set_index("event_id")["event_date"].to_dict()

    fights["fight_date"] = fights["event_id"].map(event_dates)
    fights = fights.dropna(subset=["fight_date"]).copy()
    fights["fighter_a_id"] = fights["fighter_a_url"].map(_fighter_id_from_url)
    fights["fighter_b_id"] = fights["fighter_b_url"].map(_fighter_id_from_url)

    # Label: 1 if fighter A wins, 0 if loses; drop draws/no-contests.
    results = fights["result"].str.upper().fillna("")
    keep_mask = results.isin(["W", "L"])
    fights = fights[keep_mask].copy()
    fights["label"] = (results == "W").astype(int)

    features_df, metadata_df = build_feature_rows(
        fights[[
            "bout_id",
            "fight_date",
            "fighter_a_id",
            "fighter_b_id",
            "fighter_a",
            "fighter_b",
            "round",
            "time",
            "label",
        ]].copy(),
        fighters,
        totals,
        ewma_alpha=config.ewma_alpha,
    )

    features_df = _encode_stance(features_df)
    diff_df = add_diff_features(features_df)
    dataset = diff_df.merge(metadata_df, on=["bout_id", "fight_date"], how="left")

    labeled_path = output_dir / "fights_labeled.csv"
    fights[[
        "bout_id",
        "fight_date",
        "fighter_a_id",
        "fighter_b_id",
        "fighter_a",
        "fighter_b",
        "round",
        "time",
        "label",
    ]].to_csv(labeled_path, index=False)

    dataset_path = _write_table(dataset, output_dir / "fights")
    meta_path = output_dir / "fights_metadata.csv"
    metadata_df.to_csv(meta_path, index=False)

    dataset_hash = _hash_df(dataset)
    hash_path = output_dir / "fights.hash.txt"
    hash_path.write_text(dataset_hash, encoding="utf-8")

    return {
        "dataset": dataset_path,
        "metadata": meta_path,
        "hash": hash_path,
        "fights_labeled": labeled_path,
    }


def _encode_stance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for side in ["a", "b"]:
        col = f"stance_{side}"
        if col not in df.columns:
            continue
        norm = df[col].fillna("").str.lower()
        df[f"stance_orthodox_{side}"] = (norm == "orthodox").astype(int)
        df[f"stance_southpaw_{side}"] = (norm == "southpaw").astype(int)
        df[f"stance_switch_{side}"] = (norm == "switch").astype(int)
        df[f"stance_open_{side}"] = (norm == "open stance").astype(int)
        df[f"stance_other_{side}"] = (~norm.isin(["orthodox", "southpaw", "switch", "open stance"])).astype(
            int
        )
    return df


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


def _hash_df(df: pd.DataFrame) -> str:
    data = df.sort_values(["fight_date", "bout_id"]).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
