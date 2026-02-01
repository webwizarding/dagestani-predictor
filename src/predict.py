"""Prediction helpers for CLI/API."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from dateutil import parser as dateparser

from src.config import PROCESSED_DIR
from src.data.features import add_diff_features, compute_pre_fight_features


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find_fighter_id(fighters: pd.DataFrame, name: str) -> str:
    name_norm = name.strip().lower()
    exact = fighters[fighters["name"].str.lower() == name_norm]
    if len(exact) == 1:
        return exact.iloc[0]["fighter_id"]
    if len(exact) > 1:
        raise ValueError(f"multiple fighters found for: {name}")

    contains = fighters[fighters["name"].str.lower().str.contains(name_norm, na=False)]
    if len(contains) == 1:
        return contains.iloc[0]["fighter_id"]
    if len(contains) > 1:
        options = ", ".join(sorted(contains["name"].head(5).tolist()))
        raise ValueError(f"multiple fighters match '{name}': {options}")

    options = ", ".join(sorted(fighters["name"].dropna().head(5).tolist()))
    raise ValueError(f"fighter not found: {name}. Sample available: {options}")


def predict_matchup(
    fighter_a: str,
    fighter_b: str,
    date_str: str,
    model_path: Path,
    data_dir: Path = PROCESSED_DIR,
) -> dict:
    fights_path = data_dir / "fights_labeled.csv"
    fighters_path = _pick_path(data_dir / "fighters")
    totals_path = _pick_path(data_dir / "fight_totals")

    fights = _read_table(fights_path)
    fighters = _read_table(fighters_path)
    totals = _read_table(totals_path)

    fight_date = dateparser.parse(date_str).replace(tzinfo=None)
    fighter_a_id = _find_fighter_id(fighters, fighter_a)
    fighter_b_id = _find_fighter_id(fighters, fighter_b)

    feature_row = compute_pre_fight_features(
        fights,
        fighters,
        totals,
        fighter_a_id,
        fighter_b_id,
        fight_date,
    )

    feature_df = pd.DataFrame([feature_row])
    feature_df = _encode_stance(feature_df)
    diff_df = add_diff_features(feature_df)

    model = joblib.load(model_path)
    proba = float(model.predict_proba(diff_df)[:, 1][0])
    contributions = _top_contributions(model, diff_df)

    result = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "date": fight_date.date().isoformat(),
        "prob_fighter_a_wins": proba,
    }
    if contributions:
        result["top_features"] = contributions
    return result


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


def _pick_path(stem: Path) -> Path:
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"missing {stem}.parquet or {stem}.csv")


def _top_contributions(model, X: pd.DataFrame) -> list[dict]:
    try:
        clf = model.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return []
        pre = model.named_steps["pre"]
        Xt = pre.transform(X)
        coefs = clf.coef_[0]
        names = pre.get_feature_names_out()
        contrib = Xt.toarray()[0] * coefs if hasattr(Xt, "toarray") else Xt[0] * coefs
        pairs = sorted(zip(names, contrib), key=lambda x: abs(x[1]), reverse=True)
        return [{"feature": str(f), "contribution": float(v)} for f, v in pairs[:10]]
    except Exception:
        return []


def main() -> None:
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Predict a UFC matchup")
    parser.add_argument("--fighter-a", required=True)
    parser.add_argument("--fighter-b", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--model", default="artifacts/model_logistic.joblib")
    args = parser.parse_args()

    result = predict_matchup(args.fighter_a, args.fighter_b, args.date, Path(args.model))
    print(result)


if __name__ == "__main__":
    main()
