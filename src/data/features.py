"""Feature engineering with strict as-of logic to avoid leakage."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from dateutil import parser as dateparser

from src.data.elo import EloConfig, update_elo

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return dateparser.parse(value).replace(tzinfo=None)
    except Exception:
        return None


def _parse_height(value: str) -> float | None:
    if not value:
        return None
    if "'" in value:
        try:
            feet, inches = value.replace('"', "").split("'")
            return float(feet.strip()) * 12.0 + float(inches.strip())
        except Exception:
            return None
    digits = "".join(ch for ch in value if ch.isdigit())
    return float(digits) if digits else None


def _parse_reach(value: str) -> float | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    return float(digits) if digits else None


def _fighter_id_from_url(url: str) -> str:
    if not url:
        return ""
    return urlparse(url).path.strip("/").split("/")[-1]


def _duration_minutes(round_str: str, time_str: str) -> float:
    try:
        round_num = int(round_str)
    except Exception:
        round_num = 1
    minutes = (round_num - 1) * 5
    try:
        t_parts = time_str.split(":")
        minutes += int(t_parts[0]) + int(t_parts[1]) / 60.0
    except Exception:
        minutes += 5.0
    return minutes


@dataclass
class FighterState:
    total_fights: int = 0
    ufc_fights: int = 0
    wins: int = 0
    losses: int = 0
    win_streak: int = 0
    loss_streak: int = 0
    total_rounds: int = 0
    total_minutes: float = 0.0
    sig_landed: float = 0.0
    sig_attempted: float = 0.0
    sig_absorbed: float = 0.0
    td_landed: float = 0.0
    td_attempted: float = 0.0
    control_minutes: float = 0.0
    sub_att: float = 0.0
    last_fight_date: datetime | None = None
    elo: float = 1500.0
    ewma_sig_landed_per_min: float | None = None
    ewma_sig_attempted_per_min: float | None = None
    ewma_td_landed_per_15: float | None = None
    ewma_control_per_min: float | None = None


def _update_ewma(prev: float | None, value: float, alpha: float) -> float:
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev


def build_feature_rows(
    fights: pd.DataFrame,
    fighters: pd.DataFrame,
    fight_totals: pd.DataFrame,
    ewma_alpha: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return features dataframe and metadata dataframe (pre-fight info)."""
    fighters = fighters.copy()
    fighters["height_in"] = fighters["height"].map(_parse_height)
    fighters["reach_in"] = fighters["reach"].map(_parse_reach)
    fighters["dob_dt"] = fighters["dob"].map(_parse_date)
    fighters = fighters.set_index("fighter_id")

    totals = fight_totals.copy()
    if "fighter_id" not in totals.columns:
        totals["fighter_id"] = ""
    totals_keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in totals.iterrows():
        key = (row.get("bout_id", ""), row.get("fighter_id", "") or row.get("fighter", ""))
        totals_keyed[key] = row.to_dict()

    fights_sorted = fights.sort_values("fight_date").reset_index(drop=True)
    states: dict[str, FighterState] = {}
    features = []
    metadata = []
    elo_cfg = EloConfig()

    for _, fight in fights_sorted.iterrows():
        fid_a = fight["fighter_a_id"]
        fid_b = fight["fighter_b_id"]
        state_a = states.setdefault(fid_a, FighterState(elo=elo_cfg.initial))
        state_b = states.setdefault(fid_b, FighterState(elo=elo_cfg.initial))

        fight_date = fight["fight_date"]
        height_a = fighters.at[fid_a, "height_in"] if fid_a in fighters.index else None
        height_b = fighters.at[fid_b, "height_in"] if fid_b in fighters.index else None
        reach_a = fighters.at[fid_a, "reach_in"] if fid_a in fighters.index else None
        reach_b = fighters.at[fid_b, "reach_in"] if fid_b in fighters.index else None
        stance_a = fighters.at[fid_a, "stance"] if fid_a in fighters.index else None
        stance_b = fighters.at[fid_b, "stance"] if fid_b in fighters.index else None
        dob_a = fighters.at[fid_a, "dob_dt"] if fid_a in fighters.index else None
        dob_b = fighters.at[fid_b, "dob_dt"] if fid_b in fighters.index else None

        age_a = (fight_date - dob_a).days / 365.25 if dob_a else None
        age_b = (fight_date - dob_b).days / 365.25 if dob_b else None
        days_since_a = (fight_date - state_a.last_fight_date).days if state_a.last_fight_date else None
        days_since_b = (fight_date - state_b.last_fight_date).days if state_b.last_fight_date else None

        opp_win_rate_a = state_b.wins / state_b.total_fights if state_b.total_fights else 0.5
        opp_win_rate_b = state_a.wins / state_a.total_fights if state_a.total_fights else 0.5

        features.append(
            {
                "bout_id": fight["bout_id"],
                "fight_date": fight_date,
                "fighter_a_id": fid_a,
                "fighter_b_id": fid_b,
                "age_a": age_a,
                "age_b": age_b,
                "height_a": height_a,
                "height_b": height_b,
                "reach_a": reach_a,
                "reach_b": reach_b,
                "stance_a": stance_a,
                "stance_b": stance_b,
                "days_since_last_a": days_since_a,
                "days_since_last_b": days_since_b,
                "total_fights_a": state_a.total_fights,
                "total_fights_b": state_b.total_fights,
                "ufc_fights_a": state_a.ufc_fights,
                "ufc_fights_b": state_b.ufc_fights,
                "win_streak_a": state_a.win_streak,
                "win_streak_b": state_b.win_streak,
                "loss_streak_a": state_a.loss_streak,
                "loss_streak_b": state_b.loss_streak,
                "total_rounds_a": state_a.total_rounds,
                "total_rounds_b": state_b.total_rounds,
                "sig_landed_per_min_a": _safe_div(state_a.sig_landed, state_a.total_minutes),
                "sig_landed_per_min_b": _safe_div(state_b.sig_landed, state_b.total_minutes),
                "sig_attempted_per_min_a": _safe_div(state_a.sig_attempted, state_a.total_minutes),
                "sig_attempted_per_min_b": _safe_div(state_b.sig_attempted, state_b.total_minutes),
                "sig_acc_a": _safe_div(state_a.sig_landed, state_a.sig_attempted),
                "sig_acc_b": _safe_div(state_b.sig_landed, state_b.sig_attempted),
                "sig_absorbed_per_min_a": _safe_div(state_a.sig_absorbed, state_a.total_minutes),
                "sig_absorbed_per_min_b": _safe_div(state_b.sig_absorbed, state_b.total_minutes),
                "sig_def_a": 1.0 - _safe_div(state_a.sig_absorbed, state_a.sig_attempted),
                "sig_def_b": 1.0 - _safe_div(state_b.sig_absorbed, state_b.sig_attempted),
                "td_landed_per_15_a": _safe_div(state_a.td_landed, state_a.total_minutes) * 15.0,
                "td_landed_per_15_b": _safe_div(state_b.td_landed, state_b.total_minutes) * 15.0,
                "td_acc_a": _safe_div(state_a.td_landed, state_a.td_attempted),
                "td_acc_b": _safe_div(state_b.td_landed, state_b.td_attempted),
                "control_per_min_a": _safe_div(state_a.control_minutes, state_a.total_minutes),
                "control_per_min_b": _safe_div(state_b.control_minutes, state_b.total_minutes),
                "sub_att_per_15_a": _safe_div(state_a.sub_att, state_a.total_minutes) * 15.0,
                "sub_att_per_15_b": _safe_div(state_b.sub_att, state_b.total_minutes) * 15.0,
                "opp_win_rate_a": opp_win_rate_a,
                "opp_win_rate_b": opp_win_rate_b,
                "elo_a": state_a.elo,
                "elo_b": state_b.elo,
                "ewma_sig_landed_per_min_a": state_a.ewma_sig_landed_per_min,
                "ewma_sig_landed_per_min_b": state_b.ewma_sig_landed_per_min,
                "ewma_sig_attempted_per_min_a": state_a.ewma_sig_attempted_per_min,
                "ewma_sig_attempted_per_min_b": state_b.ewma_sig_attempted_per_min,
                "ewma_td_landed_per_15_a": state_a.ewma_td_landed_per_15,
                "ewma_td_landed_per_15_b": state_b.ewma_td_landed_per_15,
                "ewma_control_per_min_a": state_a.ewma_control_per_min,
                "ewma_control_per_min_b": state_b.ewma_control_per_min,
            }
        )

        metadata.append(
            {
                "bout_id": fight["bout_id"],
                "fight_date": fight_date,
                "fighter_a_name": fight["fighter_a"],
                "fighter_b_name": fight["fighter_b"],
                "label": fight["label"],
            }
        )

        # Update state after computing features
        duration_min = _duration_minutes(str(fight["round"]), str(fight["time"]))
        totals_a = totals_keyed.get((fight["bout_id"], fid_a), {})
        totals_b = totals_keyed.get((fight["bout_id"], fid_b), {})

        _update_state(state_a, totals_a, duration_min, ewma_alpha)
        _update_state(state_b, totals_b, duration_min, ewma_alpha)

        result_a = float(fight["label"])
        state_a.elo, state_b.elo = update_elo(state_a.elo, state_b.elo, result_a, elo_cfg.k_factor)
        state_a.total_fights += 1
        state_b.total_fights += 1
        state_a.ufc_fights += 1
        state_b.ufc_fights += 1
        if result_a == 1.0:
            state_a.wins += 1
            state_b.losses += 1
            state_a.win_streak += 1
            state_a.loss_streak = 0
            state_b.loss_streak += 1
            state_b.win_streak = 0
        else:
            state_b.wins += 1
            state_a.losses += 1
            state_b.win_streak += 1
            state_b.loss_streak = 0
            state_a.loss_streak += 1
            state_a.win_streak = 0
        state_a.last_fight_date = fight_date
        state_b.last_fight_date = fight_date

    features_df = pd.DataFrame(features)
    metadata_df = pd.DataFrame(metadata)
    return features_df, metadata_df


def _update_state(state: FighterState, totals: dict[str, Any], duration_min: float, ewma_alpha: float) -> None:
    if duration_min <= 0:
        duration_min = 1.0
    state.total_minutes += duration_min
    state.total_rounds += int(max(1, round(duration_min / 5.0)))
    sig_l = float(totals.get("sig_str_landed") or 0.0)
    sig_a = float(totals.get("sig_str_attempted") or 0.0)
    tot_l = float(totals.get("total_str_landed") or 0.0)
    td_l = float(totals.get("takedowns_landed") or 0.0)
    td_a = float(totals.get("takedowns_attempted") or 0.0)
    sub_att = float(totals.get("sub_att") or 0.0)
    control = _parse_control(totals.get("control"))

    state.sig_landed += sig_l
    state.sig_attempted += sig_a
    state.sig_absorbed += max(0.0, tot_l - sig_l)
    state.td_landed += td_l
    state.td_attempted += td_a
    state.sub_att += sub_att
    state.control_minutes += control

    state.ewma_sig_landed_per_min = _update_ewma(
        state.ewma_sig_landed_per_min, sig_l / duration_min, ewma_alpha
    )
    state.ewma_sig_attempted_per_min = _update_ewma(
        state.ewma_sig_attempted_per_min, sig_a / duration_min, ewma_alpha
    )
    state.ewma_td_landed_per_15 = _update_ewma(
        state.ewma_td_landed_per_15, (td_l / duration_min) * 15.0, ewma_alpha
    )
    state.ewma_control_per_min = _update_ewma(
        state.ewma_control_per_min, control / duration_min, ewma_alpha
    )


def _parse_control(value: Any) -> float:
    if not value:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    if len(parts) != 2:
        return 0.0
    try:
        return int(parts[0]) + int(parts[1]) / 60.0
    except Exception:
        return 0.0


def _safe_div(a: float, b: float) -> float:
    if not b:
        return 0.0
    return float(a) / float(b)


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute A-B difference vector and drop paired columns if desired."""
    diff = pd.DataFrame()
    keep_cols = ["bout_id", "fight_date", "fighter_a_id", "fighter_b_id"]
    for col in df.columns:
        if col.endswith("_a"):
            base = col[:-2]
            col_b = f"{base}_b"
            if col_b in df.columns:
                if not _is_numeric_series(df[col]) or not _is_numeric_series(df[col_b]):
                    continue
                diff[f"{base}_diff"] = df[col] - df[col_b]
    diff["bout_id"] = df["bout_id"]
    diff["fight_date"] = df["fight_date"]
    diff["fighter_a_id"] = df["fighter_a_id"]
    diff["fighter_b_id"] = df["fighter_b_id"]
    return diff


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)



def compute_pre_fight_features(
    fights: pd.DataFrame,
    fighters: pd.DataFrame,
    fight_totals: pd.DataFrame,
    fighter_a_id: str,
    fighter_b_id: str,
    as_of: datetime,
    ewma_alpha: float = 0.35,
) -> dict[str, float | int | None | str]:
    """Compute a single pre-fight feature row as-of a given date."""
    fights = fights.copy()
    fights = fights[fights["fight_date"] < as_of].sort_values("fight_date")
    fighters = fighters.copy()
    fighters["height_in"] = fighters["height"].map(_parse_height)
    fighters["reach_in"] = fighters["reach"].map(_parse_reach)
    fighters["dob_dt"] = fighters["dob"].map(_parse_date)
    fighters = fighters.set_index("fighter_id")

    totals = fight_totals.copy()
    if "fighter_id" not in totals.columns:
        totals["fighter_id"] = ""
    totals_keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in totals.iterrows():
        key = (row.get("bout_id", ""), row.get("fighter_id", "") or row.get("fighter", ""))
        totals_keyed[key] = row.to_dict()

    states: dict[str, FighterState] = {}

    for _, fight in fights.iterrows():
        fid_a = fight["fighter_a_id"]
        fid_b = fight["fighter_b_id"]
        state_a = states.setdefault(fid_a, FighterState())
        state_b = states.setdefault(fid_b, FighterState())

        duration_min = _duration_minutes(str(fight["round"]), str(fight["time"]))
        totals_a = totals_keyed.get((fight["bout_id"], fid_a), {})
        totals_b = totals_keyed.get((fight["bout_id"], fid_b), {})
        _update_state(state_a, totals_a, duration_min, ewma_alpha)
        _update_state(state_b, totals_b, duration_min, ewma_alpha)

        result_a = float(fight["label"])
        state_a.elo, state_b.elo = update_elo(state_a.elo, state_b.elo, result_a, 24.0)
        state_a.total_fights += 1
        state_b.total_fights += 1
        state_a.ufc_fights += 1
        state_b.ufc_fights += 1
        if result_a == 1.0:
            state_a.wins += 1
            state_b.losses += 1
            state_a.win_streak += 1
            state_a.loss_streak = 0
            state_b.loss_streak += 1
            state_b.win_streak = 0
        else:
            state_b.wins += 1
            state_a.losses += 1
            state_b.win_streak += 1
            state_b.loss_streak = 0
            state_a.loss_streak += 1
            state_a.win_streak = 0
        state_a.last_fight_date = fight["fight_date"]
        state_b.last_fight_date = fight["fight_date"]

    state_a = states.setdefault(fighter_a_id, FighterState())
    state_b = states.setdefault(fighter_b_id, FighterState())

    height_a = fighters.at[fighter_a_id, "height_in"] if fighter_a_id in fighters.index else None
    height_b = fighters.at[fighter_b_id, "height_in"] if fighter_b_id in fighters.index else None
    reach_a = fighters.at[fighter_a_id, "reach_in"] if fighter_a_id in fighters.index else None
    reach_b = fighters.at[fighter_b_id, "reach_in"] if fighter_b_id in fighters.index else None
    stance_a = fighters.at[fighter_a_id, "stance"] if fighter_a_id in fighters.index else None
    stance_b = fighters.at[fighter_b_id, "stance"] if fighter_b_id in fighters.index else None
    dob_a = fighters.at[fighter_a_id, "dob_dt"] if fighter_a_id in fighters.index else None
    dob_b = fighters.at[fighter_b_id, "dob_dt"] if fighter_b_id in fighters.index else None

    age_a = (as_of - dob_a).days / 365.25 if dob_a else None
    age_b = (as_of - dob_b).days / 365.25 if dob_b else None
    days_since_a = (as_of - state_a.last_fight_date).days if state_a.last_fight_date else None
    days_since_b = (as_of - state_b.last_fight_date).days if state_b.last_fight_date else None

    return {
        "age_a": age_a,
        "age_b": age_b,
        "height_a": height_a,
        "height_b": height_b,
        "reach_a": reach_a,
        "reach_b": reach_b,
        "stance_a": stance_a,
        "stance_b": stance_b,
        "days_since_last_a": days_since_a,
        "days_since_last_b": days_since_b,
        "total_fights_a": state_a.total_fights,
        "total_fights_b": state_b.total_fights,
        "ufc_fights_a": state_a.ufc_fights,
        "ufc_fights_b": state_b.ufc_fights,
        "win_streak_a": state_a.win_streak,
        "win_streak_b": state_b.win_streak,
        "loss_streak_a": state_a.loss_streak,
        "loss_streak_b": state_b.loss_streak,
        "total_rounds_a": state_a.total_rounds,
        "total_rounds_b": state_b.total_rounds,
        "sig_landed_per_min_a": _safe_div(state_a.sig_landed, state_a.total_minutes),
        "sig_landed_per_min_b": _safe_div(state_b.sig_landed, state_b.total_minutes),
        "sig_attempted_per_min_a": _safe_div(state_a.sig_attempted, state_a.total_minutes),
        "sig_attempted_per_min_b": _safe_div(state_b.sig_attempted, state_b.total_minutes),
        "sig_acc_a": _safe_div(state_a.sig_landed, state_a.sig_attempted),
        "sig_acc_b": _safe_div(state_b.sig_landed, state_b.sig_attempted),
        "sig_absorbed_per_min_a": _safe_div(state_a.sig_absorbed, state_a.total_minutes),
        "sig_absorbed_per_min_b": _safe_div(state_b.sig_absorbed, state_b.total_minutes),
        "sig_def_a": 1.0 - _safe_div(state_a.sig_absorbed, state_a.sig_attempted),
        "sig_def_b": 1.0 - _safe_div(state_b.sig_absorbed, state_b.sig_attempted),
        "td_landed_per_15_a": _safe_div(state_a.td_landed, state_a.total_minutes) * 15.0,
        "td_landed_per_15_b": _safe_div(state_b.td_landed, state_b.total_minutes) * 15.0,
        "td_acc_a": _safe_div(state_a.td_landed, state_a.td_attempted),
        "td_acc_b": _safe_div(state_b.td_landed, state_b.td_attempted),
        "control_per_min_a": _safe_div(state_a.control_minutes, state_a.total_minutes),
        "control_per_min_b": _safe_div(state_b.control_minutes, state_b.total_minutes),
        "sub_att_per_15_a": _safe_div(state_a.sub_att, state_a.total_minutes) * 15.0,
        "sub_att_per_15_b": _safe_div(state_b.sub_att, state_b.total_minutes) * 15.0,
        "opp_win_rate_a": state_b.wins / state_b.total_fights if state_b.total_fights else 0.5,
        "opp_win_rate_b": state_a.wins / state_a.total_fights if state_a.total_fights else 0.5,
        "elo_a": state_a.elo,
        "elo_b": state_b.elo,
        "ewma_sig_landed_per_min_a": state_a.ewma_sig_landed_per_min,
        "ewma_sig_landed_per_min_b": state_b.ewma_sig_landed_per_min,
        "ewma_sig_attempted_per_min_a": state_a.ewma_sig_attempted_per_min,
        "ewma_sig_attempted_per_min_b": state_b.ewma_sig_attempted_per_min,
        "ewma_td_landed_per_15_a": state_a.ewma_td_landed_per_15,
        "ewma_td_landed_per_15_b": state_b.ewma_td_landed_per_15,
        "ewma_control_per_min_a": state_a.ewma_control_per_min,
        "ewma_control_per_min_b": state_b.ewma_control_per_min,
    }
