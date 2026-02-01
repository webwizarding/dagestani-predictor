from datetime import datetime

import pandas as pd

from src.data.features import build_feature_rows


def test_no_leakage_prior_fights_only():
    fights = pd.DataFrame(
        [
            {
                "bout_id": "b1",
                "fight_date": datetime(2020, 1, 1),
                "fighter_a_id": "fa",
                "fighter_b_id": "fb",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "round": 1,
                "time": "5:00",
                "label": 1,
            },
            {
                "bout_id": "b2",
                "fight_date": datetime(2020, 6, 1),
                "fighter_a_id": "fa",
                "fighter_b_id": "fb",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "round": 1,
                "time": "5:00",
                "label": 0,
            },
        ]
    )
    fighters = pd.DataFrame(
        [
            {
                "fighter_id": "fa",
                "name": "Fighter A",
                "height": "5' 11\"",
                "reach": "72\"",
                "stance": "Orthodox",
                "dob": "Jan 1, 1990",
            },
            {
                "fighter_id": "fb",
                "name": "Fighter B",
                "height": "5' 10\"",
                "reach": "70\"",
                "stance": "Southpaw",
                "dob": "Jan 1, 1991",
            },
        ]
    )
    totals = pd.DataFrame(
        [
            {
                "bout_id": "b1",
                "fighter_id": "fa",
                "fighter": "Fighter A",
                "sig_str_landed": 10,
                "sig_str_attempted": 20,
                "total_str_landed": 30,
                "total_str_attempted": 50,
                "takedowns_landed": 1,
                "takedowns_attempted": 2,
                "sub_att": 0,
                "reversals": 0,
                "control": "1:00",
            },
            {
                "bout_id": "b1",
                "fighter_id": "fb",
                "fighter": "Fighter B",
                "sig_str_landed": 5,
                "sig_str_attempted": 10,
                "total_str_landed": 20,
                "total_str_attempted": 40,
                "takedowns_landed": 0,
                "takedowns_attempted": 1,
                "sub_att": 0,
                "reversals": 0,
                "control": "0:30",
            },
        ]
    )

    features, _ = build_feature_rows(fights, fighters, totals)
    first = features.iloc[0]
    second = features.iloc[1]

    assert first["total_fights_a"] == 0
    assert second["total_fights_a"] == 1
    assert second["total_fights_b"] == 1
