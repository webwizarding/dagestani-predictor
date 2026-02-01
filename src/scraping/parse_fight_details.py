"""Parse fight detail page for per-fighter and per-round stats."""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _bout_id_from_url(url: str) -> str:
    return urlparse(url).path.strip("/").split("/")[-1]


def _parse_stat_pair(text: str) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    parts = text.replace(" ", "").split("of")
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def parse_fight_details(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    bout_id = _bout_id_from_url(url)

    summary = soup.find("div", class_="b-fight-details")
    fighter_links = soup.find_all("a", class_="b-link", href=True)
    fighters_block = []
    for link in fighter_links:
        if "/fighter-details/" in link["href"]:
            fighters_block.append(
                {
                    "fighter": link.get_text(strip=True),
                    "fighter_id": urlparse(link["href"]).path.strip("/").split("/")[-1],
                    "fighter_url": link["href"].strip(),
                }
            )
    fighter_names = [f["fighter"] for f in fighters_block] if fighters_block else [
        p.get_text(strip=True)
        for p in soup.find_all("h3", class_="b-fight-details__person-name")
    ]

    # Totals table
    totals_table = soup.find("table", class_="b-fight-details__table")
    totals_rows = totals_table.find_all("tr", class_="b-fight-details__table-row") if totals_table else []

    totals: list[dict] = []
    for row in totals_rows:
        cols = [c.get_text(" ", strip=True) for c in row.find_all("td")]
        if len(cols) < 9:
            continue
        fighter = cols[0]
        fighter_id = ""
        if fighters_block:
            match = next((f for f in fighters_block if f["fighter"] == fighter), None)
            if match:
                fighter_id = match["fighter_id"]
        kd = cols[1]
        sig_str = cols[2]
        total_str = cols[3]
        td = cols[5] if len(cols) > 5 else ""
        sub_att = cols[6] if len(cols) > 6 else ""
        rev = cols[7] if len(cols) > 7 else ""
        ctrl = cols[8] if len(cols) > 8 else ""
        sig_l, sig_a = _parse_stat_pair(sig_str)
        tot_l, tot_a = _parse_stat_pair(total_str)
        td_l, td_a = _parse_stat_pair(td)
        totals.append(
            {
                "bout_id": bout_id,
                "fighter": fighter,
                "fighter_id": fighter_id,
                "kd": kd,
                "sig_str_landed": sig_l,
                "sig_str_attempted": sig_a,
                "total_str_landed": tot_l,
                "total_str_attempted": tot_a,
                "takedowns_landed": td_l,
                "takedowns_attempted": td_a,
                "sub_att": sub_att,
                "reversals": rev,
                "control": ctrl,
            }
        )

    rounds: list[dict] = []
    round_tables = soup.find_all("table", class_="b-fight-details__table")
    for table in round_tables[1:]:
        header = table.find_previous("h2")
        round_label = header.get_text(strip=True) if header else ""
        rows = table.find_all("tr", class_="b-fight-details__table-row")
        for row in rows:
            cols = [c.get_text(" ", strip=True) for c in row.find_all("td")]
            if len(cols) < 6:
                continue
            fighter = cols[0]
            fighter_id = ""
            if fighters_block:
                match = next((f for f in fighters_block if f["fighter"] == fighter), None)
                if match:
                    fighter_id = match["fighter_id"]
            sig_str = cols[2]
            total_str = cols[3]
            td = cols[5] if len(cols) > 5 else ""
            sig_l, sig_a = _parse_stat_pair(sig_str)
            tot_l, tot_a = _parse_stat_pair(total_str)
            td_l, td_a = _parse_stat_pair(td)
            rounds.append(
                {
                    "bout_id": bout_id,
                    "round": round_label,
                "fighter": fighter,
                "fighter_id": fighter_id,
                    "sig_str_landed": sig_l,
                    "sig_str_attempted": sig_a,
                    "total_str_landed": tot_l,
                    "total_str_attempted": tot_a,
                    "takedowns_landed": td_l,
                    "takedowns_attempted": td_a,
                }
            )

    return {
        "bout_id": bout_id,
        "fighters": fighter_names,
        "totals": totals,
        "rounds": rounds,
    }
