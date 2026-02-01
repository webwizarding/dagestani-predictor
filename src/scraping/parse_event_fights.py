"""Parse fights listed on a UFCStats event page."""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _bout_id_from_url(url: str) -> str:
    return urlparse(url).path.strip("/").split("/")[-1]


def parse_event_fights(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="b-fight-details__table")
    if not table:
        logger.warning("fight details table not found")
        return []

    rows = table.find_all("tr", class_="b-fight-details__table-row")
    fights: list[dict] = []
    for row in rows:
        link = row.find("a", class_="b-link", href=True)
        if not link:
            continue
        cols = [c.get_text(" ", strip=True) for c in row.find_all("td")]
        if len(cols) < 8:
            continue
        fighter_links = row.find_all("a", class_="b-link", href=True)
        fighter_a = ""
        fighter_b = ""
        fighter_a_url = ""
        fighter_b_url = ""
        if len(fighter_links) >= 3:
            fighter_a_url = fighter_links[1]["href"].strip()
            fighter_b_url = fighter_links[2]["href"].strip()
            fighter_a = fighter_links[1].get_text(strip=True)
            fighter_b = fighter_links[2].get_text(strip=True)
        fighters = row.find_all("p", class_="b-fight-details__table-text")
        if not fighter_a and len(fighters) > 0:
            fighter_a = fighters[0].get_text(strip=True)
        if not fighter_b and len(fighters) > 1:
            fighter_b = fighters[1].get_text(strip=True)
        result = cols[0]
        weight_class = cols[6]
        method = cols[7]
        round_ = cols[8] if len(cols) > 8 else ""
        time = cols[9] if len(cols) > 9 else ""
        url = link["href"].strip()
        fights.append(
            {
                "bout_id": _bout_id_from_url(url),
                "url": url,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "fighter_a_url": fighter_a_url,
                "fighter_b_url": fighter_b_url,
                "result": result,
                "method": method,
                "round": round_,
                "time": time,
                "weight_class": weight_class,
            }
        )
    return fights

