"""Parse UFCStats events list pages."""
from __future__ import annotations

import logging
from typing import Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _event_id_from_url(url: str) -> str:
    return urlparse(url).path.strip("/").split("/")[-1]


def parse_events(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="b-statistics__table-events")
    if not table:
        logger.warning("events table not found")
        return []

    rows = table.find_all("tr", class_="b-statistics__table-row")
    events: list[dict] = []
    for row in rows:
        link = row.find("a", href=True)
        if not link:
            continue
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) < 2:
            continue
        name = cols[0]
        location = cols[1]
        date = cols[2] if len(cols) > 2 else ""
        url = link["href"].strip()
        events.append(
            {
                "event_id": _event_id_from_url(url),
                "name": name,
                "date": date,
                "location": location,
                "url": url,
            }
        )
    return events


