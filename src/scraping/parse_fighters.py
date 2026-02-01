"""Parse fighter profile pages."""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _fighter_id_from_url(url: str) -> str:
    return urlparse(url).path.strip("/").split("/")[-1]


def parse_fighter(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    name = soup.find("span", class_="b-content__title-highlight")
    name_text = name.get_text(strip=True) if name else ""

    def find_label(label: str) -> str:
        tag = soup.find("li", class_="b-list__box-list-item", string=lambda t: t and label in t)
        if not tag:
            return ""
        return tag.get_text(strip=True).split(":")[-1].strip()

    height = find_label("Height")
    reach = find_label("Reach")
    stance = find_label("STANCE") or find_label("Stance")
    dob = find_label("DOB")
    record = soup.find("span", class_="b-content__title-record")
    record_text = record.get_text(strip=True) if record else ""

    return {
        "fighter_id": _fighter_id_from_url(url),
        "name": name_text,
        "height": height,
        "reach": reach,
        "stance": stance,
        "dob": dob,
        "record": record_text,
        "url": url,
    }


