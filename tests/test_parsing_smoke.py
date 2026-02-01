from pathlib import Path

from src.scraping.parse_events import parse_events
from src.scraping.parse_event_fights import parse_event_fights
from src.scraping.parse_fighters import parse_fighter
from src.scraping.parse_fight_details import parse_fight_details


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_events_smoke():
    html = (FIXTURES / "events.html").read_text(encoding="utf-8")
    events = parse_events(html)
    assert events
    assert events[0]["event_id"] == "evt123"


def test_parse_event_fights_smoke():
    html = (FIXTURES / "event_fights.html").read_text(encoding="utf-8")
    fights = parse_event_fights(html)
    assert fights
    fight = fights[0]
    assert fight["bout_id"] == "bout123"
    assert fight["fighter_a_url"].endswith("fighterA")


def test_parse_fighter_smoke():
    html = (FIXTURES / "fighter.html").read_text(encoding="utf-8")
    fighter = parse_fighter(html, "http://ufcstats.com/fighter-details/fighterA")
    assert fighter["fighter_id"] == "fighterA"
    assert fighter["name"] == "Fighter A"


def test_parse_fight_details_smoke():
    html = (FIXTURES / "fight_detail.html").read_text(encoding="utf-8")
    details = parse_fight_details(html, "http://ufcstats.com/fight-details/bout123")
    assert details["bout_id"] == "bout123"
    assert details["totals"]
    assert details["rounds"]
