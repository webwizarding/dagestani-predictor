"""Command line interface for scraping, dataset builds, training, and prediction."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from src.config import PARSED_DIR, PROCESSED_DIR, ScrapeConfig
from src.data.build_dataset import build_dataset
from src.data.normalize import normalize
from src.models.train import train_model
from src.predict import predict_matchup
from src.scraping.fetch import CachedSession, save_jsonl
from src.scraping.parse_event_fights import parse_event_fights
from src.scraping.parse_events import parse_events
from src.scraping.parse_fight_details import parse_fight_details
from src.scraping.parse_fighters import parse_fighter

logger = logging.getLogger(__name__)

EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="UFC Stats scraper and ML pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    scrape_p = sub.add_parser("scrape", help="Scrape UFCStats into data/parsed")
    scrape_p.add_argument("--max-events", type=int, default=None)
    scrape_p.add_argument("--max-fights", type=int, default=None)
    scrape_p.add_argument("--no-robots", action="store_true")
    scrape_p.add_argument("--no-cache", action="store_true")

    sub.add_parser("normalize", help="Normalize parsed JSONL to processed tables")

    build_p = sub.add_parser("build", help="Build ML dataset with as-of features")
    build_p.add_argument("--events", default=None)
    build_p.add_argument("--fights", default=None)
    build_p.add_argument("--fighters", default=None)
    build_p.add_argument("--totals", default=None)

    train_p = sub.add_parser("train", help="Train model")
    train_p.add_argument("--data", default=None)
    train_p.add_argument("--model", choices=["logistic", "hgb"], default="logistic")

    predict_p = sub.add_parser("predict", help="Predict a matchup")
    predict_p.add_argument("--fighter-a", required=True)
    predict_p.add_argument("--fighter-b", required=True)
    predict_p.add_argument("--date", required=True)
    predict_p.add_argument("--model", default="artifacts/model_logistic.joblib")

    sub.add_parser("quickstart", help="Copy sample parsed data into data/parsed")

    args = parser.parse_args()

    if args.command == "scrape":
        scrape(args)
    elif args.command == "normalize":
        normalize()
    elif args.command == "build":
        build(args)
    elif args.command == "train":
        train(args)
    elif args.command == "predict":
        result = predict_matchup(args.fighter_a, args.fighter_b, args.date, Path(args.model))
        print(json.dumps(result, indent=2))
    elif args.command == "quickstart":
        quickstart()


def scrape(args: argparse.Namespace) -> None:
    cfg = ScrapeConfig(
        max_events=args.max_events,
        max_fights_per_event=args.max_fights,
        respect_robots=not args.no_robots,
        cache_enabled=not args.no_cache,
    )
    session = CachedSession(cfg)

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    seen = _load_seen(PARSED_DIR / "seen.json")

    events_html = session.get(EVENTS_URL)
    events = parse_events(events_html)
    if cfg.max_events:
        events = events[: cfg.max_events]

    new_events = [e for e in events if e["event_id"] not in seen["events"]]
    _append_jsonl(PARSED_DIR / "events.jsonl", new_events)
    seen["events"].update(e["event_id"] for e in new_events)

    new_fights = []
    new_fighters = []
    new_details = []

    for event in events:
        if event["event_id"] in seen["event_pages"]:
            continue
        event_html = session.get(event["url"])
        fights = parse_event_fights(event_html)
        for fight in fights[: cfg.max_fights_per_event or len(fights)]:
            fight["event_id"] = event["event_id"]
            if fight["bout_id"] not in seen["fights"]:
                new_fights.append(fight)
                seen["fights"].add(fight["bout_id"])

            if fight.get("fighter_a_url") and fight.get("fighter_a_url") not in seen["fighters"]:
                fighter_html = session.get(fight["fighter_a_url"])
                new_fighters.append(parse_fighter(fighter_html, fight["fighter_a_url"]))
                seen["fighters"].add(fight["fighter_a_url"])
            if fight.get("fighter_b_url") and fight.get("fighter_b_url") not in seen["fighters"]:
                fighter_html = session.get(fight["fighter_b_url"])
                new_fighters.append(parse_fighter(fighter_html, fight["fighter_b_url"]))
                seen["fighters"].add(fight["fighter_b_url"])

            if fight["bout_id"] not in seen["fight_details"]:
                detail_html = session.get(fight["url"])
                new_details.append(parse_fight_details(detail_html, fight["url"]))
                seen["fight_details"].add(fight["bout_id"])

        seen["event_pages"].add(event["event_id"])

    _append_jsonl(PARSED_DIR / "fights.jsonl", new_fights)
    _append_jsonl(PARSED_DIR / "fighters.jsonl", new_fighters)
    _append_jsonl(PARSED_DIR / "fight_details.jsonl", new_details)
    _save_seen(PARSED_DIR / "seen.json", seen)

    logger.info("Scrape complete: %s events, %s fights", len(new_events), len(new_fights))


def build(args: argparse.Namespace) -> None:
    events_path = Path(args.events) if args.events else _pick_processed("events.parquet")
    fights_path = Path(args.fights) if args.fights else _pick_processed("fights.parquet")
    fighters_path = Path(args.fighters) if args.fighters else _pick_processed("fighters.parquet")
    totals_path = Path(args.totals) if args.totals else _pick_processed("fight_totals.parquet")
    outputs = build_dataset(
        events_path=events_path,
        fights_path=fights_path,
        fighters_path=fighters_path,
        fight_totals_path=totals_path,
    )
    logger.info("Dataset built: %s", outputs)


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data) if args.data else _pick_processed("fights.parquet")
    outputs = train_model(data_path, args.model)
    logger.info("Training complete: %s", outputs)


def quickstart() -> None:
    sample_dir = Path("data/parsed/sample")
    if not sample_dir.exists():
        raise FileNotFoundError("sample data not found")
    for file in sample_dir.glob("*.jsonl"):
        shutil.copy(file, PARSED_DIR / file.name)
    logger.info("Sample parsed data copied to data/parsed")


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_seen(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {"events": set(), "event_pages": set(), "fights": set(), "fighters": set(), "fight_details": set()}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: set(v) for k, v in data.items()}


def _save_seen(path: Path, seen: dict[str, set[str]]) -> None:
    data = {k: sorted(v) for k, v in seen.items()}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _pick_processed(name: str) -> Path:
    parquet = PROCESSED_DIR / name
    if parquet.exists():
        return parquet
    csv = PROCESSED_DIR / name.replace(".parquet", ".csv")
    return csv


if __name__ == "__main__":
    main()
