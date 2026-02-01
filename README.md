# Dagestani Predictor

Educational project to scrape public UFCStats data, build a pre-fight dataset, and train ML models to predict fight outcomes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional API dependencies:

```bash
pip install -e .[api]
```

## Quickstart (small sample)

```bash
python -m src.cli quickstart
python -m src.cli normalize
python -m src.cli build
python -m src.cli train --model logistic
python -m src.cli predict --fighter-a "Fighter A" --fighter-b "Fighter B" --date "2020-01-02"
```

## Full scrape -> dataset

```bash
python -m src.cli scrape --max-events 50
python -m src.cli normalize
python -m src.cli build
```

Notes:
- Scraping respects robots.txt and rate limits by default.
- Use `--no-robots` only for local experiments.

## Train

```bash
python -m src.cli train --model logistic
python -m src.cli train --model hgb
```

Artifacts are saved to `artifacts/`.

## Predict (CLI)

```bash
python -m src.cli predict --fighter-a "Max Holloway" --fighter-b "Justin Gaethje" --date "2024-04-13"
```

Predictions use only data strictly before the provided date.

## API (optional)

```bash
uvicorn src.api.app:app --reload
```

POST `/predict` with:

```json
{"fighter_a":"Fighter A","fighter_b":"Fighter B","date":"2020-01-02"}
```

## Data leakage prevention

- Feature engineering uses strict chronological iteration.
- For each fight, only prior fights are used to compute aggregates and ELO.
- The current fight's stats are never used as features.
- A `tests/test_no_leakage.py` check enforces the as-of behavior.

## Limitations

- Public data can be incomplete or missing (height/reach/DOB).
- Models are probabilistic and do not guarantee accuracy.
- This project is for educational purposes and not betting advice.

## Ethics & legal

This project uses public UFCStats pages, rate limits requests, caches responses, and respects site policies. Predictions are purely analytical and are not advice.

