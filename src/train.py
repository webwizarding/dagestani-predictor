"""Module wrapper for training (python -m src.train)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.models.train import train_model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train UFC outcome models")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["logistic", "hgb"], default="logistic")
    args = parser.parse_args()

    outputs = train_model(Path(args.data), args.model)
    logging.info("Training complete: %s", outputs)


if __name__ == "__main__":
    main()

