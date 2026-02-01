"""Simple ELO implementation for fight outcomes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EloConfig:
    k_factor: float = 24.0
    initial: float = 1500.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, result_a: float, k: float) -> tuple[float, float]:
    exp_a = expected_score(rating_a, rating_b)
    exp_b = expected_score(rating_b, rating_a)
    new_a = rating_a + k * (result_a - exp_a)
    new_b = rating_b + k * ((1.0 - result_a) - exp_b)
    return new_a, new_b


