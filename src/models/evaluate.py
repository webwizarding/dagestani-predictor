"""Model evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


@dataclass
class Metrics:
    accuracy: float
    roc_auc: float
    log_loss: float
    brier: float


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_proba >= 0.5)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        log_loss=float(log_loss(y_true, y_proba, labels=[0, 1])),
        brier=float(brier_score_loss(y_true, y_proba)),
    )


