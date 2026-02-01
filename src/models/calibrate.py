"""Calibration utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve


@dataclass
class CalibrationData:
    prob_true: np.ndarray
    prob_pred: np.ndarray


def compute_calibration(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> CalibrationData:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return CalibrationData(prob_true=prob_true, prob_pred=prob_pred)


