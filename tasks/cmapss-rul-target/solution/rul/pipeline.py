"""The public interface: fit on a run-to-failure frame, predict RUL.

The evaluation calls `fit(frame)` and then `model.predict_last(frame)`. Keep those
two signatures; change anything behind them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rul.data import Frame, last_cycle_rows, rul_labels
from rul.features import build_features, build_targets
from rul.features import RUL_CAP
from rul.model import MLP, Scaler


@dataclass
class RULModel:
    scaler: Scaler
    regressor: MLP

    def predict(self, frame: Frame) -> np.ndarray:
        """Predicted RUL for every row."""
        x = self.scaler.transform(build_features(frame))
        return np.clip(self.regressor.predict(x), 0.0, RUL_CAP)

    def predict_last(self, frame: Frame) -> np.ndarray:
        """Predicted RUL at each unit's last observed cycle, ordered by unit id."""
        return self.predict(frame)[last_cycle_rows(frame)]


def fit(frame: Frame) -> RULModel:
    x = build_features(frame)
    y = build_targets(rul_labels(frame))
    scaler = Scaler.fit(x)
    return RULModel(scaler, MLP.fit(scaler.transform(x), y))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))
