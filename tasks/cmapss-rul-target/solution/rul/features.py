"""Turning raw rows into model inputs (reference solution).

Changes from the starting version:
- Only the 14 sensors that move in FD001. The settings and the other 7 sensors are
  constant or pure noise here (one operating condition), so they add nothing.
- Each sensor also gets a trailing 30-cycle mean per engine: one reading is noisy,
  the recent trend is the signal. Only past rows are used, so the feature is
  available at prediction time.
- The training target is capped at 125 cycles (piecewise-linear RUL): early in life
  a healthy engine looks the same whatever its true RUL, so asking the model to
  tell RUL 300 from RUL 200 only adds error.
"""

from __future__ import annotations

import numpy as np

from rul.data import Frame

SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
WINDOW = 30
RUL_CAP = 125.0


def trailing_mean(values: np.ndarray, unit: np.ndarray, window: int) -> np.ndarray:
    """Mean of each row and up to `window - 1` earlier rows of the same unit."""
    out = np.empty_like(values, dtype=float)
    for u in np.unique(unit):
        rows = np.flatnonzero(unit == u)
        csum = np.concatenate([[0.0], np.cumsum(values[rows])])
        end = np.arange(1, len(rows) + 1)
        start = np.maximum(0, end - window)
        out[rows] = (csum[end] - csum[start]) / (end - start)
    return out


def build_features(frame: Frame) -> np.ndarray:
    raw = [frame[s] for s in SENSORS]
    smooth = [trailing_mean(frame[s], frame["unit"], WINDOW) for s in SENSORS]
    return np.column_stack(raw + smooth)


def build_targets(rul: np.ndarray) -> np.ndarray:
    return np.minimum(rul, RUL_CAP).astype(float)
