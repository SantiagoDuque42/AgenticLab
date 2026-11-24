"""Turning raw rows into model inputs."""

from __future__ import annotations

import numpy as np

from rul.data import Frame

# All settings and sensors, plus the cycle count. (Until 2026-11-20 this was a
# hand-picked list of 14 sensors; see CHANGES.md.)
FEATURE_COLUMNS = ["cycle", "op1", "op2", "op3", *[f"s{i}" for i in range(1, 22)]]

# Piecewise-linear target: early in life an engine looks healthy whatever its true
# RUL, so the target is capped.
RUL_CAP = 125.0


def build_features(frame: Frame) -> np.ndarray:
    return np.column_stack([frame[c] for c in FEATURE_COLUMNS])


def build_targets(rul: np.ndarray) -> np.ndarray:
    return np.minimum(rul, RUL_CAP).astype(float)
