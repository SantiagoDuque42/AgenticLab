"""Turning raw rows into model inputs."""

from __future__ import annotations

import numpy as np

from rul.data import Frame

FEATURE_COLUMNS = ["op1", "op2", "op3", *[f"s{i}" for i in range(1, 22)]]


def build_features(frame: Frame) -> np.ndarray:
    """One feature row per data row: every setting and sensor, as is."""
    return np.column_stack([frame[c] for c in FEATURE_COLUMNS])


def build_targets(rul: np.ndarray) -> np.ndarray:
    return rul.astype(float)
