"""Loading C-MAPSS files.

Each row is one engine at one cycle: unit id, cycle number, 3 operational settings
and 21 sensor readings. Training engines run until failure; test engines stop at
some point before failure, and the dataset gives their true RUL separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLUMNS = ["unit", "cycle", "op1", "op2", "op3", *[f"s{i}" for i in range(1, 22)]]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_PATH = DATA_DIR / "train_FD001.txt.gz"

Frame = dict[str, np.ndarray]


def load(path: Path = TRAIN_PATH) -> Frame:
    """Load a C-MAPSS file into a dict of columns, sorted by unit then cycle."""
    raw = np.loadtxt(path)
    raw = raw[np.lexsort((raw[:, 1], raw[:, 0]))]
    return {name: raw[:, i] for i, name in enumerate(COLUMNS)}


def rul_labels(frame: Frame) -> np.ndarray:
    """Cycles left before failure for each row of a run-to-failure frame."""
    unit = frame["unit"]
    last = {u: frame["cycle"][unit == u].max() for u in np.unique(unit)}
    return np.array([last[u] for u in unit]) - frame["cycle"]


def select_units(frame: Frame, units: np.ndarray) -> Frame:
    mask = np.isin(frame["unit"], units)
    return {k: v[mask] for k, v in frame.items()}


def last_cycle_rows(frame: Frame) -> np.ndarray:
    """Index of each unit's last row (frames are sorted by unit, then cycle)."""
    unit = frame["unit"]
    return np.flatnonzero(np.append(unit[1:] != unit[:-1], True))
