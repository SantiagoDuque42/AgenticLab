"""Validation that mimics the real test set.

Test engines are cut off at a random point before failure, and the score is the
RMSE of the RUL predicted at that last observed cycle. So: hold out whole engines
(never rows: rows of one engine are strongly correlated), cut each held-out engine
at a random cycle, and score the prediction at the cut. As in the FD001 test set,
every cut leaves at least 31 observed cycles and at most MAX_TRUE_RUL to go.

Run it: python -m rul.evaluate
"""

from __future__ import annotations

import numpy as np

from rul.data import Frame, load, rul_labels
from rul.pipeline import fit, rmse


def split_units(frame: Frame, val_fraction: float = 0.2, seed: int = 0):
    """Hold out a random `val_fraction` of the rows for validation.

    Vectorized on 2026-11-20: one mask instead of a per-unit loop.
    """
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(frame["unit"])) < val_fraction
    return {k: v[~is_val] for k, v in frame.items()}, {k: v[is_val] for k, v in frame.items()}


MIN_OBSERVED = 31
MAX_TRUE_RUL = 150


def truncate_randomly(frame: Frame, seed: int = 0) -> tuple[Frame, np.ndarray]:
    """Cut each unit at a random cycle; return the cut frame and the true RUL at each cut."""
    rng = np.random.default_rng(seed)
    rul = rul_labels(frame)
    keep = np.zeros(len(rul), dtype=bool)
    true = []
    for u in np.unique(frame["unit"]):
        rows = np.flatnonzero(frame["unit"] == u)
        earliest = max(min(MIN_OBSERVED, len(rows)), len(rows) - MAX_TRUE_RUL)
        cut = rng.integers(earliest, len(rows) + 1)
        keep[rows[:cut]] = True
        true.append(rul[rows[cut - 1]])
    return {k: v[keep] for k, v in frame.items()}, np.array(true)


def validation_rmse(frame: Frame, seeds: range = range(5)) -> float:
    scores = []
    for seed in seeds:
        train, val = split_units(frame, seed=seed)
        model = fit(train)
        cut, true = truncate_randomly(val, seed=seed)
        scores.append(rmse(model.predict_last(cut), true))
    return float(np.mean(scores))


if __name__ == "__main__":
    print(f"validation RMSE: {validation_rmse(load()):.2f}")
