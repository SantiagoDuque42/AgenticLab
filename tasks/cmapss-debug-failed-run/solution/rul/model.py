"""Model: standardized features, then ridge regression (closed form, numpy only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        mean, std = x.mean(axis=0), x.std(axis=0)
        # A column that never varies in training (FD001 has several constant sensors)
        # has std 0: dividing by it gives 0/0 = NaN, which poisons the whole fit.
        # "Constant" needs a tolerance: float rounding can leave a std of 1e-11 on a
        # constant column, and dividing by that turns rounding noise into a feature
        # (and a later drift of 1.0 into 1e11). Scale such columns by 1 instead: they
        # stay ~0, ridge gives them ~0 weight, and a drifting reading can't blow up.
        constant = std <= 1e-8 * np.maximum(np.abs(mean), 1.0)
        return cls(mean, np.where(constant, 1.0, std))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass
class Ridge:
    weights: np.ndarray
    bias: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
        n_features = x.shape[1]
        y_mean = y.mean()
        weights = np.linalg.solve(x.T @ x + alpha * np.eye(n_features), x.T @ (y - y_mean))
        return cls(weights, float(y_mean))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias
