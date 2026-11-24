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
        return cls(x.mean(axis=0), x.std(axis=0))

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
