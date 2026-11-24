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
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std > 0, std, 1.0))

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


@dataclass
class MLP:
    """One hidden ReLU layer, trained with Adam on mini-batches. Seeded: same data,
    same model."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        hidden: int = 32,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 0,
    ) -> MLP:
        rng = np.random.default_rng(seed)
        n, d = x.shape
        params = [
            rng.normal(0, np.sqrt(2 / d), (d, hidden)),
            np.zeros(hidden),
            rng.normal(0, np.sqrt(1 / hidden), (hidden, 1)),
            np.array([y.mean()]),
        ]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb, yb = x[idx], y[idx, None]
                w1, b1, w2, b2 = params
                h = xb @ w1 + b1
                a = np.maximum(h, 0)
                g = 2 * (a @ w2 + b2 - yb) / len(idx)  # d(MSE)/d(output)
                gh = (g @ w2.T) * (h > 0)
                grads = [xb.T @ gh, gh.sum(0), a.T @ g, g.sum(0)]
                step += 1
                for i, (p, gp) in enumerate(zip(params, grads, strict=True)):
                    m[i] = 0.9 * m[i] + 0.1 * gp
                    v[i] = 0.999 * v[i] + 0.001 * gp**2
                    m_hat = m[i] / (1 - 0.9**step)
                    v_hat = v[i] / (1 - 0.999**step)
                    p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return cls(*params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.maximum(x @ self.w1 + self.b1, 0) @ self.w2 + self.b2).ravel()
