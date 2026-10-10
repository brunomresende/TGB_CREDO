"""Estrategias fixa e adaptativa para o nivel credal gamma."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class FixedGamma:
    gamma: float = 0.1

    def fit(self, X: ArrayLike) -> "FixedGamma":
        if not 0 < self.gamma < 1:
            raise ValueError("gamma deve estar entre 0 e 1.")
        self.is_fitted_ = True
        return self

    def compute(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("Execute fit antes de compute.")
        return np.full(np.asarray(X).shape[0], self.gamma, dtype=float)


@dataclass
class AdaptiveGamma:
    gamma_min: float = 0.1
    gamma_max: float = 0.75
    tau: float = 1.0
    epsilon: float = 1e-6
    neighbor_constant: float = 6.672
    n_neighbors: int | None = None

    def fit(self, X: ArrayLike) -> "AdaptiveGamma":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional.")
        if not 0 < self.gamma_min < self.gamma_max < 1:
            raise ValueError("Use 0 < gamma_min < gamma_max < 1.")
        if self.tau <= 0:
            raise ValueError("tau deve ser positivo.")

        n, dimension = X_arr.shape
        if self.n_neighbors is None:
            k = int(np.ceil(self.neighbor_constant * n ** (4 / (4 + dimension))))
        else:
            k = int(self.n_neighbors)
        self.n_neighbors_ = max(2, min(k, n))

        self.scaler_ = StandardScaler().fit(X_arr)
        scaled = self.scaler_.transform(X_arr)
        self.neighbors_ = NearestNeighbors(n_neighbors=self.n_neighbors_).fit(scaled)
        distances, _ = self.neighbors_.kneighbors(scaled)
        kth_distances = distances[:, -1]
        self.distance_median_ = float(np.quantile(kth_distances, 0.50))
        self.distance_high_ = float(np.quantile(kth_distances, 0.95))
        return self

    @staticmethod
    def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(values, -500, 500)
        return 1 / (1 + np.exp(-clipped))

    def compute(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "neighbors_"):
            raise RuntimeError("Execute fit antes de compute.")
        scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        distances, _ = self.neighbors_.kneighbors(scaled)
        kth = np.log1p(distances[:, -1])
        low = np.log1p(self.distance_median_)
        high = np.log1p(self.distance_high_)
        scarcity = (kth - low) / (high - low + self.epsilon)
        gamma = self.gamma_max - (
            (self.gamma_max - self.gamma_min) * self._sigmoid(scarcity / self.tau)
        )
        return np.clip(gamma, self.gamma_min, self.gamma_max)

