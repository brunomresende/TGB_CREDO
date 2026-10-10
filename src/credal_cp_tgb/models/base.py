"""Contratos minimos usados pelos metodos conformais e credais."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class QuantileModel(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike): ...

    def predict_quantiles(self, X: ArrayLike) -> NDArray[np.float64]: ...


@runtime_checkable
class PosteriorQuantileModel(QuantileModel, Protocol):
    def sample_quantiles(
        self, X: ArrayLike, *, n_samples: int = 1000, random_state: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def validate_interval_predictions(predictions: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("A predicao deve ter formato (n_observacoes, 2).")
    if np.any(values[:, 0] > values[:, 1]):
        raise ValueError("O modelo retornou quantis cruzados.")
    return values

