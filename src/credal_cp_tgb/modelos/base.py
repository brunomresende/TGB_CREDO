"""Contratos mínimos usados pelos métodos conformais e credais."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class ModeloQuantilico(Protocol):
    def ajustar(self, X: ArrayLike, y: ArrayLike): ...

    def prever_quantis(self, X: ArrayLike) -> NDArray[np.float64]: ...


@runtime_checkable
class ModeloQuantilicoPosterior(ModeloQuantilico, Protocol):
    def amostrar_quantis(
        self,
        X: ArrayLike,
        *,
        numero_amostras: int = 1000,
        semente_aleatoria: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def validar_predicoes_intervalares(predicoes: ArrayLike) -> NDArray[np.float64]:
    valores = np.asarray(predicoes, dtype=float)
    if valores.ndim != 2 or valores.shape[1] != 2:
        raise ValueError("A predição deve ter formato (n_observacoes, 2).")
    if np.any(valores[:, 0] > valores[:, 1]):
        raise ValueError("O modelo retornou quantis cruzados.")
    return valores
