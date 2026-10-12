"""Divisão reproduzível 56/24/20 usada no artigo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DivisaoDados:
    x_treino: NDArray
    y_treino: NDArray
    x_calibracao: NDArray
    y_calibracao: NDArray
    x_teste: NDArray
    y_teste: NDArray


def dividir_treino_calibracao_teste(
    X: ArrayLike,
    y: ArrayLike,
    *,
    fracao_treino: float = 0.56,
    fracao_calibracao: float = 0.24,
    fracao_teste: float = 0.20,
    semente_aleatoria: int = 125,
) -> DivisaoDados:
    if not np.isclose(fracao_treino + fracao_calibracao + fracao_teste, 1):
        raise ValueError("As três proporções devem somar 1.")
    matriz_x = np.asarray(X)
    vetor_y = np.asarray(y).reshape(-1)
    x_treino_calibracao, x_teste, y_treino_calibracao, y_teste = train_test_split(
        matriz_x,
        vetor_y,
        test_size=fracao_teste,
        random_state=semente_aleatoria,
    )
    calibracao_relativa = fracao_calibracao / (fracao_treino + fracao_calibracao)
    x_treino, x_calibracao, y_treino, y_calibracao = train_test_split(
        x_treino_calibracao,
        y_treino_calibracao,
        test_size=calibracao_relativa,
        random_state=semente_aleatoria,
    )
    return DivisaoDados(
        x_treino,
        y_treino,
        x_calibracao,
        y_calibracao,
        x_teste,
        y_teste,
    )
