"""Floresta aleatoria quantilica usada pelo CQR da Figura 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestRegressor

from .base import validar_predicoes_intervalares


@dataclass
class RegressorQuantilicoFloresta:
    """Floresta quantilica pela distribuicao das folhas das arvores.

    Para cada nova observacao, cada arvore atribui peso uniforme aos exemplos
    de treino que caem na mesma folha. A media desses pesos define a
    distribuicao condicional empirica da qual extraimos os dois quantis.
    """

    alpha: float = 0.1
    numero_arvores: int = 100
    maximo_atributos: str | float | int | None = "sqrt"
    minimo_amostras_folha: int = 5
    semente_aleatoria: int = 123
    numero_threads: int = -1

    def ajustar(self, X: ArrayLike, y: ArrayLike) -> RegressorQuantilicoFloresta:
        matriz_x = np.asarray(X, dtype=float)
        self.y_treino_ = np.asarray(y, dtype=float).reshape(-1)
        self.modelo_ = RandomForestRegressor(
            n_estimators=self.numero_arvores,
            max_features=self.maximo_atributos,
            min_samples_leaf=self.minimo_amostras_folha,
            bootstrap=True,
            random_state=self.semente_aleatoria,
            n_jobs=self.numero_threads,
        )
        self.modelo_.fit(matriz_x, self.y_treino_)
        self.folhas_treino_ = np.column_stack(
            [arvore.apply(matriz_x) for arvore in self.modelo_.estimators_]
        )
        return self

    @staticmethod
    def _quantil_ponderado(
        valores: NDArray[np.float64], pesos: NDArray[np.float64], quantil: float
    ) -> float:
        ordem = np.argsort(valores)
        valores_ordenados = valores[ordem]
        acumulados = np.cumsum(pesos[ordem])
        indice = min(np.searchsorted(acumulados, quantil, side="left"), len(valores) - 1)
        return float(valores_ordenados[indice])

    def prever_quantis(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "modelo_"):
            raise RuntimeError("A floresta quantilica ainda não foi treinada.")
        matriz_x = np.asarray(X, dtype=float)
        folhas_novas = np.column_stack(
            [arvore.apply(matriz_x) for arvore in self.modelo_.estimators_]
        )
        predicoes = np.empty((len(matriz_x), 2), dtype=float)
        quantis = (self.alpha / 2, 1 - self.alpha / 2)
        for indice, folhas_observacao in enumerate(folhas_novas):
            pesos = np.zeros(len(self.y_treino_), dtype=float)
            for coluna, folha in enumerate(folhas_observacao):
                membros = self.folhas_treino_[:, coluna] == folha
                pesos[membros] += 1.0 / (self.numero_arvores * membros.sum())
            predicoes[indice] = [
                self._quantil_ponderado(self.y_treino_, pesos, quantil)
                for quantil in quantis
            ]
        return validar_predicoes_intervalares(predicoes)
