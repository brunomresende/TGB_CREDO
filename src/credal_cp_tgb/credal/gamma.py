"""Estratégias fixa e adaptativa para o nível credal gamma."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class GammaFixo:
    gamma: float = 0.1

    def ajustar(self, X: ArrayLike) -> GammaFixo:
        if not 0 < self.gamma < 1:
            raise ValueError("gamma deve estar entre 0 e 1.")
        self.ajustado_ = True
        return self

    def calcular(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "ajustado_"):
            raise RuntimeError("Execute ajustar antes de calcular.")
        return np.full(np.asarray(X).shape[0], self.gamma, dtype=float)


@dataclass
class GammaAdaptativo:
    gamma_minimo: float = 0.1
    gamma_maximo: float = 0.75
    tau: float = 1.0
    epsilon: float = 1e-6
    constante_vizinhos: float = 6.672
    numero_vizinhos: int | None = None

    def ajustar(self, X: ArrayLike) -> GammaAdaptativo:
        matriz_x = np.asarray(X, dtype=float)
        if matriz_x.ndim != 2:
            raise ValueError("X deve ser uma matriz bidimensional.")
        if not 0 < self.gamma_minimo < self.gamma_maximo < 1:
            raise ValueError("Use 0 < gamma_minimo < gamma_maximo < 1.")
        if self.tau <= 0:
            raise ValueError("tau deve ser positivo.")

        numero_amostras, dimensao = matriz_x.shape
        if self.numero_vizinhos is None:
            k = int(
                np.ceil(
                    self.constante_vizinhos
                    * numero_amostras ** (4 / (4 + dimensao))
                )
            )
        else:
            k = int(self.numero_vizinhos)
        self.numero_vizinhos_ = max(2, min(k, numero_amostras))

        self.escalonador_ = StandardScaler().fit(matriz_x)
        dados_escalonados = self.escalonador_.transform(matriz_x)
        self.vizinhos_ = NearestNeighbors(n_neighbors=self.numero_vizinhos_).fit(
            dados_escalonados
        )
        distancias, _ = self.vizinhos_.kneighbors(dados_escalonados)
        distancias_k = distancias[:, -1]
        self.distancia_mediana_ = float(np.quantile(distancias_k, 0.50))
        self.distancia_alta_ = float(np.quantile(distancias_k, 0.95))
        return self

    @staticmethod
    def _sigmoide(valores: NDArray[np.float64]) -> NDArray[np.float64]:
        valores_limitados = np.clip(valores, -500, 500)
        return 1 / (1 + np.exp(-valores_limitados))

    def calcular(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "vizinhos_"):
            raise RuntimeError("Execute ajustar antes de calcular.")
        dados_escalonados = self.escalonador_.transform(np.asarray(X, dtype=float))
        distancias, _ = self.vizinhos_.kneighbors(dados_escalonados)
        distancia_k = np.log1p(distancias[:, -1])
        baixa = np.log1p(self.distancia_mediana_)
        alta = np.log1p(self.distancia_alta_)
        escassez = (distancia_k - baixa) / (alta - baixa + self.epsilon)
        gamma = self.gamma_maximo - (
            (self.gamma_maximo - self.gamma_minimo) * self._sigmoide(escassez / self.tau)
        )
        return np.clip(gamma, self.gamma_minimo, self.gamma_maximo)
