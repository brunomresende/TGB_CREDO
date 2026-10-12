"""UACQR-S e UACQR-P usando amostras de um conjunto quantílico.

A implementação segue a geometria do código de referência: CQR-S padroniza
escores pela dispersão do conjunto; CQR-P calibra uma estatística de ordem que
seleciona limites mais extremos entre os modelos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..conformal.escores import limiar_conformal
from ..modelos.base import ModeloQuantilicoPosterior, validar_predicoes_intervalares


@dataclass
class RegressorUACQR:
    modelo_base: ModeloQuantilicoPosterior
    alpha: float = 0.1
    tamanho_ensemble: int = 1000
    modelo_base_ajustado: bool = False
    epsilon: float = 1e-5
    semente_aleatoria: int = 125

    def ajustar(self, X: ArrayLike, y: ArrayLike) -> RegressorUACQR:
        if not self.modelo_base_ajustado:
            self.modelo_base.ajustar(X, y)
            self.modelo_base_ajustado = True
        return self

    def _componentes(self, X: ArrayLike):
        base = validar_predicoes_intervalares(self.modelo_base.prever_quantis(X))
        amostras_inferiores, amostras_superiores = self.modelo_base.amostrar_quantis(
            X,
            numero_amostras=self.tamanho_ensemble,
            semente_aleatoria=self.semente_aleatoria,
        )
        amostras_inferiores = np.asarray(amostras_inferiores, dtype=float)
        amostras_superiores = np.asarray(amostras_superiores, dtype=float)
        if amostras_inferiores.shape != amostras_superiores.shape:
            raise ValueError("Amostras inferior e superior possuem formatos distintos.")
        # Sentinelas reproduzem o último nível conservador da referência.
        ordem_inferior = np.sort(
            np.vstack(
                (
                    amostras_inferiores,
                    np.full((1, amostras_inferiores.shape[1]), -np.inf),
                )
            ),
            axis=0,
        )[::-1]
        ordem_superior = np.sort(
            np.vstack(
                (
                    amostras_superiores,
                    np.full((1, amostras_superiores.shape[1]), np.inf),
                )
            ),
            axis=0,
        )
        return base, amostras_inferiores, amostras_superiores, ordem_inferior, ordem_superior

    def calibrar(self, X: ArrayLike, y: ArrayLike) -> RegressorUACQR:
        vetor_y = np.asarray(y, dtype=float).reshape(-1)
        base, inferiores, superiores, ordem_inferior, ordem_superior = self._componentes(X)

        dispersao_inferior = np.std(inferiores, axis=0, ddof=1) + self.epsilon
        dispersao_superior = np.std(superiores, axis=0, ddof=1) + self.epsilon
        escores_s = np.maximum(
            (base[:, 0] - vetor_y) / dispersao_inferior,
            (vetor_y - base[:, 1]) / dispersao_superior,
        )
        self.limiar_s_ = limiar_conformal(escores_s, self.alpha)

        inferior_necessario = np.argmax(ordem_inferior <= vetor_y[None, :], axis=0)
        superior_necessario = np.argmax(ordem_superior >= vetor_y[None, :], axis=0)
        # Se nenhum modelo alcança y, o índice extremo representa infinito no original.
        inferior_ausente = ~np.any(ordem_inferior <= vetor_y[None, :], axis=0)
        superior_ausente = ~np.any(ordem_superior >= vetor_y[None, :], axis=0)
        escores_percentis = np.maximum(inferior_necessario, superior_necessario).astype(float)
        escores_percentis[inferior_ausente | superior_ausente] = self.tamanho_ensemble
        self.limiar_p_ = int(np.ceil(limiar_conformal(escores_percentis, self.alpha)))
        return self

    def prever(self, X: ArrayLike) -> dict[str, NDArray[np.float64]]:
        if not hasattr(self, "limiar_s_"):
            raise RuntimeError("Execute calibrar antes de prever.")
        base, inferiores, superiores, ordem_inferior, ordem_superior = self._componentes(X)
        dispersao_inferior = np.std(inferiores, axis=0, ddof=1) + self.epsilon
        dispersao_superior = np.std(superiores, axis=0, ddof=1) + self.epsilon
        intervalos_s = np.column_stack(
            (
                base[:, 0] - self.limiar_s_ * dispersao_inferior,
                base[:, 1] + self.limiar_s_ * dispersao_superior,
            )
        )
        indice = min(self.limiar_p_, ordem_inferior.shape[0] - 1)
        intervalos_p = np.column_stack((ordem_inferior[indice], ordem_superior[indice]))
        return {"UACQR-S": intervalos_s, "UACQR-P": intervalos_p}
