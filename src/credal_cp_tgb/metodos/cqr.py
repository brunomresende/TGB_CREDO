"""CQR e CQR-r independentes do modelo-base."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..conformal.calibracao import CalibradorConformal
from ..modelos.base import ModeloQuantilico, validar_predicoes_intervalares


@dataclass
class RegressorCQR:
    modelo_base: ModeloQuantilico
    alpha: float = 0.1
    relativo: bool = False
    modelo_base_ajustado: bool = False

    def ajustar(self, X: ArrayLike, y: ArrayLike) -> RegressorCQR:
        if not self.modelo_base_ajustado:
            self.modelo_base.ajustar(X, y)
            self.modelo_base_ajustado = True
        return self

    def calibrar(self, X: ArrayLike, y: ArrayLike) -> RegressorCQR:
        predicoes = validar_predicoes_intervalares(self.modelo_base.prever_quantis(X))
        self.calibrador_ = CalibradorConformal(alpha=self.alpha, relativo=self.relativo)
        self.calibrador_.ajustar(y, predicoes[:, 0], predicoes[:, 1])
        return self

    @property
    def limiar_(self) -> float:
        if not hasattr(self, "calibrador_"):
            raise RuntimeError("O método ainda não foi calibrado.")
        return self.calibrador_.limiar_

    def prever(
        self, X: ArrayLike, *, conformalizar: bool = True
    ) -> NDArray[np.float64]:
        predicoes = validar_predicoes_intervalares(self.modelo_base.prever_quantis(X))
        if not conformalizar:
            return predicoes
        if not hasattr(self, "calibrador_"):
            raise RuntimeError("Execute calibrar antes de prever.")
        return self.calibrador_.transformar(predicoes[:, 0], predicoes[:, 1])
