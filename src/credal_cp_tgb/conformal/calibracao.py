"""Objeto pequeno que armazena e aplica a calibração conformal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .escores import escores_quantilicos, escores_quantilicos_relativos, limiar_conformal


@dataclass
class CalibradorConformal:
    alpha: float = 0.1
    relativo: bool = False
    epsilon: float = 1e-5

    def ajustar(
        self, y: ArrayLike, inferior: ArrayLike, superior: ArrayLike
    ) -> CalibradorConformal:
        funcao_escore = escores_quantilicos_relativos if self.relativo else escores_quantilicos
        if self.relativo:
            self.escores_ = funcao_escore(y, inferior, superior, epsilon=self.epsilon)
        else:
            self.escores_ = funcao_escore(y, inferior, superior)
        self.limiar_ = limiar_conformal(self.escores_, self.alpha)
        return self

    def transformar(self, inferior: ArrayLike, superior: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "limiar_"):
            raise RuntimeError("Execute ajustar antes de transformar.")
        vetor_inferior = np.asarray(inferior, dtype=float).reshape(-1)
        vetor_superior = np.asarray(superior, dtype=float).reshape(-1)
        if vetor_inferior.size != vetor_superior.size:
            raise ValueError("inferior e superior devem ter o mesmo tamanho.")
        escala = (
            np.maximum(vetor_superior - vetor_inferior, self.epsilon)
            if self.relativo
            else 1.0
        )
        return np.column_stack(
            (
                vetor_inferior - self.limiar_ * escala,
                vetor_superior + self.limiar_ * escala,
            )
        )
