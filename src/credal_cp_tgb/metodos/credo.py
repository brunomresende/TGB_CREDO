"""Envoltório CREDO pequeno e independente do modelo posterior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..conformal.calibracao import CalibradorConformal
from ..credal.envelope import construir_envelope_credal
from ..credal.gamma import GammaAdaptativo, GammaFixo
from ..modelos.base import ModeloQuantilicoPosterior


@dataclass
class RegressorCREDO:
    modelo_base: ModeloQuantilicoPosterior
    estrategia_gamma: GammaFixo | GammaAdaptativo
    alpha: float = 0.1
    numero_amostras_posteriores: int = 1000
    modelo_base_ajustado: bool = False
    semente_aleatoria: int = 125

    def ajustar(self, X: ArrayLike, y: ArrayLike) -> RegressorCREDO:
        if not self.modelo_base_ajustado:
            self.modelo_base.ajustar(X, y)
            self.modelo_base_ajustado = True
        self.estrategia_gamma.ajustar(X)
        return self

    def _envelope_bruto(
        self, X: ArrayLike, *, semente_aleatoria: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        amostras_inferiores, amostras_superiores = self.modelo_base.amostrar_quantis(
            X,
            numero_amostras=self.numero_amostras_posteriores,
            semente_aleatoria=(
                self.semente_aleatoria
                if semente_aleatoria is None
                else semente_aleatoria
            ),
        )
        gamma = self.estrategia_gamma.calcular(X)
        envelope = construir_envelope_credal(
            amostras_inferiores,
            amostras_superiores,
            gamma,
        )
        aleatoria = np.mean(amostras_superiores - amostras_inferiores, axis=0)
        total = envelope[:, 1] - envelope[:, 0]
        epistemica = total - aleatoria
        # Pequenos valores negativos podem surgir por Monte Carlo.
        epistemica = np.maximum(epistemica, 0.0)
        decomposicao = np.column_stack((aleatoria, epistemica, total))
        return envelope, gamma, decomposicao

    def calibrar(self, X: ArrayLike, y: ArrayLike) -> RegressorCREDO:
        envelope, gamma, decomposicao = self._envelope_bruto(X)
        self.calibrador_ = CalibradorConformal(alpha=self.alpha).ajustar(
            y, envelope[:, 0], envelope[:, 1]
        )
        self.gamma_calibracao_ = gamma
        self.decomposicao_calibracao_ = decomposicao
        return self

    @property
    def limiar_(self) -> float:
        if not hasattr(self, "calibrador_"):
            raise RuntimeError("O método ainda não foi calibrado.")
        return self.calibrador_.limiar_

    def prever(
        self,
        X: ArrayLike,
        *,
        conformalizar: bool = True,
        retornar_decomposicao: bool = False,
        semente_aleatoria: int | None = None,
    ):
        envelope, gamma, decomposicao = self._envelope_bruto(
            X,
            semente_aleatoria=semente_aleatoria,
        )
        intervalos = envelope
        if conformalizar:
            if not hasattr(self, "calibrador_"):
                raise RuntimeError("Execute calibrar antes de prever.")
            intervalos = self.calibrador_.transformar(envelope[:, 0], envelope[:, 1])
        self.ultimo_gamma_ = gamma
        if not retornar_decomposicao:
            return intervalos
        folga_calibracao = np.full(
            len(envelope),
            2 * self.limiar_ if conformalizar else 0.0,
        )
        resultado = {
            "aleatoria": decomposicao[:, 0],
            "epistemica": decomposicao[:, 1],
            "largura_nao_conformal": decomposicao[:, 2],
            "folga_calibracao": folga_calibracao,
            "fracao_epistemica": np.divide(
                decomposicao[:, 1],
                decomposicao[:, 2],
                out=np.zeros(len(envelope)),
                where=decomposicao[:, 2] > 0,
            ),
            "gamma": gamma,
        }
        return intervalos, resultado
