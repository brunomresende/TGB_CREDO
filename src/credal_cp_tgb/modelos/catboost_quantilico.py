"""Par de regressores quantílicos CatBoost com importação opcional."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import validar_predicoes_intervalares


@dataclass
class RegressorQuantilicoCatBoost:
    alpha: float = 0.1
    iteracoes: int = 1000
    profundidade: int = 6
    regularizacao_l2_folha: float = 3.0
    temperatura_bagging: float = 1.0
    semente_aleatoria: int = 125
    numero_threads: int = -1
    parametros_extras: dict = field(default_factory=dict)

    def ajustar(self, X: ArrayLike, y: ArrayLike) -> RegressorQuantilicoCatBoost:
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise ImportError("Instale o extra catboost: pip install -e '.[catboost]'") from error

        parametros_comuns = dict(
            iterations=self.iteracoes,
            depth=self.profundidade,
            l2_leaf_reg=self.regularizacao_l2_folha,
            bagging_temperature=self.temperatura_bagging,
            random_seed=self.semente_aleatoria,
            thread_count=self.numero_threads,
            verbose=False,
            **self.parametros_extras,
        )
        self.modelo_inferior_ = CatBoostRegressor(
            loss_function=f"Quantile:alpha={self.alpha / 2}", **parametros_comuns
        )
        self.modelo_superior_ = CatBoostRegressor(
            loss_function=f"Quantile:alpha={1 - self.alpha / 2}", **parametros_comuns
        )
        self.modelo_inferior_.fit(X, y)
        self.modelo_superior_.fit(X, y)
        return self

    def prever_quantis(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "modelo_inferior_"):
            raise RuntimeError("O CatBoost ainda não foi treinado.")
        predicoes = np.column_stack(
            (self.modelo_inferior_.predict(X), self.modelo_superior_.predict(X))
        )
        # Modelos separados podem cruzar; ordenar e registrar o ocorrido.
        self.ultimo_numero_cruzamentos_ = int(np.sum(predicoes[:, 0] > predicoes[:, 1]))
        predicoes.sort(axis=1)
        return validar_predicoes_intervalares(predicoes)

    def amostrar_quantis(
        self,
        X: ArrayLike,
        *,
        numero_amostras: int | None = None,
        semente_aleatoria: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Retorna previsoes acumuladas ao longo das iteracoes do boosting.

        Essa é a representação de conjunto empregada pela implementação UACQR
        original com CatBoost: cada número de árvores define um estimador-base.
        """
        del semente_aleatoria
        if not hasattr(self, "modelo_inferior_"):
            raise RuntimeError("O CatBoost ainda não foi treinado.")
        total = min(self.modelo_inferior_.tree_count_, self.modelo_superior_.tree_count_)
        quantidade = total if numero_amostras is None else min(int(numero_amostras), total)
        if quantidade < 2:
            raise ValueError("São necessárias pelo menos duas iterações do CatBoost.")
        finais_arvores = np.unique(np.linspace(1, total, quantidade, dtype=int))
        inferiores = np.asarray(
            [
                self.modelo_inferior_.predict(X, ntree_end=int(fim))
                for fim in finais_arvores
            ]
        )
        superiores = np.asarray(
            [
                self.modelo_superior_.predict(X, ntree_end=int(fim))
                for fim in finais_arvores
            ]
        )
        return inferiores, superiores
