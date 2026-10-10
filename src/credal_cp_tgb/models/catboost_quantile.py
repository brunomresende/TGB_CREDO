"""Par de regressores quantilicos CatBoost com importacao opcional."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import validate_interval_predictions


@dataclass
class CatBoostQuantileRegressor:
    alpha: float = 0.1
    iterations: int = 1000
    depth: int = 6
    l2_leaf_reg: float = 3.0
    bagging_temperature: float = 1.0
    random_state: int = 125
    thread_count: int = -1
    extra_params: dict = field(default_factory=dict)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "CatBoostQuantileRegressor":
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise ImportError("Instale o extra catboost: pip install -e '.[catboost]'") from error

        common = dict(
            iterations=self.iterations,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            bagging_temperature=self.bagging_temperature,
            random_seed=self.random_state,
            thread_count=self.thread_count,
            verbose=False,
            **self.extra_params,
        )
        self.lower_model_ = CatBoostRegressor(
            loss_function=f"Quantile:alpha={self.alpha / 2}", **common
        )
        self.upper_model_ = CatBoostRegressor(
            loss_function=f"Quantile:alpha={1 - self.alpha / 2}", **common
        )
        self.lower_model_.fit(X, y)
        self.upper_model_.fit(X, y)
        return self

    def predict_quantiles(self, X: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "lower_model_"):
            raise RuntimeError("O CatBoost ainda nao foi treinado.")
        predictions = np.column_stack(
            (self.lower_model_.predict(X), self.upper_model_.predict(X))
        )
        # Modelos separados podem cruzar; ordenar e registrar o ocorrido.
        self.last_crossing_count_ = int(np.sum(predictions[:, 0] > predictions[:, 1]))
        predictions.sort(axis=1)
        return validate_interval_predictions(predictions)

    def sample_quantiles(
        self,
        X: ArrayLike,
        *,
        n_samples: int | None = None,
        random_state: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Retorna previsoes acumuladas ao longo das iteracoes do boosting.

        Essa e a representacao de ensemble empregada pela implementacao UACQR
        original com CatBoost: cada numero de arvores define um estimador-base.
        """
        del random_state
        if not hasattr(self, "lower_model_"):
            raise RuntimeError("O CatBoost ainda nao foi treinado.")
        total = min(self.lower_model_.tree_count_, self.upper_model_.tree_count_)
        count = total if n_samples is None else min(int(n_samples), total)
        if count < 2:
            raise ValueError("Sao necessarias pelo menos duas iteracoes do CatBoost.")
        tree_ends = np.unique(np.linspace(1, total, count, dtype=int))
        lower = np.asarray(
            [self.lower_model_.predict(X, ntree_end=int(end)) for end in tree_ends]
        )
        upper = np.asarray(
            [self.upper_model_.predict(X, ntree_end=int(end)) for end in tree_ends]
        )
        return lower, upper
