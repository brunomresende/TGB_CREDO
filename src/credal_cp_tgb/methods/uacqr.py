"""UACQR-S e UACQR-P usando amostras de um ensemble quantilico.

A implementacao segue a geometria do codigo de referencia: CQR-S padroniza
scores pela dispersao do ensemble; CQR-P calibra uma estatistica de ordem que
seleciona limites mais extremos entre os modelos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..conformal.scores import conformal_cutoff
from ..models.base import PosteriorQuantileModel, validate_interval_predictions


@dataclass
class UACQRRegressor:
    base_model: PosteriorQuantileModel
    alpha: float = 0.1
    ensemble_size: int = 1000
    base_model_is_fitted: bool = False
    epsilon: float = 1e-5
    random_state: int = 125

    def fit(self, X: ArrayLike, y: ArrayLike) -> UACQRRegressor:
        if not self.base_model_is_fitted:
            self.base_model.fit(X, y)
            self.base_model_is_fitted = True
        return self

    def _components(self, X: ArrayLike):
        base = validate_interval_predictions(self.base_model.predict_quantiles(X))
        lower_draws, upper_draws = self.base_model.sample_quantiles(
            X, n_samples=self.ensemble_size, random_state=self.random_state
        )
        lower_draws = np.asarray(lower_draws, dtype=float)
        upper_draws = np.asarray(upper_draws, dtype=float)
        if lower_draws.shape != upper_draws.shape:
            raise ValueError("Amostras inferior e superior possuem formatos distintos.")
        # Sentinelas reproduzem o ultimo nivel conservador da referencia.
        lower_order = np.sort(
            np.vstack((lower_draws, np.full((1, lower_draws.shape[1]), -np.inf))), axis=0
        )[::-1]
        upper_order = np.sort(
            np.vstack((upper_draws, np.full((1, upper_draws.shape[1]), np.inf))), axis=0
        )
        return base, lower_draws, upper_draws, lower_order, upper_order

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> UACQRRegressor:
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        base, lower, upper, lower_order, upper_order = self._components(X)

        spread_lower = np.std(lower, axis=0, ddof=1) + self.epsilon
        spread_upper = np.std(upper, axis=0, ddof=1) + self.epsilon
        scores_s = np.maximum(
            (base[:, 0] - y_arr) / spread_lower,
            (y_arr - base[:, 1]) / spread_upper,
        )
        self.cutoff_s_ = conformal_cutoff(scores_s, self.alpha)

        lower_needed = np.argmax(lower_order <= y_arr[None, :], axis=0)
        upper_needed = np.argmax(upper_order >= y_arr[None, :], axis=0)
        # Se nenhum modelo alcanca y, o indice extremo representa infinito no original.
        lower_missing = ~np.any(lower_order <= y_arr[None, :], axis=0)
        upper_missing = ~np.any(upper_order >= y_arr[None, :], axis=0)
        percentile_scores = np.maximum(lower_needed, upper_needed).astype(float)
        percentile_scores[lower_missing | upper_missing] = self.ensemble_size
        self.cutoff_p_ = int(np.ceil(conformal_cutoff(percentile_scores, self.alpha)))
        return self

    def predict(self, X: ArrayLike) -> dict[str, NDArray[np.float64]]:
        if not hasattr(self, "cutoff_s_"):
            raise RuntimeError("Execute calibrate antes de predict.")
        base, lower, upper, lower_order, upper_order = self._components(X)
        spread_lower = np.std(lower, axis=0, ddof=1) + self.epsilon
        spread_upper = np.std(upper, axis=0, ddof=1) + self.epsilon
        intervals_s = np.column_stack(
            (
                base[:, 0] - self.cutoff_s_ * spread_lower,
                base[:, 1] + self.cutoff_s_ * spread_upper,
            )
        )
        index = min(self.cutoff_p_, lower_order.shape[0] - 1)
        intervals_p = np.column_stack((lower_order[index], upper_order[index]))
        return {"UACQR-S": intervals_s, "UACQR-P": intervals_p}
