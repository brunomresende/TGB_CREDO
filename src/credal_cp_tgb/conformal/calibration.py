"""Objeto pequeno que armazena e aplica a calibracao conformal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .scores import conformal_cutoff, quantile_scores, relative_quantile_scores


@dataclass
class ConformalCalibrator:
    alpha: float = 0.1
    relative: bool = False
    epsilon: float = 1e-5

    def fit(self, y: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> "ConformalCalibrator":
        score_fn = relative_quantile_scores if self.relative else quantile_scores
        if self.relative:
            self.scores_ = score_fn(y, lower, upper, epsilon=self.epsilon)
        else:
            self.scores_ = score_fn(y, lower, upper)
        self.cutoff_ = conformal_cutoff(self.scores_, self.alpha)
        return self

    def transform(self, lower: ArrayLike, upper: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "cutoff_"):
            raise RuntimeError("Execute fit antes de transform.")
        lower_arr = np.asarray(lower, dtype=float).reshape(-1)
        upper_arr = np.asarray(upper, dtype=float).reshape(-1)
        if lower_arr.size != upper_arr.size:
            raise ValueError("lower e upper devem ter o mesmo tamanho.")
        scale = np.maximum(upper_arr - lower_arr, self.epsilon) if self.relative else 1.0
        return np.column_stack(
            (lower_arr - self.cutoff_ * scale, upper_arr + self.cutoff_ * scale)
        )

