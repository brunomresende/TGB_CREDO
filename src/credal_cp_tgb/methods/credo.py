"""Wrapper CREDO pequeno, independente do backend posterior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..conformal.calibration import ConformalCalibrator
from ..credal.envelope import build_credal_envelope
from ..credal.gamma import AdaptiveGamma, FixedGamma
from ..models.base import PosteriorQuantileModel


@dataclass
class CredalRegressor:
    base_model: PosteriorQuantileModel
    gamma_strategy: FixedGamma | AdaptiveGamma
    alpha: float = 0.1
    n_posterior_samples: int = 1000
    base_model_is_fitted: bool = False
    random_state: int = 125

    def fit(self, X: ArrayLike, y: ArrayLike) -> CredalRegressor:
        if not self.base_model_is_fitted:
            self.base_model.fit(X, y)
            self.base_model_is_fitted = True
        self.gamma_strategy.fit(X)
        return self

    def _raw_envelope(
        self, X: ArrayLike, *, random_state: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        lower_samples, upper_samples = self.base_model.sample_quantiles(
            X,
            n_samples=self.n_posterior_samples,
            random_state=self.random_state if random_state is None else random_state,
        )
        gamma = self.gamma_strategy.compute(X)
        envelope = build_credal_envelope(lower_samples, upper_samples, gamma)
        aleatoric = np.mean(upper_samples - lower_samples, axis=0)
        total = envelope[:, 1] - envelope[:, 0]
        epistemic = total - aleatoric
        # Pequenos valores negativos podem surgir por Monte Carlo.
        epistemic = np.maximum(epistemic, 0.0)
        decomposition = np.column_stack((aleatoric, epistemic, total))
        return envelope, gamma, decomposition

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> CredalRegressor:
        envelope, gamma, decomposition = self._raw_envelope(X)
        self.calibrator_ = ConformalCalibrator(alpha=self.alpha).fit(
            y, envelope[:, 0], envelope[:, 1]
        )
        self.calibration_gamma_ = gamma
        self.calibration_decomposition_ = decomposition
        return self

    @property
    def cutoff_(self) -> float:
        if not hasattr(self, "calibrator_"):
            raise RuntimeError("O metodo ainda nao foi calibrado.")
        return self.calibrator_.cutoff_

    def predict(
        self,
        X: ArrayLike,
        *,
        conformalize: bool = True,
        return_decomposition: bool = False,
        random_state: int | None = None,
    ):
        envelope, gamma, decomposition = self._raw_envelope(X, random_state=random_state)
        intervals = envelope
        if conformalize:
            if not hasattr(self, "calibrator_"):
                raise RuntimeError("Execute calibrate antes de predict.")
            intervals = self.calibrator_.transform(envelope[:, 0], envelope[:, 1])
        self.last_gamma_ = gamma
        if not return_decomposition:
            return intervals
        calibration_slack = np.full(len(envelope), 2 * self.cutoff_ if conformalize else 0.0)
        result = {
            "aleatoric": decomposition[:, 0],
            "epistemic": decomposition[:, 1],
            "nonconformal_width": decomposition[:, 2],
            "calibration_slack": calibration_slack,
            "epistemic_fraction": np.divide(
                decomposition[:, 1],
                decomposition[:, 2],
                out=np.zeros(len(envelope)),
                where=decomposition[:, 2] > 0,
            ),
            "gamma": gamma,
        }
        return intervals, result

