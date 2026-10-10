"""CQR e CQR-r independentes do modelo-base."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..conformal.calibration import ConformalCalibrator
from ..models.base import QuantileModel, validate_interval_predictions


@dataclass
class CQRRegressor:
    base_model: QuantileModel
    alpha: float = 0.1
    relative: bool = False
    base_model_is_fitted: bool = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> CQRRegressor:
        if not self.base_model_is_fitted:
            self.base_model.fit(X, y)
            self.base_model_is_fitted = True
        return self

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> CQRRegressor:
        predictions = validate_interval_predictions(self.base_model.predict_quantiles(X))
        self.calibrator_ = ConformalCalibrator(alpha=self.alpha, relative=self.relative)
        self.calibrator_.fit(y, predictions[:, 0], predictions[:, 1])
        return self

    @property
    def cutoff_(self) -> float:
        if not hasattr(self, "calibrator_"):
            raise RuntimeError("O metodo ainda nao foi calibrado.")
        return self.calibrator_.cutoff_

    def predict(self, X: ArrayLike, *, conformalize: bool = True) -> NDArray[np.float64]:
        predictions = validate_interval_predictions(self.base_model.predict_quantiles(X))
        if not conformalize:
            return predictions
        if not hasattr(self, "calibrator_"):
            raise RuntimeError("Execute calibrate antes de predict.")
        return self.calibrator_.transform(predictions[:, 0], predictions[:, 1])

