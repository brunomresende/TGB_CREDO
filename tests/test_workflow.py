import numpy as np

from credal_cp_tgb.credal.gamma import FixedGamma
from credal_cp_tgb.methods.cqr import CQRRegressor
from credal_cp_tgb.methods.credo import CredalRegressor


class FakePosteriorModel:
    def fit(self, X, y):
        return self

    def predict_quantiles(self, X):
        center = np.asarray(X)[:, 0]
        return np.column_stack((center - 1, center + 1))

    def sample_quantiles(self, X, *, n_samples=100, random_state=None):
        center = np.asarray(X)[:, 0]
        offsets = np.linspace(-0.5, 0.5, n_samples)[:, None]
        return center[None, :] - 1 + offsets, center[None, :] + 1 + offsets


def test_cqr_and_credo_end_to_end():
    X = np.arange(20, dtype=float)[:, None]
    y = X[:, 0]
    base = FakePosteriorModel()
    cqr = CQRRegressor(base).fit(X[:10], y[:10]).calibrate(X[10:15], y[10:15])
    assert cqr.predict(X[15:]).shape == (5, 2)

    credo = CredalRegressor(base, FixedGamma(0.1), n_posterior_samples=20)
    credo.fit(X[:10], y[:10]).calibrate(X[10:15], y[10:15])
    intervals, parts = credo.predict(X[15:], return_decomposition=True)
    assert intervals.shape == (5, 2)
    assert set(parts) == {
        "aleatoric",
        "epistemic",
        "nonconformal_width",
        "calibration_slack",
        "epistemic_fraction",
        "gamma",
    }

