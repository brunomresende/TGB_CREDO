"""Tabela pequena de CQR-QNN e CREDO-QNN no cenario sintetico."""

import pandas as pd

from credal_cp_tgb.credal.gamma import AdaptiveGamma, FixedGamma
from credal_cp_tgb.evaluation.metrics import evaluate_intervals
from credal_cp_tgb.methods.cqr import CQRRegressor
from credal_cp_tgb.methods.credo import CredalRegressor

from _shared import fitted_qnn, small_problem


split = small_problem()
qnn = fitted_qnn(split)
methods = {
    "CQR-QNN": CQRRegressor(qnn, base_model_is_fitted=True),
    "CQR-r-QNN": CQRRegressor(qnn, relative=True, base_model_is_fitted=True),
    "CREDO-QNN": CredalRegressor(qnn, FixedGamma(), base_model_is_fitted=True, n_posterior_samples=300),
    "CREDO-QNN-Adaptive": CredalRegressor(qnn, AdaptiveGamma(), base_model_is_fitted=True, n_posterior_samples=300),
}
rows = []
for name, method in methods.items():
    if isinstance(method, CredalRegressor):
        method.fit(split.X_train, split.y_train)
    method.calibrate(split.X_calibration, split.y_calibration)
    rows.append({"method": name, **evaluate_intervals(split.y_test, method.predict(split.X_test), alpha=0.1)})
print(pd.DataFrame(rows).to_string(index=False))

