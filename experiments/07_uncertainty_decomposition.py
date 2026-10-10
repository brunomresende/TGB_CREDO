"""Mostra as parcelas aleatoria, epistemica e conformal do CREDO."""

import matplotlib.pyplot as plt
import numpy as np

from credal_cp_tgb.credal.gamma import AdaptiveGamma
from credal_cp_tgb.methods.credo import CredalRegressor

from _shared import fitted_qnn, small_problem


split = small_problem()
qnn = fitted_qnn(split)
method = CredalRegressor(qnn, AdaptiveGamma(), base_model_is_fitted=True, n_posterior_samples=300)
method.fit(split.X_train, split.y_train).calibrate(split.X_calibration, split.y_calibration)
_, parts = method.predict(split.X_test, return_decomposition=True)
order = np.argsort(split.X_test[:, 0])
for key in ("aleatoric", "epistemic", "calibration_slack"):
    plt.plot(split.X_test[order, 0], parts[key][order], label=key)
plt.xlabel("x")
plt.ylabel("comprimento")
plt.legend()
plt.tight_layout()
plt.savefig("uncertainty_decomposition.png", dpi=200)

