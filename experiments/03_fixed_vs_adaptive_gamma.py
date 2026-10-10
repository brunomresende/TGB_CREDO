"""Compara CREDO-QNN fixo e adaptativo no mesmo modelo treinado."""

from credal_cp_tgb.credal.gamma import AdaptiveGamma, FixedGamma
from credal_cp_tgb.methods.credo import CredalRegressor

from _shared import fitted_qnn, plot_intervals, small_problem


split = small_problem()
qnn = fitted_qnn(split)
strategies = {"fixed": FixedGamma(0.1), "adaptive": AdaptiveGamma()}
for name, strategy in strategies.items():
    method = CredalRegressor(qnn, strategy, base_model_is_fitted=True, n_posterior_samples=300)
    method.fit(split.X_train, split.y_train).calibrate(split.X_calibration, split.y_calibration)
    intervals = method.predict(split.X_test)
    plot_intervals(split.X_test, split.y_test, intervals, f"CREDO {name}", f"credo_{name}.png")

