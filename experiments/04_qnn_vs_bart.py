"""Compara backends QNN e BART; requer o extra ``bart``."""

from credal_cp_tgb.credal.gamma import AdaptiveGamma
from credal_cp_tgb.methods.credo import CredalRegressor
from credal_cp_tgb.models.bart import HeteroscedasticBART

from _shared import fitted_qnn, plot_intervals, small_problem


split = small_problem()
models = {
    "QNN": fitted_qnn(split),
    "BART": HeteroscedasticBART(draws=500, chains=2, cores=2).fit(split.X_train, split.y_train),
}
for name, model in models.items():
    method = CredalRegressor(model, AdaptiveGamma(), base_model_is_fitted=True, n_posterior_samples=300)
    method.fit(split.X_train, split.y_train).calibrate(split.X_calibration, split.y_calibration)
    plot_intervals(split.X_test, split.y_test, method.predict(split.X_test), f"CREDO-{name}", f"credo_{name.lower()}.png")

