"""Exemplo minimo do envelope credal e de sua conformalizacao."""

from credal_cp_tgb.credal.gamma import FixedGamma
from credal_cp_tgb.methods.credo import CredalRegressor

from _shared import fitted_qnn, plot_intervals, small_problem


split = small_problem()
qnn = fitted_qnn(split)
credo = CredalRegressor(qnn, FixedGamma(0.1), base_model_is_fitted=True, n_posterior_samples=300)
credo.fit(split.X_train, split.y_train).calibrate(split.X_calibration, split.y_calibration)
raw = credo.predict(split.X_test, conformalize=False)
final = credo.predict(split.X_test)
print(f"Cutoff conformal: {credo.cutoff_:.4f}")
plot_intervals(split.X_test, split.y_test, raw, "Envelope credal", "credo_raw.png")
plot_intervals(split.X_test, split.y_test, final, "CREDO conformalizado", "credo_final.png")

