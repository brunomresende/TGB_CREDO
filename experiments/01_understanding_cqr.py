"""Exemplo minimo: quantis da QNN, scores e expansao conformal."""

from _shared import fitted_qnn, plot_intervals, small_problem

from credal_cp_tgb.methods.cqr import CQRRegressor

split = small_problem()
qnn = fitted_qnn(split)
cqr = CQRRegressor(qnn, base_model_is_fitted=True).calibrate(
    split.X_calibration, split.y_calibration
)
print(f"Cutoff conformal: {cqr.cutoff_:.4f}")
plot_intervals(split.X_test, split.y_test, cqr.predict(split.X_test), "CQR-QNN", "cqr_qnn.png")

