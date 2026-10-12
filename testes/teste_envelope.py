import numpy as np

from credal_cp_tgb.credal.envelope import construir_envelope_credal


def teste_envelope_corresponde_quantis_por_coluna():
    inferiores = np.array([[0, 10], [1, 11], [2, 12], [3, 13]])
    superiores = inferiores + 5
    gamma = np.array([0.5, 0.25])
    resultado = construir_envelope_credal(inferiores, superiores, gamma)
    assert resultado.shape == (2, 2)
    assert resultado[0, 0] == np.quantile(inferiores[:, 0], 0.25)
    assert resultado[1, 1] == np.quantile(superiores[:, 1], 0.875)
