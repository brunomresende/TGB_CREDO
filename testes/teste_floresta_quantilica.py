import numpy as np

from credal_cp_tgb.modelos.floresta_quantilica import RegressorQuantilicoFloresta


def teste_floresta_quantilica_produz_intervalos_ordenados():
    X = np.linspace(-1, 1, 60)[:, None]
    y = X[:, 0] ** 2
    modelo = RegressorQuantilicoFloresta(
        numero_arvores=10,
        minimo_amostras_folha=3,
        numero_threads=1,
    ).ajustar(X, y)

    intervalos = modelo.prever_quantis(X[:7])

    assert intervalos.shape == (7, 2)
    assert np.all(np.isfinite(intervalos))
    assert np.all(intervalos[:, 0] <= intervalos[:, 1])
