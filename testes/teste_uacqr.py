import numpy as np

from credal_cp_tgb.metodos.uacqr import RegressorUACQR


class ModeloQuantilicoFalso:
    def prever_quantis(self, X):
        centro = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack((centro - 1.0, centro + 1.0))

    def amostrar_quantis(self, X, *, numero_amostras, semente_aleatoria):
        del semente_aleatoria
        centro = np.asarray(X, dtype=float)[:, 0]
        deslocamentos = np.linspace(-0.5, 0.5, numero_amostras)[:, None]
        return centro - 1.0 + deslocamentos, centro + 1.0 + deslocamentos


def teste_uacqr_p_usa_limite_extremo_finito_quando_calibracao_seleciona_sentinela():
    X_calibracao = np.arange(20, dtype=float)[:, None]
    y_calibracao = np.full(20, 1_000.0)
    metodo = RegressorUACQR(
        ModeloQuantilicoFalso(),
        tamanho_ensemble=10,
        modelo_base_ajustado=True,
    )

    metodo.calibrar(X_calibracao, y_calibracao)
    intervalos = metodo.prever(np.arange(5, dtype=float)[:, None])["UACQR-P"]

    assert metodo.limiar_p_ == metodo.tamanho_ensemble
    assert np.isfinite(intervalos).all()
