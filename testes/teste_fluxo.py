import numpy as np

from credal_cp_tgb.credal.gamma import GammaFixo
from credal_cp_tgb.metodos.cqr import RegressorCQR
from credal_cp_tgb.metodos.credo import RegressorCREDO


class ModeloPosteriorFalso:
    def ajustar(self, X, y):
        return self

    def prever_quantis(self, X):
        centro = np.asarray(X)[:, 0]
        return np.column_stack((centro - 1, centro + 1))

    def amostrar_quantis(
        self,
        X,
        *,
        numero_amostras=100,
        semente_aleatoria=None,
    ):
        del semente_aleatoria
        centro = np.asarray(X)[:, 0]
        deslocamentos = np.linspace(-0.5, 0.5, numero_amostras)[:, None]
        return (
            centro[None, :] - 1 + deslocamentos,
            centro[None, :] + 1 + deslocamentos,
        )


def teste_cqr_e_credo_de_ponta_a_ponta():
    X = np.arange(20, dtype=float)[:, None]
    y = X[:, 0]
    modelo_base = ModeloPosteriorFalso()
    cqr = (
        RegressorCQR(modelo_base)
        .ajustar(X[:10], y[:10])
        .calibrar(X[10:15], y[10:15])
    )
    assert cqr.prever(X[15:]).shape == (5, 2)

    credo = RegressorCREDO(
        modelo_base,
        GammaFixo(0.1),
        numero_amostras_posteriores=20,
    )
    credo.ajustar(X[:10], y[:10]).calibrar(X[10:15], y[10:15])
    intervalos, parcelas = credo.prever(X[15:], retornar_decomposicao=True)
    assert intervalos.shape == (5, 2)
    assert set(parcelas) == {
        "aleatoria",
        "epistemica",
        "largura_nao_conformal",
        "folga_calibracao",
        "fracao_epistemica",
        "gamma",
    }
    np.testing.assert_allclose(
        parcelas["aleatoria"] + parcelas["epistemica"],
        parcelas["largura_nao_conformal"],
    )
    # A folga conformal e uma terceira parcela e nao deve ser atribuida a
    # incerteza aleatoria ou epistemica.
    np.testing.assert_allclose(
        intervalos[:, 1] - intervalos[:, 0],
        parcelas["largura_nao_conformal"] + parcelas["folga_calibracao"],
        atol=1e-14,
    )
