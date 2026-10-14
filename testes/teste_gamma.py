import numpy as np

from credal_cp_tgb.credal.gamma import GammaAdaptativo, GammaFixo


def teste_gamma_fixo_retornar_constante():
    modelo = GammaFixo(0.2).ajustar(np.zeros((5, 2)))
    np.testing.assert_allclose(modelo.calcular(np.ones((3, 2))), 0.2)


def teste_gamma_adaptativo_ser_limitado_e_menor_longe():
    gerador = np.random.default_rng(1)
    X = gerador.normal(0, 0.1, size=(100, 1))
    modelo = GammaAdaptativo(
        gamma_minimo=0.1,
        gamma_maximo=0.75,
        numero_vizinhos=10,
    ).ajustar(X)
    gamma = modelo.calcular(np.array([[0.0], [10.0]]))
    assert np.all((0.1 <= gamma) & (gamma <= 0.75))
    assert gamma[1] < gamma[0]


def teste_gamma_padrao_detecta_lacuna_central():
    esquerda = np.linspace(-1.0, -0.2, 100, endpoint=False)
    direita = np.linspace(0.2, 1.0, 100)
    modelo = GammaAdaptativo(gamma_minimo=0.05).ajustar(
        np.concatenate((esquerda, direita))[:, None]
    )

    gamma_denso_esquerda, gamma_lacuna, gamma_denso_direita = modelo.calcular(
        np.array([[-0.6], [0.0], [0.6]])
    )

    assert modelo.numero_vizinhos_ == 6
    assert gamma_lacuna < gamma_denso_esquerda
    assert gamma_lacuna < gamma_denso_direita
