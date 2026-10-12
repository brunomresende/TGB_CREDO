import numpy as np

from credal_cp_tgb.dados.divisao import dividir_treino_calibracao_teste


def teste_tamanhos_divisao_artigo():
    X = np.arange(1000)[:, None]
    y = np.arange(1000)
    divisao = dividir_treino_calibracao_teste(X, y, semente_aleatoria=1)
    assert len(divisao.x_treino) == 560
    assert len(divisao.x_calibracao) == 240
    assert len(divisao.x_teste) == 200
