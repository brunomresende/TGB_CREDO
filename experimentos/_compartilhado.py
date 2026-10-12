from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from credal_cp_tgb.dados import criar_dados_lacuna_epistemica, dividir_treino_calibracao_teste
from credal_cp_tgb.modelos.rede_neural_quantilica import RedeNeuralQuantilica


def problema_pequeno(semente: int = 125):
    dados = criar_dados_lacuna_epistemica(700, semente_aleatoria=semente)
    return dividir_treino_calibracao_teste(
        dados[["x"]].to_numpy(),
        dados["y"].to_numpy(),
        semente_aleatoria=semente,
    )


def qnn_ajustada(divisao, alpha: float = 0.1, semente: int = 125):
    return RedeNeuralQuantilica(
        1,
        alpha=alpha,
        camadas_ocultas=(64, 64),
        epocas=300,
        paciencia=25,
        tamanho_lote=32,
        semente_aleatoria=semente,
    ).ajustar(divisao.x_treino, divisao.y_treino)


def plotar_intervalos(X, y, intervalos, titulo: str, arquivo_saida: str):
    ordem = np.argsort(X[:, 0])
    plt.figure(figsize=(8, 4.5))
    plt.scatter(X[:, 0], y, s=12, color="black", alpha=0.35)
    plt.fill_between(
        X[ordem, 0],
        intervalos[ordem, 0],
        intervalos[ordem, 1],
        alpha=0.3,
    )
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=200)
