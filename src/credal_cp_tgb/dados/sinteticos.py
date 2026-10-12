"""Cenários pequenos para visualizar incertezas aleatória e epistêmica."""

from __future__ import annotations

import numpy as np
import pandas as pd


def criar_dados_lacuna_epistemica(
    numero_amostras: int = 1500,
    *,
    probabilidade_centro: float = 0.01,
    desvio_ruido: float = 0.1,
    semente_aleatoria: int = 125,
) -> pd.DataFrame:
    gerador = np.random.default_rng(semente_aleatoria)
    numero_centro = max(1, int(numero_amostras * probabilidade_centro))
    numero_esquerda = (numero_amostras - numero_centro) // 2
    numero_direita = numero_amostras - numero_centro - numero_esquerda
    x = np.concatenate(
        (
            gerador.uniform(-1.0, -0.2, numero_esquerda),
            gerador.uniform(-0.2, 0.2, numero_centro),
            gerador.uniform(0.2, 1.0, numero_direita),
        )
    )
    media = np.where(
        x < -0.2,
        -0.5 + 1.5 * (x + 0.8) - (x + 0.8) ** 2,
        np.where(x <= 0.2, 0.2 * np.sin(8 * x), 0.7 + 0.8 * (x - 0.5) ** 2),
    )
    y = media + gerador.normal(0, desvio_ruido, len(x))
    permutacao = gerador.permutation(len(x))
    return pd.DataFrame(
        {
            "x": x[permutacao],
            "y": y[permutacao],
            "media": media[permutacao],
        }
    )
