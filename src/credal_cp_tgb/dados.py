"""Leitura, divisão e geração dos dados usados nos exemplos e experimentos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DivisaoDados:
    """Partes de um conjunto usadas para treino, calibração e teste."""

    x_treino: NDArray
    y_treino: NDArray
    x_calibracao: NDArray
    y_calibracao: NDArray
    x_teste: NDArray
    y_teste: NDArray


def carregar_conjunto_processado(caminho: str | Path, alvo: str = "alvo"):
    """Lê um CSV processado e separa as variáveis explicativas do alvo."""
    dados = pd.read_csv(caminho)
    if alvo not in dados:
        raise ValueError(f"A coluna alvo {alvo!r} não existe em {caminho}.")
    return (
        dados.drop(columns=[alvo]).to_numpy(dtype=float),
        dados[alvo].to_numpy(dtype=float),
    )


def dividir_treino_calibracao_teste(
    X: ArrayLike,
    y: ArrayLike,
    *,
    fracao_treino: float = 0.56,
    fracao_calibracao: float = 0.24,
    fracao_teste: float = 0.20,
    semente_aleatoria: int = 125,
) -> DivisaoDados:
    """Faz a divisão reproduzível 56/24/20 usada no artigo."""
    if not np.isclose(fracao_treino + fracao_calibracao + fracao_teste, 1):
        raise ValueError("As três proporções devem somar 1.")
    matriz_x = np.asarray(X)
    vetor_y = np.asarray(y).reshape(-1)
    x_treino_calibracao, x_teste, y_treino_calibracao, y_teste = train_test_split(
        matriz_x,
        vetor_y,
        test_size=fracao_teste,
        random_state=semente_aleatoria,
    )
    calibracao_relativa = fracao_calibracao / (fracao_treino + fracao_calibracao)
    x_treino, x_calibracao, y_treino, y_calibracao = train_test_split(
        x_treino_calibracao,
        y_treino_calibracao,
        test_size=calibracao_relativa,
        random_state=semente_aleatoria,
    )
    return DivisaoDados(
        x_treino,
        y_treino,
        x_calibracao,
        y_calibracao,
        x_teste,
        y_teste,
    )


def criar_dados_ruido_variavel(
    numero_amostras: int = 800,
    *,
    desvio_base: float = 0.2,
    probabilidade_regiao_escassa: float = 0.02,
    semente_aleatoria: int = 42,
) -> pd.DataFrame:
    """Cria o cenário heteroscedástico usado na figura CREDO versus CQR.

    A maior parte dos valores de ``x`` fica fora do intervalo ``[0, 0.4]``.
    Essa região recebe apenas uma pequena fração das observações e, portanto,
    representa escassez de dados (incerteza epistêmica). O desvio do ruído muda
    com ``x``, representando também incerteza aleatória heteroscedástica.

    A função reproduz o processo gerador de ``make_variable_data_v3`` do
    repositório original, usando nomes em português. ``RandomState`` é mantido
    para gerar a mesma sequência da API antiga do NumPy, mas localmente, sem
    modificar o estado aleatório global do programa.
    """
    if numero_amostras < 2:
        raise ValueError("numero_amostras deve ser pelo menos 2.")
    if desvio_base <= 0:
        raise ValueError("desvio_base deve ser positivo.")
    if not 0 <= probabilidade_regiao_escassa <= 1:
        raise ValueError("probabilidade_regiao_escassa deve estar entre 0 e 1.")

    gerador = np.random.RandomState(semente_aleatoria)
    numero_escasso = int(numero_amostras * probabilidade_regiao_escassa)
    numero_denso = numero_amostras - numero_escasso

    # Amostragem por rejeicao: na parte densa, descartamos [0, 0.4].
    partes_densas: list[np.ndarray] = []
    quantidade_densa = 0
    while quantidade_densa < numero_denso:
        candidatos = gerador.uniform(-1.0, 1.0, numero_denso)
        aceitos = candidatos[(candidatos < 0.0) | (candidatos > 0.4)]
        partes_densas.append(aceitos)
        quantidade_densa += len(aceitos)
    if partes_densas:
        x_denso = np.concatenate(partes_densas)[:numero_denso]
    else:
        x_denso = np.empty(0, dtype=float)
    x_escasso = gerador.uniform(0.0, 0.4, numero_escasso)

    x = np.concatenate((x_denso, x_escasso))
    gerador.shuffle(x)
    media = x**3 + 2 * np.exp(-6 * (x - 0.3) ** 2)

    desvio_ruido = np.zeros_like(x)
    mascara_esquerda = x <= -0.3
    mascara_meio_esquerdo = (x > -0.3) & (x < 0.0)
    mascara_escassa = (x >= 0.0) & (x <= 0.4)
    mascara_direita = x > 0.4

    oscilacao_meio_esquerdo = np.abs(np.sin(10 * x[mascara_meio_esquerdo]))
    oscilacao_escassa = np.abs(np.sin(12 * x[mascara_escassa]))
    oscilacao_direita = np.abs(np.sin(8 * x[mascara_direita]))

    desvio_ruido[mascara_esquerda] = 0.1
    desvio_ruido[mascara_meio_esquerdo] = 0.2 + 0.15 * oscilacao_meio_esquerdo
    desvio_ruido[mascara_escassa] = 0.05 + 0.1 * oscilacao_escassa
    desvio_ruido[mascara_direita] = (
        0.25 + 0.5 * (x[mascara_direita] - 0.4) + 0.15 * oscilacao_direita
    )
    desvio_ruido *= desvio_base / 0.2
    y = media + gerador.normal(0.0, desvio_ruido, numero_amostras)

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "media": media,
            "desvio_ruido": desvio_ruido,
        }
    )


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
