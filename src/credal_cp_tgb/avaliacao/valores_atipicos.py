"""Diagnóstico externo de valores atípicos com t-SNE e LOF, conforme o artigo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GruposValoresAtipicos:
    valores_atipicos: NDArray[np.int64]
    internos_ilr: NDArray[np.int64]
    internos_decomposicao: NDArray[np.int64]
    escores_lof: NDArray[np.float64]


def detectar_grupos_valores_atipicos(
    X: ArrayLike,
    *,
    numero_vizinhos: int = 15,
    contaminacao: float = 0.05,
    fracao_internos_ilr: float = 0.20,
    fracao_internos_decomposicao: float = 0.10,
    semente_aleatoria: int = 120,
) -> GruposValoresAtipicos:
    matriz_x = np.asarray(X, dtype=float)
    if len(matriz_x) < 4:
        raise ValueError("São necessárias ao menos quatro observações para t-SNE/LOF.")
    perplexidade = min(30.0, max(2.0, (len(matriz_x) - 1) / 3))
    representacao = TSNE(
        n_components=2,
        random_state=semente_aleatoria,
        perplexity=perplexidade,
        # PCA não inicializa duas dimensões quando X possui uma única característica.
        init="pca" if matriz_x.shape[1] >= 2 else "random",
        learning_rate="auto",
    ).fit_transform(matriz_x)
    representacao = StandardScaler().fit_transform(representacao)
    lof = LocalOutlierFactor(
        n_neighbors=min(numero_vizinhos, len(matriz_x) - 1),
        contamination=contaminacao,
    )
    rotulos = lof.fit_predict(representacao)
    valores_atipicos = np.flatnonzero(rotulos == -1)
    internos = np.flatnonzero(rotulos == 1)
    # ``negative_outlier_factor_`` pertence ao scikit-learn; valores menos
    # negativos representam pontos mais centrais.
    ordenados = internos[np.argsort(lof.negative_outlier_factor_[internos])[::-1]]
    numero_ilr = max(1, int(len(internos) * fracao_internos_ilr))
    numero_decomposicao = max(
        1,
        int(len(internos) * fracao_internos_decomposicao),
    )
    return GruposValoresAtipicos(
        valores_atipicos=valores_atipicos,
        internos_ilr=ordenados[:numero_ilr],
        internos_decomposicao=ordenados[:numero_decomposicao],
        escores_lof=lof.negative_outlier_factor_.astype(float),
    )
