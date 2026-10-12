"""Escores e quantis da calibração conformal."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vetores(y: ArrayLike, inferior: ArrayLike, superior: ArrayLike):
    vetor_y = np.asarray(y, dtype=float).reshape(-1)
    vetor_inferior = np.asarray(inferior, dtype=float).reshape(-1)
    vetor_superior = np.asarray(superior, dtype=float).reshape(-1)
    if not (vetor_y.size == vetor_inferior.size == vetor_superior.size):
        raise ValueError("y, inferior e superior devem ter o mesmo tamanho.")
    if np.any(vetor_inferior > vetor_superior):
        raise ValueError("Todo limite inferior deve ser menor ou igual ao superior.")
    return vetor_y, vetor_inferior, vetor_superior


def escores_quantilicos(
    y: ArrayLike, inferior: ArrayLike, superior: ArrayLike
) -> NDArray[np.float64]:
    """Calcula ``max(inferior - y, y - superior)``, permitindo escores negativos."""
    vetor_y, vetor_inferior, vetor_superior = _vetores(y, inferior, superior)
    return np.maximum(vetor_inferior - vetor_y, vetor_y - vetor_superior)


def escores_quantilicos_relativos(
    y: ArrayLike,
    inferior: ArrayLike,
    superior: ArrayLike,
    *,
    epsilon: float = 1e-5,
) -> NDArray[np.float64]:
    """Escore do CQR-r normalizado pela largura do intervalo inicial."""
    vetor_y, vetor_inferior, vetor_superior = _vetores(y, inferior, superior)
    largura = np.maximum(vetor_superior - vetor_inferior, epsilon)
    return np.maximum(
        (vetor_inferior - vetor_y) / largura,
        (vetor_y - vetor_superior) / largura,
    )


def limiar_conformal(escores: ArrayLike, alpha: float) -> float:
    """Quantil split-conformal com correcao exata de amostra finita.

    Retorna a estatística de ordem k = ceil((n + 1)(1-alpha)), truncada em n.
    Esta forma evita diferenças entre métodos de interpolação de ``np.quantile``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar estritamente entre 0 e 1.")
    valores = np.asarray(escores, dtype=float).reshape(-1)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        raise ValueError("escores deve conter ao menos um valor finito.")
    posto = min(int(np.ceil((valores.size + 1) * (1 - alpha))), valores.size)
    return float(np.partition(valores, posto - 1)[posto - 1])
