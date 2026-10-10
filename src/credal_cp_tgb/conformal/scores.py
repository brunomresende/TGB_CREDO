"""Scores e quantis da calibracao conformal."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vectors(y: ArrayLike, lower: ArrayLike, upper: ArrayLike):
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    lower_arr = np.asarray(lower, dtype=float).reshape(-1)
    upper_arr = np.asarray(upper, dtype=float).reshape(-1)
    if not (y_arr.size == lower_arr.size == upper_arr.size):
        raise ValueError("y, lower e upper devem ter o mesmo tamanho.")
    if np.any(lower_arr > upper_arr):
        raise ValueError("Todo limite inferior deve ser menor ou igual ao superior.")
    return y_arr, lower_arr, upper_arr


def quantile_scores(y: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> NDArray[np.float64]:
    """Calcula max(lower - y, y - upper), permitindo scores negativos."""
    y_arr, lower_arr, upper_arr = _vectors(y, lower, upper)
    return np.maximum(lower_arr - y_arr, y_arr - upper_arr)


def relative_quantile_scores(
    y: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    epsilon: float = 1e-5,
) -> NDArray[np.float64]:
    """Score do CQR-r normalizado pela largura do intervalo inicial."""
    y_arr, lower_arr, upper_arr = _vectors(y, lower, upper)
    width = np.maximum(upper_arr - lower_arr, epsilon)
    return np.maximum((lower_arr - y_arr) / width, (y_arr - upper_arr) / width)


def conformal_cutoff(scores: ArrayLike, alpha: float) -> float:
    """Quantil split-conformal com correcao exata de amostra finita.

    Retorna a estatistica de ordem k = ceil((n + 1)(1-alpha)), truncada em n.
    Esta forma evita diferencas entre metodos de interpolacao de ``np.quantile``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar estritamente entre 0 e 1.")
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("scores deve conter ao menos um valor finito.")
    rank = min(int(np.ceil((values.size + 1) * (1 - alpha))), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])

