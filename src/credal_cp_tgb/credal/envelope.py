"""Construcao do envelope credal a partir de amostras de quantis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _sample_matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} deve ter formato (n_amostras, n_observacoes).")
    return matrix


def build_credal_envelope(
    lower_samples: ArrayLike,
    upper_samples: ArrayLike,
    gamma: float | ArrayLike,
) -> NDArray[np.float64]:
    """Agrega amostras posteriores em um envelope ``[lower, upper]``.

    Para cada observacao x, aplica os quantis gamma(x)/2 nas fronteiras
    inferiores e 1-gamma(x)/2 nas fronteiras superiores.
    """
    lower = _sample_matrix(lower_samples, "lower_samples")
    upper = _sample_matrix(upper_samples, "upper_samples")
    if lower.shape != upper.shape:
        raise ValueError("As matrizes de amostras devem ter o mesmo formato.")

    n_observations = lower.shape[1]
    gamma_values = np.broadcast_to(np.asarray(gamma, dtype=float), (n_observations,))
    if np.any((gamma_values <= 0) | (gamma_values >= 1)):
        raise ValueError("gamma deve estar estritamente entre 0 e 1.")

    lower_envelope = np.empty(n_observations, dtype=float)
    upper_envelope = np.empty(n_observations, dtype=float)
    for index, gamma_x in enumerate(gamma_values):
        lower_envelope[index] = np.quantile(lower[:, index], gamma_x / 2)
        upper_envelope[index] = np.quantile(upper[:, index], 1 - gamma_x / 2)

    if np.any(lower_envelope > upper_envelope):
        raise ValueError("O envelope gerado possui fronteiras cruzadas.")
    return np.column_stack((lower_envelope, upper_envelope))

