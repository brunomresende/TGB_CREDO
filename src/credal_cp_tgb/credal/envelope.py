"""Construção do envelope credal a partir de amostras de quantis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matriz_amostras(valores: ArrayLike, nome: str) -> NDArray[np.float64]:
    matriz = np.asarray(valores, dtype=float)
    if matriz.ndim != 2:
        raise ValueError(f"{nome} deve ter formato (n_amostras, n_observacoes).")
    return matriz


def construir_envelope_credal(
    amostras_inferiores: ArrayLike,
    amostras_superiores: ArrayLike,
    gamma: float | ArrayLike,
) -> NDArray[np.float64]:
    """Agrega amostras posteriores em um envelope ``[inferior, superior]``.

    Para cada observação x, aplica os quantis gamma(x)/2 nas fronteiras
    inferiores e 1-gamma(x)/2 nas fronteiras superiores.
    """
    inferiores = _matriz_amostras(amostras_inferiores, "amostras_inferiores")
    superiores = _matriz_amostras(amostras_superiores, "amostras_superiores")
    if inferiores.shape != superiores.shape:
        raise ValueError("As matrizes de amostras devem ter o mesmo formato.")

    numero_observacoes = inferiores.shape[1]
    valores_gamma = np.broadcast_to(np.asarray(gamma, dtype=float), (numero_observacoes,))
    if np.any((valores_gamma <= 0) | (valores_gamma >= 1)):
        raise ValueError("gamma deve estar estritamente entre 0 e 1.")

    envelope_inferior = np.empty(numero_observacoes, dtype=float)
    envelope_superior = np.empty(numero_observacoes, dtype=float)
    for indice, gamma_x in enumerate(valores_gamma):
        envelope_inferior[indice] = np.quantile(inferiores[:, indice], gamma_x / 2)
        envelope_superior[indice] = np.quantile(superiores[:, indice], 1 - gamma_x / 2)

    if np.any(envelope_inferior > envelope_superior):
        raise ValueError("O envelope gerado possui fronteiras cruzadas.")
    return np.column_stack((envelope_inferior, envelope_superior))
