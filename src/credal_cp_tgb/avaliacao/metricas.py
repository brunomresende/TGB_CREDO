"""Métricas usadas no artigo e diagnósticos complementares."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def indicadores_cobertura(
    y: ArrayLike,
    inferior: ArrayLike,
    superior: ArrayLike,
) -> np.ndarray:
    vetor_y = np.asarray(y).reshape(-1)
    return (np.asarray(inferior).reshape(-1) <= vetor_y) & (
        vetor_y <= np.asarray(superior).reshape(-1)
    )


def cobertura_marginal(y: ArrayLike, inferior: ArrayLike, superior: ArrayLike) -> float:
    """AMC: proporção de respostas contidas nos intervalos."""
    return float(np.mean(indicadores_cobertura(y, inferior, superior)))


def comprimento_medio_intervalo(inferior: ArrayLike, superior: ArrayLike) -> float:
    """IL: comprimento médio dos intervalos; menor é melhor sob cobertura válida."""
    return float(np.mean(np.asarray(superior) - np.asarray(inferior)))


def escore_medio_intervalo_escalonado(
    y: ArrayLike,
    inferior: ArrayLike,
    superior: ArrayLike,
    alpha: float,
) -> float:
    """SMIS, ou escore médio do intervalo; menor é melhor."""
    vetor_y = np.asarray(y).reshape(-1)
    vetor_inferior = np.asarray(inferior).reshape(-1)
    vetor_superior = np.asarray(superior).reshape(-1)
    escores = (
        vetor_superior
        - vetor_inferior
        + (2 / alpha) * (vetor_inferior - vetor_y) * (vetor_y < vetor_inferior)
        + (2 / alpha) * (vetor_y - vetor_superior) * (vetor_y > vetor_superior)
    )
    return float(np.mean(escores))


def correlacao_cobertura_largura(
    y: ArrayLike,
    inferior: ArrayLike,
    superior: ArrayLike,
) -> float:
    """Correlação absoluta entre o indicador de cobertura e a largura."""
    cobertos = indicadores_cobertura(y, inferior, superior).astype(float)
    larguras = np.asarray(superior).reshape(-1) - np.asarray(inferior).reshape(-1)
    if np.std(cobertos) == 0 or np.std(larguras) == 0:
        return float("nan")
    return float(abs(np.corrcoef(cobertos, larguras)[0, 1]))


def razao_comprimento_atipicos_internos(
    inferior: ArrayLike,
    superior: ArrayLike,
    indices_atipicos: ArrayLike,
    indices_internos: ArrayLike,
) -> float:
    """ILR: largura média nos atípicos dividida pela largura nos internos centrais."""
    larguras = np.asarray(superior).reshape(-1) - np.asarray(inferior).reshape(-1)
    indices_atipicos = np.asarray(indices_atipicos, dtype=int)
    indices_internos = np.asarray(indices_internos, dtype=int)
    if indices_atipicos.size == 0 or indices_internos.size == 0:
        return float("nan")
    denominador = float(np.mean(larguras[indices_internos]))
    if denominador <= 0:
        return float("nan")
    return float(np.mean(larguras[indices_atipicos]) / denominador)


def avaliar_intervalos(
    y: ArrayLike,
    intervalos: ArrayLike,
    *,
    alpha: float,
    indices_atipicos: ArrayLike | None = None,
    indices_internos: ArrayLike | None = None,
) -> dict[str, float]:
    valores = np.asarray(intervalos, dtype=float)
    inferior, superior = valores[:, 0], valores[:, 1]
    resultado = {
        "amc": cobertura_marginal(y, inferior, superior),
        "smis": escore_medio_intervalo_escalonado(y, inferior, superior, alpha),
        "il": comprimento_medio_intervalo(inferior, superior),
        "pcorr": correlacao_cobertura_largura(y, inferior, superior),
    }
    if indices_atipicos is not None and indices_internos is not None:
        indices_atipicos = np.asarray(indices_atipicos, dtype=int)
        resultado["aco"] = cobertura_marginal(
            np.asarray(y)[indices_atipicos],
            inferior[indices_atipicos],
            superior[indices_atipicos],
        )
        resultado["ilr"] = razao_comprimento_atipicos_internos(
            inferior,
            superior,
            indices_atipicos,
            np.asarray(indices_internos, dtype=int),
        )
    return resultado
