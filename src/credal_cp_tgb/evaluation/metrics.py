"""Metricas usadas no artigo e diagnosticos complementares."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def coverage_indicators(y: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> np.ndarray:
    y_arr = np.asarray(y).reshape(-1)
    return (np.asarray(lower).reshape(-1) <= y_arr) & (y_arr <= np.asarray(upper).reshape(-1))


def marginal_coverage(y: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """AMC: proporcao de respostas contidas nos intervalos."""
    return float(np.mean(coverage_indicators(y, lower, upper)))


def mean_interval_length(lower: ArrayLike, upper: ArrayLike) -> float:
    """IL: comprimento medio dos intervalos; menor e melhor sob cobertura valida."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def scaled_mean_interval_score(
    y: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> float:
    """SMIS/interval score medio; menor e melhor."""
    y_arr = np.asarray(y).reshape(-1)
    lower_arr = np.asarray(lower).reshape(-1)
    upper_arr = np.asarray(upper).reshape(-1)
    scores = (
        upper_arr
        - lower_arr
        + (2 / alpha) * (lower_arr - y_arr) * (y_arr < lower_arr)
        + (2 / alpha) * (y_arr - upper_arr) * (y_arr > upper_arr)
    )
    return float(np.mean(scores))


def coverage_width_correlation(y: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Correlacao absoluta entre indicador de cobertura e largura."""
    covered = coverage_indicators(y, lower, upper).astype(float)
    widths = np.asarray(upper).reshape(-1) - np.asarray(lower).reshape(-1)
    if np.std(covered) == 0 or np.std(widths) == 0:
        return float("nan")
    return float(abs(np.corrcoef(covered, widths)[0, 1]))


def outlier_inlier_length_ratio(
    lower: ArrayLike,
    upper: ArrayLike,
    outlier_indices: ArrayLike,
    inlier_indices: ArrayLike,
) -> float:
    """ILR: largura media nos outliers dividida pela largura nos inliers centrais."""
    widths = np.asarray(upper).reshape(-1) - np.asarray(lower).reshape(-1)
    out_idx = np.asarray(outlier_indices, dtype=int)
    in_idx = np.asarray(inlier_indices, dtype=int)
    if out_idx.size == 0 or in_idx.size == 0:
        return float("nan")
    denominator = float(np.mean(widths[in_idx]))
    return float(np.mean(widths[out_idx]) / denominator) if denominator > 0 else float("nan")


def evaluate_intervals(
    y: ArrayLike,
    intervals: ArrayLike,
    *,
    alpha: float,
    outlier_indices: ArrayLike | None = None,
    inlier_indices: ArrayLike | None = None,
) -> dict[str, float]:
    values = np.asarray(intervals, dtype=float)
    lower, upper = values[:, 0], values[:, 1]
    result = {
        "amc": marginal_coverage(y, lower, upper),
        "smis": scaled_mean_interval_score(y, lower, upper, alpha),
        "il": mean_interval_length(lower, upper),
        "pcorr": coverage_width_correlation(y, lower, upper),
    }
    if outlier_indices is not None and inlier_indices is not None:
        out_idx = np.asarray(outlier_indices, dtype=int)
        result["aco"] = marginal_coverage(
            np.asarray(y)[out_idx], lower[out_idx], upper[out_idx]
        )
        result["ilr"] = outlier_inlier_length_ratio(
            lower, upper, out_idx, np.asarray(inlier_indices, dtype=int)
        )
    return result

