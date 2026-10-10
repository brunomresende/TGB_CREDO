"""Diagnostico externo de outliers com t-SNE e LOF, conforme o artigo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class OutlierGroups:
    outliers: NDArray[np.int64]
    ilr_inliers: NDArray[np.int64]
    decomposition_inliers: NDArray[np.int64]
    lof_scores: NDArray[np.float64]


def detect_outlier_groups(
    X: ArrayLike,
    *,
    n_neighbors: int = 15,
    contamination: float = 0.05,
    ilr_inlier_fraction: float = 0.20,
    decomposition_inlier_fraction: float = 0.10,
    random_state: int = 120,
) -> OutlierGroups:
    X_arr = np.asarray(X, dtype=float)
    if len(X_arr) < 4:
        raise ValueError("Sao necessarias ao menos quatro observacoes para t-SNE/LOF.")
    perplexity = min(30.0, max(2.0, (len(X_arr) - 1) / 3))
    embedded = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        # PCA nao pode inicializar duas dimensoes quando X possui uma unica feature.
        init="pca" if X_arr.shape[1] >= 2 else "random",
        learning_rate="auto",
    ).fit_transform(X_arr)
    embedded = StandardScaler().fit_transform(embedded)
    lof = LocalOutlierFactor(
        n_neighbors=min(n_neighbors, len(X_arr) - 1), contamination=contamination
    )
    labels = lof.fit_predict(embedded)
    outliers = np.flatnonzero(labels == -1)
    inliers = np.flatnonzero(labels == 1)
    # negative_outlier_factor_: valores menos negativos sao mais centrais.
    ordered = inliers[np.argsort(lof.negative_outlier_factor_[inliers])[::-1]]
    n_ilr = max(1, int(len(inliers) * ilr_inlier_fraction))
    n_decomp = max(1, int(len(inliers) * decomposition_inlier_fraction))
    return OutlierGroups(
        outliers=outliers,
        ilr_inliers=ordered[:n_ilr],
        decomposition_inliers=ordered[:n_decomp],
        lof_scores=lof.negative_outlier_factor_.astype(float),
    )
