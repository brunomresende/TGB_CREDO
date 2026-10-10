"""Agregacao das repeticoes sem esconder os resultados brutos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    identifier_columns = {"dataset", "repeat", "seed", "method", "backend"}
    metric_columns = [
        column
        for column in results.select_dtypes(include=[np.number]).columns
        if column not in identifier_columns
    ]
    rows = []
    for (dataset, method), group in results.groupby(["dataset", "method"], sort=False):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": sd,
                    "se": sd / np.sqrt(len(values)),
                    "ci95_half_width": 1.96 * sd / np.sqrt(len(values)),
                    "n_repetitions": len(values),
                }
            )
    return pd.DataFrame(rows)

