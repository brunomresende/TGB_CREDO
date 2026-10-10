"""Cenarios pequenos para visualizar incertezas aleatoria e epistemica."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_epistemic_gap_data(
    n_samples: int = 1500,
    *,
    middle_probability: float = 0.01,
    noise_std: float = 0.1,
    random_state: int = 125,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_middle = max(1, int(n_samples * middle_probability))
    n_left = (n_samples - n_middle) // 2
    n_right = n_samples - n_middle - n_left
    x = np.concatenate(
        (
            rng.uniform(-1.0, -0.2, n_left),
            rng.uniform(-0.2, 0.2, n_middle),
            rng.uniform(0.2, 1.0, n_right),
        )
    )
    mean = np.where(
        x < -0.2,
        -0.5 + 1.5 * (x + 0.8) - (x + 0.8) ** 2,
        np.where(x <= 0.2, 0.2 * np.sin(8 * x), 0.7 + 0.8 * (x - 0.5) ** 2),
    )
    y = mean + rng.normal(0, noise_std, len(x))
    permutation = rng.permutation(len(x))
    return pd.DataFrame({"x": x[permutation], "y": y[permutation], "mean": mean[permutation]})

