from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from credal_cp_tgb.data import make_epistemic_gap_data, split_train_calibration_test
from credal_cp_tgb.models.quantile_neural_network import QuantileNeuralNetwork


def small_problem(seed: int = 125):
    data = make_epistemic_gap_data(700, random_state=seed)
    return split_train_calibration_test(
        data[["x"]].to_numpy(), data["y"].to_numpy(), random_state=seed
    )


def fitted_qnn(split, alpha: float = 0.1, seed: int = 125):
    return QuantileNeuralNetwork(
        1,
        alpha=alpha,
        hidden_layers=(64, 64),
        epochs=300,
        patience=25,
        batch_size=32,
        random_state=seed,
    ).fit(split.X_train, split.y_train)


def plot_intervals(X, y, intervals, title: str, output: str):
    order = np.argsort(X[:, 0])
    plt.figure(figsize=(8, 4.5))
    plt.scatter(X[:, 0], y, s=12, color="black", alpha=0.35)
    plt.fill_between(X[order, 0], intervals[order, 0], intervals[order, 1], alpha=0.3)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(output, dpi=200)

