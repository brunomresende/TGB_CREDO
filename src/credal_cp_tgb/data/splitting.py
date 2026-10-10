"""Divisao reproduzivel 56/24/20 usada no artigo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplit:
    X_train: NDArray
    y_train: NDArray
    X_calibration: NDArray
    y_calibration: NDArray
    X_test: NDArray
    y_test: NDArray


def split_train_calibration_test(
    X: ArrayLike,
    y: ArrayLike,
    *,
    train_fraction: float = 0.56,
    calibration_fraction: float = 0.24,
    test_fraction: float = 0.20,
    random_state: int = 125,
) -> DataSplit:
    if not np.isclose(train_fraction + calibration_fraction + test_fraction, 1):
        raise ValueError("As tres proporcoes devem somar 1.")
    X_arr, y_arr = np.asarray(X), np.asarray(y).reshape(-1)
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X_arr, y_arr, test_size=test_fraction, random_state=random_state
    )
    relative_calibration = calibration_fraction / (train_fraction + calibration_fraction)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_cal,
        y_train_cal,
        test_size=relative_calibration,
        random_state=random_state,
    )
    return DataSplit(X_train, y_train, X_cal, y_cal, X_test, y_test)

