import numpy as np

from credal_cp_tgb.evaluation.metrics import (
    marginal_coverage,
    mean_interval_length,
    outlier_inlier_length_ratio,
    scaled_mean_interval_score,
)


def test_basic_interval_metrics():
    y = np.array([0.0, 2.0])
    lower = np.array([-1.0, 0.0])
    upper = np.array([1.0, 1.0])
    assert marginal_coverage(y, lower, upper) == 0.5
    assert mean_interval_length(lower, upper) == 1.5
    assert scaled_mean_interval_score(y, lower, upper, 0.1) == 11.5


def test_ilr():
    lower = np.zeros(4)
    upper = np.array([1.0, 1.0, 2.0, 2.0])
    assert outlier_inlier_length_ratio(lower, upper, [2, 3], [0, 1]) == 2.0

