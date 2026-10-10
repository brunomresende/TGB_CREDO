import numpy as np

from credal_cp_tgb.conformal.scores import (
    conformal_cutoff,
    quantile_scores,
    relative_quantile_scores,
)


def test_quantile_scores_allow_negative_values():
    scores = quantile_scores([0.5, 2.0, 4.0], [0.0, 0.0, 0.0], [1.0, 1.0, 3.0])
    np.testing.assert_allclose(scores, [-0.5, 1.0, 1.0])


def test_relative_scores_use_interval_width():
    scores = relative_quantile_scores([3.0, 3.0], [0.0, 0.0], [2.0, 1.0])
    np.testing.assert_allclose(scores, [0.5, 2.0])


def test_cutoff_is_finite_sample_order_statistic():
    scores = np.arange(1, 11)
    # ceil(11 * .8) = 9
    assert conformal_cutoff(scores, 0.2) == 9

