import numpy as np

from credal_cp_tgb.credal.envelope import build_credal_envelope


def test_envelope_matches_columnwise_quantiles():
    lower = np.array([[0, 10], [1, 11], [2, 12], [3, 13]])
    upper = lower + 5
    gamma = np.array([0.5, 0.25])
    result = build_credal_envelope(lower, upper, gamma)
    assert result.shape == (2, 2)
    assert result[0, 0] == np.quantile(lower[:, 0], 0.25)
    assert result[1, 1] == np.quantile(upper[:, 1], 0.875)

