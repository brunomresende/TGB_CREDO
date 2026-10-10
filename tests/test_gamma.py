import numpy as np

from credal_cp_tgb.credal.gamma import AdaptiveGamma, FixedGamma


def test_fixed_gamma_returns_constant():
    model = FixedGamma(0.2).fit(np.zeros((5, 2)))
    np.testing.assert_allclose(model.compute(np.ones((3, 2))), 0.2)


def test_adaptive_gamma_is_bounded_and_smaller_far_away():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.1, size=(100, 1))
    model = AdaptiveGamma(gamma_min=0.1, gamma_max=0.75, n_neighbors=10).fit(X)
    gamma = model.compute(np.array([[0.0], [10.0]]))
    assert np.all((0.1 <= gamma) & (gamma <= 0.75))
    assert gamma[1] < gamma[0]

