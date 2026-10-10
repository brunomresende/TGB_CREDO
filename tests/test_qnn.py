import numpy as np

from credal_cp_tgb.models.quantile_neural_network import QuantileNeuralNetwork


def test_deterministic_prediction_and_mc_dropout_modes():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.1, 80)).astype(np.float32)
    model = QuantileNeuralNetwork(
        2,
        hidden_layers=(8,),
        epochs=3,
        patience=2,
        batch_size=16,
        random_state=2,
    ).fit(X, y)
    first = model.predict_quantiles(X[:5])
    second = model.predict_quantiles(X[:5])
    np.testing.assert_allclose(first, second)

    batch_norm = next(module for module in model.network.modules() if module.__class__.__name__ == "BatchNorm1d")
    running_mean = batch_norm.running_mean.detach().clone()
    lower, upper = model.sample_quantiles(X[:5], n_samples=5, random_state=3)
    assert lower.shape == upper.shape == (5, 5)
    assert np.any(np.std(lower, axis=0) > 0)
    np.testing.assert_allclose(batch_norm.running_mean.detach().numpy(), running_mean.numpy())

