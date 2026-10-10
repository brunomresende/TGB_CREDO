"""BART heteroscedastico para os experimentos sinteticos do CREDO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


@dataclass
class HeteroscedasticBART:
    alpha: float = 0.1
    n_trees: int = 100
    draws: int = 1000
    chains: int = 4
    cores: int = 4
    tree_alpha: float = 0.985
    tree_beta: float = 2.0
    normalize_y: bool = True
    progressbar: bool = False
    random_state: int = 1250

    def fit(self, X: ArrayLike, y: ArrayLike) -> "HeteroscedasticBART":
        try:
            import pymc as pm
            import pymc_bart as pmb
        except ImportError as error:
            raise ImportError("Instale o extra BART: pip install -e '.[bart]'") from error

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        self.y_scaler_ = StandardScaler().fit(y_arr[:, None]) if self.normalize_y else None
        y_fit = (
            self.y_scaler_.transform(y_arr[:, None]).reshape(-1)
            if self.y_scaler_ is not None
            else y_arr
        )
        with pm.Model() as self.model_:
            self.X_data_ = pm.Data("X", X_arr)
            latent = pmb.BART(
                "latent",
                self.X_data_,
                y_fit,
                m=self.n_trees,
                shape=(2, len(y_fit)),
                alpha=self.tree_alpha,
                beta=self.tree_beta,
            )
            pm.Normal(
                "y",
                mu=latent[0],
                sigma=pm.math.exp(latent[1]),
                observed=y_fit,
                shape=self.X_data_.shape[0],
            )
            self.trace_ = pm.sample(
                draws=self.draws,
                chains=self.chains,
                cores=self.cores,
                random_seed=self.random_state,
                progressbar=self.progressbar,
            )
        return self

    def sample_quantiles(
        self,
        X: ArrayLike,
        *,
        n_samples: int = 1000,
        random_state: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        try:
            import arviz as az
            import pymc as pm
        except ImportError as error:
            raise ImportError("Instale o extra BART: pip install -e '.[bart]'") from error
        if not hasattr(self, "trace_"):
            raise RuntimeError("O BART ainda nao foi treinado.")

        X_arr = np.asarray(X, dtype=float)
        with self.model_:
            self.X_data_.set_value(X_arr)
            posterior = pm.sample_posterior_predictive(
                self.trace_,
                var_names=["latent"],
                predictions=True,
                random_seed=self.random_state if random_state is None else random_state,
                progressbar=self.progressbar,
            )
        latent = az.extract(
            posterior, group="predictions", var_names=["latent"]
        ).to_numpy()
        mu, sigma = latent[0].T, np.exp(latent[1].T)
        if mu.shape[0] > n_samples:
            rng = np.random.default_rng(random_state)
            selected = rng.choice(mu.shape[0], n_samples, replace=False)
            mu, sigma = mu[selected], sigma[selected]
        z_lower, z_upper = norm.ppf([self.alpha / 2, 1 - self.alpha / 2])
        lower, upper = mu + z_lower * sigma, mu + z_upper * sigma
        if self.y_scaler_ is not None:
            shape = lower.shape
            lower = self.y_scaler_.inverse_transform(lower.reshape(-1, 1)).reshape(shape)
            upper = self.y_scaler_.inverse_transform(upper.reshape(-1, 1)).reshape(shape)
        return lower, upper

    def predict_quantiles(self, X: ArrayLike) -> NDArray[np.float64]:
        lower, upper = self.sample_quantiles(X, n_samples=min(self.draws, 1000))
        return np.column_stack((np.median(lower, axis=0), np.median(upper, axis=0)))

