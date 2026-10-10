"""EPICScore com MDN gaussiana e MC Dropout para scores de CQR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.special import ndtr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..conformal.scores import conformal_cutoff, quantile_scores
from ..models.base import QuantileModel, validate_interval_predictions


class _GaussianMDN(nn.Module):
    def __init__(self, input_size: int, hidden_layers: tuple[int, ...], components: int, dropout: float):
        super().__init__()
        modules: list[nn.Module] = []
        previous = input_size
        for width in hidden_layers:
            modules.extend(
                [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            )
            previous = width
        self.hidden = nn.Sequential(*modules)
        self.output = nn.Linear(previous, components * 3)
        self.components = components

    def forward(self, X: torch.Tensor):
        raw = self.output(self.hidden(X))
        logits, means, raw_scales = torch.chunk(raw, 3, dim=1)
        return torch.softmax(logits, dim=1), means, nn.functional.softplus(raw_scales) + 1e-5


def _mdn_loss(pi, means, scales, target):
    target = target[:, None]
    log_density = (
        -0.5 * ((target - means) / scales) ** 2
        - torch.log(scales)
        - 0.5 * np.log(2 * np.pi)
    )
    return -torch.logsumexp(torch.log(pi + 1e-10) + log_density, dim=1).mean()


@dataclass
class EPICScoreRegressor:
    base_model: QuantileModel
    alpha: float = 0.1
    base_model_is_fitted: bool = False
    hidden_layers: tuple[int, ...] = (64, 64)
    components: int = 5
    dropout: float = 0.5
    epochs: int = 2000
    patience: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 40
    n_mc_samples: int = 500
    random_state: int = 125
    device: str = "cpu"

    def fit(self, X: ArrayLike, y: ArrayLike) -> EPICScoreRegressor:
        if not self.base_model_is_fitted:
            self.base_model.fit(X, y)
            self.base_model_is_fitted = True
        return self

    def _fit_score_model(self, X: np.ndarray, scores: np.ndarray) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X, scores, test_size=0.2, random_state=self.random_state
        )
        self.X_scaler_ = StandardScaler().fit(X_train)
        self.score_scaler_ = StandardScaler().fit(y_train[:, None])
        X_train = self.X_scaler_.transform(X_train).astype(np.float32)
        X_val = self.X_scaler_.transform(X_val).astype(np.float32)
        y_train = self.score_scaler_.transform(y_train[:, None]).reshape(-1).astype(np.float32)
        y_val = self.score_scaler_.transform(y_val[:, None]).reshape(-1).astype(np.float32)

        self.score_model_ = _GaussianMDN(
            X.shape[1], self.hidden_layers, self.components, self.dropout
        ).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=min(self.batch_size, len(X_train)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
            drop_last=len(X_train) % min(self.batch_size, len(X_train)) == 1,
        )
        optimizer = torch.optim.Adam(self.score_model_.parameters(), lr=self.learning_rate)
        val_X = torch.from_numpy(X_val).to(self.device)
        val_y = torch.from_numpy(y_val).to(self.device)
        best_state = deepcopy(self.score_model_.state_dict())
        best_loss, stale = np.inf, 0
        for _ in range(self.epochs):
            self.score_model_.train()
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = _mdn_loss(*self.score_model_(batch_X), batch_y)
                loss.backward()
                optimizer.step()
            self.score_model_.eval()
            with torch.no_grad():
                loss = float(_mdn_loss(*self.score_model_(val_X), val_y).cpu())
            if loss < best_loss - 1e-8:
                best_loss, stale = loss, 0
                best_state = deepcopy(self.score_model_.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.score_model_.load_state_dict(best_state)
        self.score_model_.eval()

    def _mdn_draws(self, X: np.ndarray, n_mc: int | None = None) -> tuple[np.ndarray, ...]:
        X_scaled = self.X_scaler_.transform(X).astype(np.float32)
        tensor = torch.from_numpy(X_scaled).to(self.device)
        n_mc = self.n_mc_samples if n_mc is None else n_mc
        self.score_model_.eval()
        for module in self.score_model_.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        pi_draws, mean_draws, scale_draws = [], [], []
        with torch.no_grad():
            for _ in range(n_mc):
                pi, means, scales = self.score_model_(tensor)
                pi_draws.append(pi.cpu().numpy())
                mean_draws.append(means.cpu().numpy())
                scale_draws.append(scales.cpu().numpy())
        self.score_model_.eval()
        return np.asarray(pi_draws), np.asarray(mean_draws), np.asarray(scale_draws)

    def _cdf(self, X: np.ndarray, scores: np.ndarray) -> np.ndarray:
        pi, means, scales = self._mdn_draws(X)
        score_scaled = self.score_scaler_.transform(scores[:, None]).reshape(-1)
        component_cdf = ndtr((score_scaled[None, :, None] - means) / scales)
        return np.mean(np.sum(pi * component_cdf, axis=2), axis=0)

    def _score_quantile(self, X: np.ndarray, probability: float) -> np.ndarray:
        pi, means, scales = self._mdn_draws(X)
        rng = np.random.default_rng(self.random_state)
        n_mc, n_obs, n_components = pi.shape
        samples = np.empty((n_mc, n_obs), dtype=float)
        for draw in range(n_mc):
            uniforms = rng.random(n_obs)
            chosen = (uniforms[:, None] > np.cumsum(pi[draw], axis=1)).sum(axis=1)
            chosen = np.minimum(chosen, n_components - 1)
            rows = np.arange(n_obs)
            samples[draw] = rng.normal(means[draw, rows, chosen], scales[draw, rows, chosen])
        scaled_cutoff = np.quantile(samples, probability, axis=0)
        return self.score_scaler_.inverse_transform(scaled_cutoff[:, None]).reshape(-1)

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> EPICScoreRegressor:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        predictions = validate_interval_predictions(self.base_model.predict_quantiles(X_arr))
        scores = quantile_scores(y_arr, predictions[:, 0], predictions[:, 1])
        indices = np.arange(len(X_arr))
        train_idx, cutoff_idx = train_test_split(
            indices,
            test_size=0.3 if len(indices) < 2000 else min(1000 / len(indices), 0.5),
            random_state=self.random_state,
        )
        self._fit_score_model(X_arr[train_idx], scores[train_idx])
        transformed_scores = self._cdf(X_arr[cutoff_idx], scores[cutoff_idx])
        self.probability_cutoff_ = conformal_cutoff(transformed_scores, self.alpha)
        return self

    def predict(self, X: ArrayLike):
        if not hasattr(self, "probability_cutoff_"):
            raise RuntimeError("Execute calibrate antes de predict.")
        X_arr = np.asarray(X, dtype=float)
        base = validate_interval_predictions(self.base_model.predict_quantiles(X_arr))
        local_cutoff = self._score_quantile(X_arr, self.probability_cutoff_)
        return np.column_stack((base[:, 0] - local_cutoff, base[:, 1] + local_cutoff))

