"""QNN compartilhavel entre CQR e CREDO.

CQR usa ``predict_quantiles`` em modo deterministico. CREDO usa
``sample_quantiles`` com apenas Dropout ativo durante a inferencia.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import validate_interval_predictions


class QuantileNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_layers: Sequence[int], dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        previous = input_size
        for width in hidden_layers:
            blocks.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        self.hidden = nn.Sequential(*blocks)
        self.output = nn.Linear(previous, 2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Duas saidas livres, como no codigo de referencia. O eventual cruzamento
        # e corrigido por ordenacao somente durante a inferencia.
        return self.output(self.hidden(X))


def pinball_loss(
    predictions: torch.Tensor, targets: torch.Tensor, quantiles: tuple[float, float]
) -> torch.Tensor:
    errors = targets[:, None] - predictions
    levels = torch.as_tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
    losses = torch.maximum(levels * errors, (levels - 1) * errors)
    # O original soma a media de cada quantil (em vez de tirar a media geral).
    return losses.mean(dim=0).sum()


@dataclass
class QuantileNeuralNetwork:
    input_size: int
    alpha: float = 0.1
    hidden_layers: tuple[int, ...] = (128, 128, 64)
    dropout: float = 0.3
    epochs: int = 2000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    scheduler_step: int = 10
    scheduler_gamma: float = 0.99
    batch_size: int = 32
    patience: int = 50
    validation_fraction: float = 0.2
    random_state: int = 1250
    validation_random_state: int = 0
    scale_X: bool = True
    scale_y: bool = True
    device: str = "cpu"

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("alpha deve estar entre 0 e 1.")
        self.quantile_levels = (self.alpha / 2, 1 - self.alpha / 2)
        self.network = QuantileNetwork(self.input_size, self.hidden_layers, self.dropout)

    def _seed(self, seed: int | None = None) -> None:
        seed = self.random_state if seed is None else seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "QuantileNeuralNetwork":
        self._seed()
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_arr,
            y_arr,
            test_size=self.validation_fraction,
            random_state=self.validation_random_state,
        )

        self.X_scaler_ = StandardScaler().fit(X_train) if self.scale_X else None
        self.y_scaler_ = StandardScaler().fit(y_train[:, None]) if self.scale_y else None
        if self.X_scaler_ is not None:
            X_train = self.X_scaler_.transform(X_train)
            X_val = self.X_scaler_.transform(X_val)
        if self.y_scaler_ is not None:
            y_train = self.y_scaler_.transform(y_train[:, None]).reshape(-1)
            y_val = self.y_scaler_.transform(y_val[:, None]).reshape(-1)

        train_data = TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        )
        val_X = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
        val_y = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            train_data,
            batch_size=min(self.batch_size, len(train_data)),
            shuffle=True,
            generator=generator,
            drop_last=len(train_data) % min(self.batch_size, len(train_data)) == 1,
        )

        self.network.to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma
        )
        best_state = deepcopy(self.network.state_dict())
        best_loss = np.inf
        stale_epochs = 0
        self.history_ = []

        for _ in range(self.epochs):
            self.network.train()
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = pinball_loss(self.network(batch_X), batch_y, self.quantile_levels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()

            self.network.eval()
            with torch.no_grad():
                val_loss = float(
                    pinball_loss(self.network(val_X), val_y, self.quantile_levels).cpu()
                )
            self.history_.append(val_loss)
            scheduler.step()
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = deepcopy(self.network.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.network.load_state_dict(best_state)
        self.network.eval()
        self.is_fitted_ = True
        return self

    def _prepare_X(self, X: ArrayLike) -> torch.Tensor:
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("A QNN ainda nao foi treinada.")
        X_arr = np.asarray(X, dtype=np.float32)
        if self.X_scaler_ is not None:
            X_arr = self.X_scaler_.transform(X_arr)
        return torch.as_tensor(X_arr, dtype=torch.float32, device=self.device)

    def _inverse_y(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.y_scaler_ is None:
            return values
        shape = values.shape
        restored = self.y_scaler_.inverse_transform(values.reshape(-1, 1))
        return restored.reshape(shape)

    def predict_quantiles(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predicao deterministica usada por CQR e CQR-r."""
        tensor = self._prepare_X(X)
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(tensor).cpu().numpy().astype(float)
        predictions.sort(axis=1)
        return validate_interval_predictions(self._inverse_y(predictions))

    def _enable_mc_dropout(self) -> None:
        # Primeiro congela tudo (em especial BatchNorm), depois ativa so Dropout.
        self.network.eval()
        for module in self.network.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def sample_quantiles(
        self,
        X: ArrayLike,
        *,
        n_samples: int = 1000,
        random_state: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Aproxima o posterior dos quantis usando MC Dropout."""
        if n_samples < 2:
            raise ValueError("n_samples deve ser pelo menos 2.")
        self._seed(random_state)
        tensor = self._prepare_X(X)
        self._enable_mc_dropout()
        draws = []
        with torch.no_grad():
            for _ in range(n_samples):
                draws.append(self.network(tensor).cpu().numpy())
        self.network.eval()
        samples = self._inverse_y(np.asarray(draws, dtype=float))
        samples.sort(axis=2)
        return samples[:, :, 0], samples[:, :, 1]
