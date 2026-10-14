"""QNN compartilhável entre CQR e CREDO.

CQR usa :meth:`prever_quantis` em modo determinístico. CREDO usa
:meth:`amostrar_quantis` com apenas o dropout ativo durante a inferência.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import validar_predicoes_intervalares


class RedeQuantilica(nn.Module):
    """Rede interna com duas saídas quantílicas livres."""

    def __init__(
        self,
        tamanho_entrada: int,
        camadas_ocultas: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        blocos: list[nn.Module] = []
        largura_anterior = tamanho_entrada
        for largura in camadas_ocultas:
            blocos.extend(
                [
                    nn.Linear(largura_anterior, largura),
                    nn.BatchNorm1d(largura),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            largura_anterior = largura
        self.ocultas = nn.Sequential(*blocos)
        self.saida = nn.Linear(largura_anterior, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a rede; o nome ``forward`` é obrigatório no PyTorch."""
        # O eventual cruzamento é corrigido por ordenação somente na inferência.
        return self.saida(self.ocultas(x))


def perda_pinball(
    predicoes: torch.Tensor,
    alvos: torch.Tensor,
    quantis: tuple[float, float],
) -> torch.Tensor:
    """Calcula a soma das perdas pinball dos dois quantis."""
    erros = alvos[:, None] - predicoes
    niveis = torch.as_tensor(quantis, dtype=predicoes.dtype, device=predicoes.device)
    perdas = torch.maximum(niveis * erros, (niveis - 1) * erros)
    # A implementação original soma a média de cada quantil.
    return perdas.mean(dim=0).sum()


@dataclass
class RedeNeuralQuantilica:
    tamanho_entrada: int
    alpha: float = 0.1
    camadas_ocultas: tuple[int, ...] = (128, 128, 64)
    dropout: float = 0.3
    epocas: int = 2000
    taxa_aprendizado: float = 1e-3
    decaimento_pesos: float = 1e-6
    passo_agendador: int = 10
    gamma_agendador: float = 0.99
    tamanho_lote: int = 32
    paciencia: int = 50
    fracao_validacao: float = 0.2
    semente_aleatoria: int = 1250
    semente_validacao: int = 0
    escalonar_x: bool = True
    escalonar_y: bool = True
    dispositivo: str = "cpu"

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha deve estar entre 0 e 1.")
        # Inclui a inicializacao dos pesos na reproducibilidade do modelo.
        self._fixar_semente()
        self.niveis_quantis = (self.alpha / 2, 1 - self.alpha / 2)
        self.rede = RedeQuantilica(
            self.tamanho_entrada,
            self.camadas_ocultas,
            self.dropout,
        )

    def _fixar_semente(self, semente: int | None = None) -> None:
        semente = self.semente_aleatoria if semente is None else semente
        np.random.seed(semente)
        torch.manual_seed(semente)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(semente)

    def ajustar(self, X: ArrayLike, y: ArrayLike) -> RedeNeuralQuantilica:
        """Treina a QNN com dropout ativo e parada antecipada."""
        self._fixar_semente()
        matriz_x = np.asarray(X, dtype=np.float32)
        vetor_y = np.asarray(y, dtype=np.float32).reshape(-1)
        x_treino, x_validacao, y_treino, y_validacao = train_test_split(
            matriz_x,
            vetor_y,
            test_size=self.fracao_validacao,
            random_state=self.semente_validacao,
        )

        self.escalonador_x_ = StandardScaler().fit(x_treino) if self.escalonar_x else None
        self.escalonador_y_ = StandardScaler().fit(y_treino[:, None]) if self.escalonar_y else None
        if self.escalonador_x_ is not None:
            x_treino = self.escalonador_x_.transform(x_treino)
            x_validacao = self.escalonador_x_.transform(x_validacao)
        if self.escalonador_y_ is not None:
            y_treino = self.escalonador_y_.transform(y_treino[:, None]).reshape(-1)
            y_validacao = self.escalonador_y_.transform(y_validacao[:, None]).reshape(-1)

        dados_treino = TensorDataset(
            torch.as_tensor(x_treino, dtype=torch.float32),
            torch.as_tensor(y_treino, dtype=torch.float32),
        )
        tensor_x_validacao = torch.as_tensor(
            x_validacao,
            dtype=torch.float32,
            device=self.dispositivo,
        )
        tensor_y_validacao = torch.as_tensor(
            y_validacao,
            dtype=torch.float32,
            device=self.dispositivo,
        )
        gerador = torch.Generator().manual_seed(self.semente_aleatoria)
        tamanho_lote_real = min(self.tamanho_lote, len(dados_treino))
        carregador = DataLoader(
            dados_treino,
            batch_size=tamanho_lote_real,
            shuffle=True,
            generator=gerador,
            drop_last=len(dados_treino) % tamanho_lote_real == 1,
        )

        self.rede.to(self.dispositivo)
        otimizador = torch.optim.Adam(
            self.rede.parameters(),
            lr=self.taxa_aprendizado,
            weight_decay=self.decaimento_pesos,
        )
        agendador = torch.optim.lr_scheduler.StepLR(
            otimizador,
            step_size=self.passo_agendador,
            gamma=self.gamma_agendador,
        )
        melhor_estado = deepcopy(self.rede.state_dict())
        melhor_perda = np.inf
        epocas_sem_melhora = 0
        self.historico_ = []

        for _ in range(self.epocas):
            self.rede.train()
            for lote_x, lote_y in carregador:
                lote_x = lote_x.to(self.dispositivo)
                lote_y = lote_y.to(self.dispositivo)
                otimizador.zero_grad()
                perda = perda_pinball(self.rede(lote_x), lote_y, self.niveis_quantis)
                perda.backward()
                torch.nn.utils.clip_grad_norm_(self.rede.parameters(), max_norm=1.0)
                otimizador.step()

            self.rede.eval()
            with torch.no_grad():
                perda_validacao = float(
                    perda_pinball(
                        self.rede(tensor_x_validacao),
                        tensor_y_validacao,
                        self.niveis_quantis,
                    ).cpu()
                )
            self.historico_.append(perda_validacao)
            agendador.step()
            if perda_validacao < melhor_perda - 1e-8:
                melhor_perda = perda_validacao
                melhor_estado = deepcopy(self.rede.state_dict())
                epocas_sem_melhora = 0
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= self.paciencia:
                    break

        self.rede.load_state_dict(melhor_estado)
        self.rede.eval()
        self.ajustada_ = True
        return self

    def _preparar_x(self, X: ArrayLike) -> torch.Tensor:
        if not hasattr(self, "ajustada_"):
            raise RuntimeError("A QNN ainda não foi treinada.")
        matriz_x = np.asarray(X, dtype=np.float32)
        if self.escalonador_x_ is not None:
            matriz_x = self.escalonador_x_.transform(matriz_x)
        return torch.as_tensor(matriz_x, dtype=torch.float32, device=self.dispositivo)

    def _desfazer_escala_y(self, valores: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.escalonador_y_ is None:
            return valores
        formato = valores.shape
        restaurados = self.escalonador_y_.inverse_transform(valores.reshape(-1, 1))
        return restaurados.reshape(formato)

    def prever_quantis(self, X: ArrayLike) -> NDArray[np.float64]:
        """Faz a predição determinística usada por CQR e CQR-r."""
        tensor = self._preparar_x(X)
        self.rede.eval()
        with torch.no_grad():
            predicoes = self.rede(tensor).cpu().numpy().astype(float)
        predicoes.sort(axis=1)
        return validar_predicoes_intervalares(self._desfazer_escala_y(predicoes))

    def _ativar_dropout_monte_carlo(self) -> None:
        # Congela tudo, especialmente BatchNorm, e depois ativa somente Dropout.
        self.rede.eval()
        for modulo in self.rede.modules():
            if isinstance(modulo, nn.Dropout):
                modulo.train()

    def amostrar_quantis(
        self,
        X: ArrayLike,
        *,
        numero_amostras: int = 1000,
        semente_aleatoria: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Aproxima a posteriori dos quantis usando MC Dropout."""
        if numero_amostras < 2:
            raise ValueError("numero_amostras deve ser pelo menos 2.")
        self._fixar_semente(semente_aleatoria)
        tensor = self._preparar_x(X)
        self._ativar_dropout_monte_carlo()
        amostras = []
        with torch.no_grad():
            for _ in range(numero_amostras):
                amostras.append(self.rede(tensor).cpu().numpy())
        self.rede.eval()
        amostras = self._desfazer_escala_y(np.asarray(amostras, dtype=float))
        amostras.sort(axis=2)
        return amostras[:, :, 0], amostras[:, :, 1]
