"""EPICScore com rede de mistura gaussiana e MC Dropout para escores CQR."""

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

from ..conformal.escores import escores_quantilicos, limiar_conformal
from ..modelos.base import ModeloQuantilico, validar_predicoes_intervalares


class _RedeMisturaGaussiana(nn.Module):
    """Rede de densidade de mistura gaussiana usada pelo EPICScore."""

    def __init__(
        self,
        tamanho_entrada: int,
        camadas_ocultas: tuple[int, ...],
        componentes: int,
        dropout: float,
    ):
        super().__init__()
        modulos: list[nn.Module] = []
        largura_anterior = tamanho_entrada
        for largura in camadas_ocultas:
            modulos.extend(
                [
                    nn.Linear(largura_anterior, largura),
                    nn.BatchNorm1d(largura),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            largura_anterior = largura
        self.ocultas = nn.Sequential(*modulos)
        self.saida = nn.Linear(largura_anterior, componentes * 3)
        self.componentes = componentes

    def forward(self, x: torch.Tensor):
        """Executa a rede; o nome ``forward`` é obrigatório no PyTorch."""
        saida_bruta = self.saida(self.ocultas(x))
        logits, medias, escalas_brutas = torch.chunk(saida_bruta, 3, dim=1)
        pesos = torch.softmax(logits, dim=1)
        escalas = nn.functional.softplus(escalas_brutas) + 1e-5
        return pesos, medias, escalas


def _perda_mistura_gaussiana(pesos, medias, escalas, alvo):
    alvo = alvo[:, None]
    log_densidade = (
        -0.5 * ((alvo - medias) / escalas) ** 2
        - torch.log(escalas)
        - 0.5 * np.log(2 * np.pi)
    )
    return -torch.logsumexp(torch.log(pesos + 1e-10) + log_densidade, dim=1).mean()


@dataclass
class RegressorEPICScore:
    modelo_base: ModeloQuantilico
    alpha: float = 0.1
    modelo_base_ajustado: bool = False
    camadas_ocultas: tuple[int, ...] = (64, 64)
    componentes: int = 5
    dropout: float = 0.5
    epocas: int = 2000
    paciencia: int = 50
    taxa_aprendizado: float = 1e-3
    tamanho_lote: int = 40
    numero_amostras_mc: int = 500
    semente_aleatoria: int = 125
    dispositivo: str = "cpu"

    def ajustar(self, X: ArrayLike, y: ArrayLike) -> RegressorEPICScore:
        if not self.modelo_base_ajustado:
            self.modelo_base.ajustar(X, y)
            self.modelo_base_ajustado = True
        return self

    def _ajustar_modelo_escores(self, X: np.ndarray, escores: np.ndarray) -> None:
        np.random.seed(self.semente_aleatoria)
        torch.manual_seed(self.semente_aleatoria)
        x_treino, x_validacao, y_treino, y_validacao = train_test_split(
            X,
            escores,
            test_size=0.2,
            random_state=self.semente_aleatoria,
        )
        self.escalonador_x_ = StandardScaler().fit(x_treino)
        self.escalonador_escore_ = StandardScaler().fit(y_treino[:, None])
        x_treino = self.escalonador_x_.transform(x_treino).astype(np.float32)
        x_validacao = self.escalonador_x_.transform(x_validacao).astype(np.float32)
        y_treino = (
            self.escalonador_escore_.transform(y_treino[:, None])
            .reshape(-1)
            .astype(np.float32)
        )
        y_validacao = (
            self.escalonador_escore_.transform(y_validacao[:, None])
            .reshape(-1)
            .astype(np.float32)
        )

        self.modelo_escores_ = _RedeMisturaGaussiana(
            X.shape[1],
            self.camadas_ocultas,
            self.componentes,
            self.dropout,
        ).to(self.dispositivo)
        tamanho_lote_real = min(self.tamanho_lote, len(x_treino))
        carregador = DataLoader(
            TensorDataset(torch.from_numpy(x_treino), torch.from_numpy(y_treino)),
            batch_size=tamanho_lote_real,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.semente_aleatoria),
            drop_last=len(x_treino) % tamanho_lote_real == 1,
        )
        otimizador = torch.optim.Adam(
            self.modelo_escores_.parameters(),
            lr=self.taxa_aprendizado,
        )
        tensor_x_validacao = torch.from_numpy(x_validacao).to(self.dispositivo)
        tensor_y_validacao = torch.from_numpy(y_validacao).to(self.dispositivo)
        melhor_estado = deepcopy(self.modelo_escores_.state_dict())
        melhor_perda = np.inf
        epocas_sem_melhora = 0
        for _ in range(self.epocas):
            self.modelo_escores_.train()
            for lote_x, lote_y in carregador:
                lote_x = lote_x.to(self.dispositivo)
                lote_y = lote_y.to(self.dispositivo)
                otimizador.zero_grad()
                perda = _perda_mistura_gaussiana(
                    *self.modelo_escores_(lote_x),
                    lote_y,
                )
                perda.backward()
                otimizador.step()
            self.modelo_escores_.eval()
            with torch.no_grad():
                perda = float(
                    _perda_mistura_gaussiana(
                        *self.modelo_escores_(tensor_x_validacao),
                        tensor_y_validacao,
                    ).cpu()
                )
            if perda < melhor_perda - 1e-8:
                melhor_perda = perda
                epocas_sem_melhora = 0
                melhor_estado = deepcopy(self.modelo_escores_.state_dict())
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= self.paciencia:
                    break
        self.modelo_escores_.load_state_dict(melhor_estado)
        self.modelo_escores_.eval()

    def _amostras_rede(
        self,
        X: np.ndarray,
        numero_amostras_mc: int | None = None,
    ) -> tuple[np.ndarray, ...]:
        x_escalonado = self.escalonador_x_.transform(X).astype(np.float32)
        tensor = torch.from_numpy(x_escalonado).to(self.dispositivo)
        quantidade = (
            self.numero_amostras_mc
            if numero_amostras_mc is None
            else numero_amostras_mc
        )
        self.modelo_escores_.eval()
        for modulo in self.modelo_escores_.modules():
            if isinstance(modulo, nn.Dropout):
                modulo.train()
        amostras_pesos = []
        amostras_medias = []
        amostras_escalas = []
        with torch.no_grad():
            for _ in range(quantidade):
                pesos, medias, escalas = self.modelo_escores_(tensor)
                amostras_pesos.append(pesos.cpu().numpy())
                amostras_medias.append(medias.cpu().numpy())
                amostras_escalas.append(escalas.cpu().numpy())
        self.modelo_escores_.eval()
        return (
            np.asarray(amostras_pesos),
            np.asarray(amostras_medias),
            np.asarray(amostras_escalas),
        )

    def _funcao_distribuicao(self, X: np.ndarray, escores: np.ndarray) -> np.ndarray:
        pesos, medias, escalas = self._amostras_rede(X)
        escores_escalonados = self.escalonador_escore_.transform(escores[:, None]).reshape(-1)
        distribuicao_componentes = ndtr(
            (escores_escalonados[None, :, None] - medias) / escalas
        )
        return np.mean(np.sum(pesos * distribuicao_componentes, axis=2), axis=0)

    def _quantil_escore(self, X: np.ndarray, probabilidade: float) -> np.ndarray:
        pesos, medias, escalas = self._amostras_rede(X)
        gerador = np.random.default_rng(self.semente_aleatoria)
        numero_amostras, numero_observacoes, numero_componentes = pesos.shape
        amostras = np.empty((numero_amostras, numero_observacoes), dtype=float)
        for amostra in range(numero_amostras):
            uniformes = gerador.random(numero_observacoes)
            escolhidos = (
                uniformes[:, None] > np.cumsum(pesos[amostra], axis=1)
            ).sum(axis=1)
            escolhidos = np.minimum(escolhidos, numero_componentes - 1)
            linhas = np.arange(numero_observacoes)
            amostras[amostra] = gerador.normal(
                medias[amostra, linhas, escolhidos],
                escalas[amostra, linhas, escolhidos],
            )
        limiar_escalonado = np.quantile(amostras, probabilidade, axis=0)
        return self.escalonador_escore_.inverse_transform(
            limiar_escalonado[:, None]
        ).reshape(-1)

    def calibrar(self, X: ArrayLike, y: ArrayLike) -> RegressorEPICScore:
        matriz_x = np.asarray(X, dtype=float)
        vetor_y = np.asarray(y, dtype=float).reshape(-1)
        predicoes = validar_predicoes_intervalares(self.modelo_base.prever_quantis(matriz_x))
        escores = escores_quantilicos(vetor_y, predicoes[:, 0], predicoes[:, 1])
        indices = np.arange(len(matriz_x))
        indices_treino, indices_limiar = train_test_split(
            indices,
            test_size=0.3 if len(indices) < 2000 else min(1000 / len(indices), 0.5),
            random_state=self.semente_aleatoria,
        )
        self._ajustar_modelo_escores(matriz_x[indices_treino], escores[indices_treino])
        escores_transformados = self._funcao_distribuicao(
            matriz_x[indices_limiar],
            escores[indices_limiar],
        )
        self.limiar_probabilidade_ = limiar_conformal(escores_transformados, self.alpha)
        return self

    def prever(self, X: ArrayLike) -> np.ndarray:
        if not hasattr(self, "limiar_probabilidade_"):
            raise RuntimeError("Execute calibrar antes de prever.")
        matriz_x = np.asarray(X, dtype=float)
        base = validar_predicoes_intervalares(self.modelo_base.prever_quantis(matriz_x))
        limiar_local = self._quantil_escore(matriz_x, self.limiar_probabilidade_)
        return np.column_stack((base[:, 0] - limiar_local, base[:, 1] + limiar_local))
