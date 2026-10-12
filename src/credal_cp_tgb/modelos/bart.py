"""BART heteroscedástico para os experimentos sintéticos do CREDO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


@dataclass
class BARTHeteroscedastico:
    alpha: float = 0.1
    numero_arvores: int = 100
    amostras: int = 1000
    cadeias: int = 4
    nucleos: int = 4
    alfa_arvore: float = 0.985
    beta_arvore: float = 2.0
    normalizar_y: bool = True
    barra_progresso: bool = False
    semente_aleatoria: int = 1250

    def ajustar(self, X: ArrayLike, y: ArrayLike) -> BARTHeteroscedastico:
        try:
            import pymc as pm
            import pymc_bart as pmb
        except ImportError as error:
            raise ImportError("Instale o extra BART: pip install -e '.[bart]'") from error

        matriz_x = np.asarray(X, dtype=float)
        vetor_y = np.asarray(y, dtype=float).reshape(-1)
        self.escalonador_y_ = (
            StandardScaler().fit(vetor_y[:, None]) if self.normalizar_y else None
        )
        y_ajuste = (
            self.escalonador_y_.transform(vetor_y[:, None]).reshape(-1)
            if self.escalonador_y_ is not None
            else vetor_y
        )
        with pm.Model() as self.modelo_:
            self.dados_x_ = pm.Data("X", matriz_x)
            latente = pmb.BART(
                "latente",
                self.dados_x_,
                y_ajuste,
                m=self.numero_arvores,
                shape=(2, len(y_ajuste)),
                alpha=self.alfa_arvore,
                beta=self.beta_arvore,
            )
            pm.Normal(
                "y",
                mu=latente[0],
                sigma=pm.math.exp(latente[1]),
                observed=y_ajuste,
                shape=self.dados_x_.shape[0],
            )
            self.traco_ = pm.sample(
                draws=self.amostras,
                chains=self.cadeias,
                cores=self.nucleos,
                random_seed=self.semente_aleatoria,
                progressbar=self.barra_progresso,
            )
        return self

    def amostrar_quantis(
        self,
        X: ArrayLike,
        *,
        numero_amostras: int = 1000,
        semente_aleatoria: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        try:
            import arviz as az
            import pymc as pm
        except ImportError as error:
            raise ImportError("Instale o extra BART: pip install -e '.[bart]'") from error
        if not hasattr(self, "traco_"):
            raise RuntimeError("O BART ainda não foi treinado.")

        matriz_x = np.asarray(X, dtype=float)
        with self.modelo_:
            self.dados_x_.set_value(matriz_x)
            posterior = pm.sample_posterior_predictive(
                self.traco_,
                var_names=["latente"],
                predictions=True,
                random_seed=(
                    self.semente_aleatoria
                    if semente_aleatoria is None
                    else semente_aleatoria
                ),
                progressbar=self.barra_progresso,
            )
        latente = az.extract(
            posterior, group="predictions", var_names=["latente"]
        ).to_numpy()
        media, desvio = latente[0].T, np.exp(latente[1].T)
        if media.shape[0] > numero_amostras:
            gerador = np.random.default_rng(semente_aleatoria)
            selecionadas = gerador.choice(media.shape[0], numero_amostras, replace=False)
            media, desvio = media[selecionadas], desvio[selecionadas]
        z_inferior, z_superior = norm.ppf([self.alpha / 2, 1 - self.alpha / 2])
        inferior = media + z_inferior * desvio
        superior = media + z_superior * desvio
        if self.escalonador_y_ is not None:
            formato = inferior.shape
            inferior = self.escalonador_y_.inverse_transform(inferior.reshape(-1, 1)).reshape(
                formato
            )
            superior = self.escalonador_y_.inverse_transform(superior.reshape(-1, 1)).reshape(
                formato
            )
        return inferior, superior

    def prever_quantis(self, X: ArrayLike) -> NDArray[np.float64]:
        inferior, superior = self.amostrar_quantis(
            X, numero_amostras=min(self.amostras, 1000)
        )
        return np.column_stack(
            (np.median(inferior, axis=0), np.median(superior, axis=0))
        )
