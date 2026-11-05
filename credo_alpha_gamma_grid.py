"""Gera uma grade CREDO-BART para diferentes valores de alpha e gamma.

Execute na raiz do projeto:
    uv run --extra bart python credo_alpha_gamma_grid.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from credal_cp_tgb.credal.gamma import GammaFixo
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.bart import BARTHeteroscedastico

SEMENTE = 125
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados" / "figuras"


def gerar_dados_com_lacuna(
    n: int, desvio_ruido: float = 0.1, proporcao_centro: float = 0.01
) -> pd.DataFrame:
    """Cria dados unidimensionais com poucas observacoes no centro."""
    rng = np.random.default_rng(SEMENTE)
    n_centro = max(1, int(n * proporcao_centro))
    n_lado = (n - n_centro) // 2
    n_direita = n - n_centro - n_lado

    x = np.concatenate(
        (
            rng.uniform(-1.0, -0.2, size=n_lado),
            rng.uniform(-0.2, 0.2, size=n_centro),
            rng.uniform(0.2, 1.0, size=n_direita),
        )
    )
    media = np.zeros_like(x)
    esquerda = x < -0.2
    centro = (x >= -0.2) & (x <= 0.2)
    direita = x > 0.2
    media[esquerda] = -0.5 + 1.5 * (x[esquerda] + 0.8) - (x[esquerda] + 0.8) ** 2
    media[centro] = 0.2 * np.sin(8 * x[centro])
    media[direita] = 0.7 + 0.8 * (x[direita] - 0.5) ** 2 - 0.3 * (x[direita] - 0.5) ** 3
    dados = pd.DataFrame({"x": x, "y": media + rng.normal(0, desvio_ruido, len(x))})
    return dados.sample(frac=1.0, random_state=0).reset_index(drop=True)


def ajustar_credo(
    x_treino: np.ndarray,
    y_treino: np.ndarray,
    x_calibracao: np.ndarray,
    y_calibracao: np.ndarray,
    alpha: float,
    gamma: float,
    n_mcmc: int = 1000,
) -> RegressorCREDO:
    """Ajusta e calibra uma configuracao fixa do CREDO-BART."""
    modelo_base = BARTHeteroscedastico(
        alpha=alpha,
        amostras=n_mcmc,
        cadeias=4,
        nucleos=4,
        alfa_arvore=0.985,
        barra_progresso=True,
        semente_aleatoria=SEMENTE,
    )
    modelo = RegressorCREDO(
        modelo_base=modelo_base,
        estrategia_gamma=GammaFixo(gamma),
        alpha=alpha,
        numero_amostras_posteriores=n_mcmc,
        semente_aleatoria=SEMENTE,
    )
    return modelo.ajustar(x_treino, y_treino).calibrar(x_calibracao, y_calibracao)


def plotar_grade(
    x_treino: np.ndarray,
    y_treino: np.ndarray,
    x_calibracao: np.ndarray,
    y_calibracao: np.ndarray,
    x_teste: np.ndarray,
    y_teste: np.ndarray,
    alphas: list[float],
    gammas: list[float],
    n_mcmc: int = 1000,
) -> Path:
    """Ajusta a grade de hiperparametros e salva a figura resultante."""
    grade = np.linspace(-1.15, 1.15, 500, dtype=np.float32).reshape(-1, 1)
    figura, eixos = plt.subplots(
        len(gammas),
        len(alphas),
        figsize=(5 * len(alphas), 3.8 * len(gammas)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for linha, gamma in enumerate(gammas):
        for coluna, alpha in enumerate(alphas):
            modelo = ajustar_credo(
                x_treino,
                y_treino,
                x_calibracao,
                y_calibracao,
                alpha,
                gamma,
                n_mcmc,
            )
            intervalos = modelo.prever(grade)
            eixo = eixos[linha, coluna]
            eixo.scatter(x_teste.ravel(), y_teste, s=12, color="k", alpha=0.35, zorder=3)
            eixo.fill_between(
                grade.ravel(), intervalos[:, 0], intervalos[:, 1], color="C0", alpha=0.25
            )
            eixo.plot(grade.ravel(), intervalos.mean(axis=1), color="C0", linewidth=1.2)
            eixo.grid(alpha=0.3)
            eixo.set_ylim(-1.35, 1.25)
            eixo.set_title(rf"$\alpha={alpha:.2f}$, $\gamma={gamma:.3f}$")
            if linha == len(gammas) - 1:
                eixo.set_xlabel("x")
            if coluna == 0:
                eixo.set_ylabel("y")

    figura.suptitle("CREDO-BART: grade de gamma fixo", fontsize=16, y=0.995)
    figura.tight_layout(rect=(0, 0, 1, 0.97))
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = RESULTADOS_DIR / "credo_alpha_gamma_grid.png"
    figura.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.show()
    return caminho


def main() -> None:
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    dados = gerar_dados_com_lacuna(1500)
    treino, restante = train_test_split(dados, test_size=0.5, random_state=42)
    calibracao, teste = train_test_split(restante, test_size=0.5, random_state=42)

    def extrair(amostra: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            amostra["x"].to_numpy(dtype=np.float32).reshape(-1, 1),
            amostra["y"].to_numpy(dtype=np.float32),
        )

    x_treino, y_treino = extrair(treino)
    x_calibracao, y_calibracao = extrair(calibracao)
    x_teste, y_teste = extrair(teste)
    caminho = plotar_grade(
        x_treino,
        y_treino,
        x_calibracao,
        y_calibracao,
        x_teste,
        y_teste,
        alphas=[0.05, 0.15, 0.30],
        gammas=[0.025, 0.10, 0.20],
    )
    print(f"Figura salva em {caminho}")


if __name__ == "__main__":
    main()
