"""Gera grades de gamma adaptativo variando gamma_min e gamma_max.

O script produz duas figuras:

1. intervalos CREDO-BART para cada combinacao de gamma_min e gamma_max;
2. funcoes gamma(x) correspondentes.

Execute na raiz do projeto:
    uv run --extra bart python credo_adaptive_gamma_min_max_grid.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from credal_cp_tgb.credal.gamma import GammaAdaptativo
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.bart import BARTHeteroscedastico
from credo_alpha_gamma_grid import SEMENTE, gerar_dados_com_lacuna


RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados" / "figuras"


def ajustar_credo(
    x_treino: np.ndarray,
    y_treino: np.ndarray,
    x_calibracao: np.ndarray,
    y_calibracao: np.ndarray,
    *,
    alpha: float,
    gamma_minimo: float,
    gamma_maximo: float,
    tau: float,
    k: int,
    n_mcmc: int,
) -> RegressorCREDO:
    """Ajusta e calibra uma configuracao adaptativa do CREDO-BART."""
    if gamma_maximo <= gamma_minimo:
        raise ValueError("gamma_maximo deve ser maior que gamma_minimo.")

    modelo = RegressorCREDO(
        modelo_base=BARTHeteroscedastico(
            alpha=alpha,
            amostras=n_mcmc,
            cadeias=4,
            nucleos=4,
            alfa_arvore=0.985,
            barra_progresso=True,
            semente_aleatoria=SEMENTE,
        ),
        estrategia_gamma=GammaAdaptativo(
            gamma_minimo=gamma_minimo,
            gamma_maximo=gamma_maximo,
            tau=tau,
            numero_vizinhos=k,
        ),
        alpha=alpha,
        numero_amostras_posteriores=n_mcmc,
        semente_aleatoria=SEMENTE,
    )
    return modelo.ajustar(x_treino, y_treino).calibrar(
        x_calibracao,
        y_calibracao,
    )


def plotar_grade_intervalos(
    x_treino: np.ndarray,
    y_treino: np.ndarray,
    x_calibracao: np.ndarray,
    y_calibracao: np.ndarray,
    x_teste: np.ndarray,
    y_teste: np.ndarray,
    gamma_minimos: list[float],
    gamma_maximos: list[float],
    *,
    alpha: float = 0.1,
    tau: float = 1.0,
    k: int = 50,
    n_mcmc: int = 1000,
) -> tuple[dict[tuple[float, float], RegressorCREDO], np.ndarray, Path]:
    """Ajusta a grade e plota os intervalos preditivos conformalizados."""
    grade = np.linspace(-1.15, 1.15, 500, dtype=np.float32).reshape(-1, 1)
    figura, eixos = plt.subplots(
        len(gamma_minimos),
        len(gamma_maximos),
        figsize=(5 * len(gamma_maximos), 3.8 * len(gamma_minimos)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    modelos: dict[tuple[float, float], RegressorCREDO] = {}
    for linha, gamma_minimo in enumerate(gamma_minimos):
        for coluna, gamma_maximo in enumerate(gamma_maximos):
            modelo = ajustar_credo(
                x_treino,
                y_treino,
                x_calibracao,
                y_calibracao,
                alpha=alpha,
                gamma_minimo=gamma_minimo,
                gamma_maximo=gamma_maximo,
                tau=tau,
                k=k,
                n_mcmc=n_mcmc,
            )
            modelos[(gamma_minimo, gamma_maximo)] = modelo
            intervalos = modelo.prever(grade)

            eixo = eixos[linha, coluna]
            eixo.scatter(
                x_teste.ravel(),
                y_teste,
                s=12,
                color="k",
                alpha=0.35,
                zorder=3,
            )
            eixo.fill_between(
                grade.ravel(),
                intervalos[:, 0],
                intervalos[:, 1],
                color="C0",
                alpha=0.25,
            )
            eixo.plot(
                grade.ravel(),
                intervalos.mean(axis=1),
                color="C0",
                linewidth=1.2,
            )
            eixo.grid(alpha=0.3)
            eixo.set_ylim(-1.35, 1.25)
            eixo.set_title(
                rf"$\gamma_{{\min}}={gamma_minimo:.3f}$, "
                rf"$\gamma_{{\max}}={gamma_maximo:.2f}$"
            )
            if linha == len(gamma_minimos) - 1:
                eixo.set_xlabel("x")
            if coluna == 0:
                eixo.set_ylabel("y")

    figura.tight_layout()
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = RESULTADOS_DIR / "credo_adaptive_gamma_min_max_interval_grid.png"
    figura.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.show()
    return modelos, grade, caminho


def plotar_grade_gamma(
    modelos: dict[tuple[float, float], RegressorCREDO],
    grade: np.ndarray,
    gamma_minimos: list[float],
    gamma_maximos: list[float],
    *,
    tau: float,
) -> Path:
    """Plota gamma(x) para os modelos ja ajustados na grade."""
    figura, eixos = plt.subplots(
        len(gamma_minimos),
        len(gamma_maximos),
        figsize=(5 * len(gamma_maximos), 3 * len(gamma_minimos)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for linha, gamma_minimo in enumerate(gamma_minimos):
        for coluna, gamma_maximo in enumerate(gamma_maximos):
            modelo = modelos[(gamma_minimo, gamma_maximo)]
            gamma = modelo.estrategia_gamma.calcular(grade)

            eixo = eixos[linha, coluna]
            eixo.plot(grade.ravel(), gamma, color="C4", linewidth=2)
            eixo.grid(alpha=0.3)
            eixo.set_ylim(0, 1)
            eixo.set_title(
                rf"$\gamma_{{\min}}={gamma_minimo:.3f}$, "
                rf"$\gamma_{{\max}}={gamma_maximo:.2f}$"
            )
            if linha == len(gamma_minimos) - 1:
                eixo.set_xlabel("x")
            if coluna == 0:
                eixo.set_ylabel(r"$\gamma(x)$")

    figura.tight_layout()
    caminho = RESULTADOS_DIR / "credo_adaptive_gamma_min_max_schedule_grid.png"
    figura.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.show()
    return caminho


def main() -> None:
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    dados = gerar_dados_com_lacuna(1500)
    treino, restante = train_test_split(dados, test_size=0.5, random_state=42)
    calibracao, teste = train_test_split(
        restante,
        test_size=0.5,
        random_state=42,
    )

    def extrair(amostra: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            amostra["x"].to_numpy(dtype=np.float32).reshape(-1, 1),
            amostra["y"].to_numpy(dtype=np.float32),
        )

    x_treino, y_treino = extrair(treino)
    x_calibracao, y_calibracao = extrair(calibracao)
    x_teste, y_teste = extrair(teste)

    alpha = 0.10
    tau = 1.0
    k = 50
    gamma_minimos = [0.01, 0.05, 0.15]
    gamma_maximos = [0.75, 0.85, 0.95]

    modelos, grade, caminho_intervalos = plotar_grade_intervalos(
        x_treino,
        y_treino,
        x_calibracao,
        y_calibracao,
        x_teste,
        y_teste,
        gamma_minimos,
        gamma_maximos,
        alpha=alpha,
        tau=tau,
        k=k,
        n_mcmc=1000,
    )
    caminho_gamma = plotar_grade_gamma(
        modelos,
        grade,
        gamma_minimos,
        gamma_maximos,
        tau=tau,
    )
    print(f"Grade de intervalos salva em {caminho_intervalos}")
    print(f"Grade de gamma salva em {caminho_gamma}")


if __name__ == "__main__":
    main()
