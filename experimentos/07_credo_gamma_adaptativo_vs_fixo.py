"""Compara o CREDO com gamma adaptativo e fixo em dados com duas lacunas.

Execute na raiz do projeto:
    uv run --extra bart python experimentos/07_credo_gamma_adaptativo_vs_fixo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from credal_cp_tgb.credal.gamma import GammaAdaptativo, GammaFixo
from credal_cp_tgb.dados import dividir_treino_calibracao_teste
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.bart import BARTHeteroscedastico

SEMENTE = 2026
ALPHA = 0.10
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_SAIDA = (
    RAIZ_PROJETO / "resultados" / "figuras" / "credo_gamma_adaptativo_vs_fixo.png"
)


def gerar_tres_nuvens() -> tuple[np.ndarray, np.ndarray]:
    """Cria três grupos densos e poucas observações nas duas lacunas."""
    rng = np.random.default_rng(SEMENTE)
    intervalos = ((-1.0, -0.48), (-0.16, 0.24), (0.56, 1.0))
    quantidades = (260, 220, 260)
    partes_x = [
        rng.uniform(inferior, superior, quantidade)
        for (inferior, superior), quantidade in zip(intervalos, quantidades, strict=True)
    ]
    partes_x.extend(
        (
            rng.uniform(-0.45, -0.19, 14),
            rng.uniform(0.27, 0.53, 14),
        )
    )
    x = np.concatenate(partes_x)

    media = -1.05 * x + 0.22 * np.sin(2.4 * np.pi * x)
    desvio = np.select(
        (x < -0.48, x < 0.24),
        (0.10 + 0.05 * (x + 1.0), 0.19),
        default=0.13 + 0.10 * (x - 0.56) / 0.44,
    )
    y = media + rng.normal(0.0, desvio)

    ordem = rng.permutation(len(x))
    return x[ordem, None], y[ordem]


def ajustar_metodos():
    """Ajusta um BART compartilhado e calibra as duas versões do CREDO."""
    X, y = gerar_tres_nuvens()
    divisao = dividir_treino_calibracao_teste(
        X,
        y,
        semente_aleatoria=SEMENTE,
    )
    modelo = BARTHeteroscedastico(
        alpha=ALPHA,
        numero_arvores=100,
        amostras=1000,
        cadeias=4,
        nucleos=4,
        alfa_arvore=0.985,
        barra_progresso=True,
        semente_aleatoria=SEMENTE,
    ).ajustar(divisao.x_treino, divisao.y_treino)

    comum = {
        "modelo_base": modelo,
        "alpha": ALPHA,
        "modelo_base_ajustado": True,
        "numero_amostras_posteriores": 500,
        "semente_aleatoria": SEMENTE,
    }
    credo_fixo = RegressorCREDO(
        estrategia_gamma=GammaFixo(0.10),
        **comum,
    ).ajustar(divisao.x_treino, divisao.y_treino)
    credo_adaptativo = RegressorCREDO(
        estrategia_gamma=GammaAdaptativo(
            gamma_minimo=0.05,
            gamma_maximo=0.60,
            tau=0.65,
        ),
        **comum,
    ).ajustar(divisao.x_treino, divisao.y_treino)

    for metodo in (credo_fixo, credo_adaptativo):
        metodo.calibrar(divisao.x_calibracao, divisao.y_calibracao)
    return divisao, credo_fixo, credo_adaptativo


def criar_figura(divisao, credo_fixo, credo_adaptativo) -> None:
    """Mostra os dois envelopes e o gamma adaptativo ao longo do domínio."""
    grade = np.linspace(-1.18, 1.18, 500)
    matriz_grade = grade[:, None]
    intervalos_fixos = credo_fixo.prever(matriz_grade)
    intervalos_adaptativos, decomposicao = credo_adaptativo.prever(
        matriz_grade,
        retornar_decomposicao=True,
    )

    figura, (eixo_superior, eixo_inferior) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (2.1, 1), "hspace": 0.12},
    )

    eixo_superior.fill_between(
        grade,
        intervalos_adaptativos[:, 0],
        intervalos_adaptativos[:, 1],
        color="#4C8BF5",
        alpha=0.25,
        label="CREDO adaptativo",
    )
    eixo_superior.fill_between(
        grade,
        intervalos_fixos[:, 0],
        intervalos_fixos[:, 1],
        color="#F4A261",
        alpha=0.28,
        label="CREDO fixo",
    )
    eixo_superior.plot(
        grade,
        intervalos_adaptativos.mean(axis=1),
        color="#2F6FD6",
        linewidth=1.4,
    )
    eixo_superior.plot(
        grade,
        intervalos_fixos.mean(axis=1),
        color="#E76F00",
        linewidth=1.2,
    )
    eixo_superior.scatter(
        divisao.x_teste[:, 0],
        divisao.y_teste,
        color="0.15",
        s=20,
        alpha=0.65,
        edgecolors="none",
        label="Dados de teste",
        zorder=3,
    )
    eixo_superior.set(ylabel="y")
    eixo_superior.legend(ncols=3, loc="upper center", frameon=False)
    eixo_superior.grid(alpha=0.25)

    eixo_inferior.plot(
        grade,
        decomposicao["gamma"],
        color="#7E57C2",
        linewidth=2,
    )
    eixo_inferior.fill_between(
        grade,
        0,
        decomposicao["gamma"],
        color="#7E57C2",
        alpha=0.10,
    )
    eixo_inferior.set(
        title=r"Gamma adaptativo $\gamma(x)$",
        xlabel="x",
        ylabel=r"$\gamma(x)$",
        xlim=(-1.18, 1.18),
        ylim=(0, 0.65),
    )
    eixo_inferior.grid(alpha=0.25)

    for eixo in (eixo_superior, eixo_inferior):
        eixo.spines[["top", "right"]].set_visible(False)

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=250, bbox_inches="tight")
    plt.close(figura)
    print(f"Figura salva em: {ARQUIVO_SAIDA}")


def principal() -> None:
    divisao, credo_fixo, credo_adaptativo = ajustar_metodos()
    criar_figura(divisao, credo_fixo, credo_adaptativo)


if __name__ == "__main__":
    principal()
