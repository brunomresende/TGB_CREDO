#!/usr/bin/env python3
"""Gera boxplots da fração epistêmica do CREDO em pontos internos e atípicos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
PASTA_RESULTADOS = RAIZ / "resultados" / "brutos"
PASTA_FIGURAS = RAIZ / "resultados" / "figuras"

ORDEM_CONJUNTOS = [
    "concrete",
    "airfoil",
    "winewhite",
    "star",
    "winered",
    "cycle",
    "electric",
    "meps19",
    "superconductivity",
    "homes",
    "protein",
    "WEC",
]

METODO = "CREDO-QNN-Adaptativo"
COR_INTERNOS = "#86a6cf"
COR_ATIPICOS = "#e9ad89"


def carregar_resultados() -> pd.DataFrame:
    """Reúne as 30 repetições de cada conjunto do protocolo principal."""
    arquivos = sorted(PASTA_RESULTADOS.glob("resultados_*/metricas.csv"))
    if not arquivos:
        raise FileNotFoundError("Nenhum arquivo de métricas foi encontrado.")

    dados = pd.concat((pd.read_csv(arquivo) for arquivo in arquivos), ignore_index=True)
    dados = dados[dados["metodo"] == METODO].copy()
    ausentes = set(ORDEM_CONJUNTOS) - set(dados["conjunto"])
    if ausentes:
        raise ValueError(f"Conjuntos sem resultados: {sorted(ausentes)}")
    return dados


def desenhar_boxplot(dados: pd.DataFrame) -> plt.Figure:
    """Constrói os doze painéis com escalas horizontais próprias."""
    figura, eixos = plt.subplots(2, 6, figsize=(13.2, 6.1), sharey=True)

    for indice, (eixo, conjunto) in enumerate(zip(eixos.flat, ORDEM_CONJUNTOS)):
        recorte = dados[dados["conjunto"] == conjunto]
        internos = recorte["fracao_epistemica_internos"].dropna().to_numpy()
        atipicos = recorte["fracao_epistemica_atipicos"].dropna().to_numpy()

        caixas = eixo.boxplot(
            [internos, atipicos],
            orientation="horizontal",
            tick_labels=["Internos", "Atípicos"],
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#4a4a4a", "linewidth": 1.2},
            whiskerprops={"color": "#5c5c5c", "linewidth": 1.0},
            capprops={"color": "#5c5c5c", "linewidth": 1.0},
        )
        for caixa, cor in zip(caixas["boxes"], [COR_INTERNOS, COR_ATIPICOS]):
            caixa.set(facecolor=cor, edgecolor="#5c5c5c", linewidth=1.0, alpha=0.9)

        eixo.set_title(conjunto, fontsize=20)
        eixo.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
        eixo.tick_params(axis="both", labelsize=15)
        eixo.set_axisbelow(True)
        if indice % 6 != 0:
            eixo.tick_params(axis="y", labelleft=False)
        for borda in eixo.spines.values():
            borda.set_color("#cccccc")

    figura.supxlabel("Fração epistêmica", fontsize=15, y=0.035)
    figura.suptitle(
        "",
        fontsize=12,
        y=0.995,
    )
    figura.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.91, wspace=0.16, hspace=0.28)
    return figura


def main() -> None:
    dados = carregar_resultados()
    figura = desenhar_boxplot(dados)
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho_png = PASTA_FIGURAS / "boxplot_decomposicao_epistemica.png"
    caminho_pdf = PASTA_FIGURAS / "boxplot_decomposicao_epistemica.pdf"
    figura.savefig(caminho_png, dpi=300, bbox_inches="tight")
    figura.savefig(caminho_pdf, bbox_inches="tight")
    plt.close(figura)
    print(f"Figura salva em {caminho_png}")
    print(f"Versão vetorial salva em {caminho_pdf}")


if __name__ == "__main__":
    main()
