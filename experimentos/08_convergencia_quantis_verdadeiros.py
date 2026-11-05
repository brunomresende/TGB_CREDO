"""Compara CREDO-BART e CQR-QRF com quantis verdadeiros conhecidos.

Quatro cenários aumentam progressivamente a quantidade de observações em uma
região inicialmente escassa. Execute na raiz do projeto:

    uv run --extra bart python experimentos/08_convergencia_quantais_verdadeiros.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

from credal_cp_tgb.credal.gamma import GammaAdaptativo
from credal_cp_tgb.dados import dividir_treino_calibracao_teste
from credal_cp_tgb.metodos.cqr import RegressorCQR
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.bart import BARTHeteroscedastico
from credal_cp_tgb.modelos.floresta_quantilica import RegressorQuantilicoFloresta

SEMENTE = 2026
ALPHA = 0.10
REGIAO_ESCASSA = (-0.20, 0.20)
QUANTIDADES_ESCASSAS = (5, 25, 100, 400)
NUMERO_DENSO = 1000
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_SAIDA = (
    RAIZ_PROJETO / "resultados" / "figuras" / "convergencia_quantis_verdadeiros.png"
)


def media_verdadeira(x: np.ndarray) -> np.ndarray:
    """Média condicional usada no processo gerador."""
    return 0.85 * np.sin(1.7 * np.pi * x) - 0.25 * x


def desvio_verdadeiro(x: np.ndarray) -> np.ndarray:
    """Desvio-padrão condicional heteroscedástico."""
    return 0.10 + 0.20 * ((x + 1.0) / 2.0) ** 2


def gerar_bancos_de_dados() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gera uma parte densa fixa e um banco aninhado para a região escassa."""
    rng = np.random.default_rng(SEMENTE)
    n_esquerda = NUMERO_DENSO // 2
    x_denso = np.concatenate(
        (
            rng.uniform(-1.0, REGIAO_ESCASSA[0], n_esquerda),
            rng.uniform(REGIAO_ESCASSA[1], 1.0, NUMERO_DENSO - n_esquerda),
        )
    )
    x_escasso = rng.uniform(*REGIAO_ESCASSA, max(QUANTIDADES_ESCASSAS))

    y_denso = media_verdadeira(x_denso) + rng.normal(
        0.0,
        desvio_verdadeiro(x_denso),
    )
    y_escasso = media_verdadeira(x_escasso) + rng.normal(
        0.0,
        desvio_verdadeiro(x_escasso),
    )
    return x_denso, y_denso, x_escasso, y_escasso


def montar_cenario(
    x_denso: np.ndarray,
    y_denso: np.ndarray,
    x_escasso: np.ndarray,
    y_escasso: np.ndarray,
    quantidade_escassa: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Acrescenta à parte densa os primeiros pontos do banco escasso."""
    x = np.concatenate((x_denso, x_escasso[:quantidade_escassa]))
    y = np.concatenate((y_denso, y_escasso[:quantidade_escassa]))
    rng = np.random.default_rng(SEMENTE + quantidade_escassa)
    ordem = rng.permutation(len(x))
    return x[ordem, None], y[ordem]


def ajustar_e_prever(X: np.ndarray, y: np.ndarray, grade: np.ndarray, semente: int):
    """Ajusta os dois métodos no mesmo particionamento e prevê sobre a grade."""
    divisao = dividir_treino_calibracao_teste(
        X,
        y,
        fracao_treino=0.50,
        fracao_calibracao=0.25,
        fracao_teste=0.25,
        semente_aleatoria=semente,
    )

    bart = BARTHeteroscedastico(
        alpha=ALPHA,
        numero_arvores=100,
        amostras=500,
        cadeias=2,
        nucleos=2,
        alfa_arvore=0.985,
        barra_progresso=True,
        semente_aleatoria=semente,
    )
    credo = RegressorCREDO(
        modelo_base=bart,
        estrategia_gamma=GammaAdaptativo(
            gamma_minimo=0.05,
            gamma_maximo=0.75,
            tau=0.75,
        ),
        alpha=ALPHA,
        numero_amostras_posteriores=300,
        semente_aleatoria=semente,
    )
    credo.ajustar(divisao.x_treino, divisao.y_treino)
    credo.calibrar(divisao.x_calibracao, divisao.y_calibracao)

    qrf = RegressorQuantilicoFloresta(
        alpha=ALPHA,
        numero_arvores=300,
        maximo_atributos=1,
        minimo_amostras_folha=5,
        semente_aleatoria=semente,
    )
    cqr = RegressorCQR(qrf, alpha=ALPHA)
    cqr.ajustar(divisao.x_treino, divisao.y_treino)
    cqr.calibrar(divisao.x_calibracao, divisao.y_calibracao)

    return credo.prever(grade[:, None]), cqr.prever(grade[:, None])


def criar_figura() -> None:
    """Ajusta os quatro cenários e produz o painel de convergência."""
    x_denso, y_denso, x_escasso, y_escasso = gerar_bancos_de_dados()
    grade = np.linspace(-1.08, 1.08, 500)
    media = media_verdadeira(grade)
    desvio = desvio_verdadeiro(grade)
    quantil_inferior = media + norm.ppf(ALPHA / 2) * desvio
    quantil_superior = media + norm.ppf(1 - ALPHA / 2) * desvio

    figura, eixos = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for indice, (eixo, quantidade) in enumerate(
        zip(eixos.flat, QUANTIDADES_ESCASSAS, strict=True)
    ):
        print(f"Cenário {indice + 1}/4: {quantidade} pontos na região escassa")
        X, y = montar_cenario(x_denso, y_denso, x_escasso, y_escasso, quantidade)
        intervalos_credo, intervalos_cqr = ajustar_e_prever(
            X,
            y,
            grade,
            SEMENTE + indice,
        )

        eixo.axvspan(*REGIAO_ESCASSA, color="0.85", alpha=0.55, zorder=0)
        eixo.fill_between(
            grade,
            intervalos_credo[:, 0],
            intervalos_credo[:, 1],
            color="#4C8BF5",
            alpha=0.25,
            linewidth=0,
        )
        eixo.fill_between(
            grade,
            intervalos_cqr[:, 0],
            intervalos_cqr[:, 1],
            color="#F4A261",
            alpha=0.25,
            linewidth=0,
        )
        limite_inferior_comum = np.maximum(
            intervalos_credo[:, 0],
            intervalos_cqr[:, 0],
        )
        limite_superior_comum = np.minimum(
            intervalos_credo[:, 1],
            intervalos_cqr[:, 1],
        )
        eixo.fill_between(
            grade,
            limite_inferior_comum,
            limite_superior_comum,
            where=limite_inferior_comum <= limite_superior_comum,
            color="#8E63B0",
            alpha=0.42,
            linewidth=0,
        )
        eixo.plot(grade, quantil_inferior, color="black", linestyle="--", linewidth=1.7)
        eixo.plot(grade, quantil_superior, color="black", linestyle="--", linewidth=1.7)
        eixo.scatter(X[:, 0], y, s=8, color="0.20", alpha=0.22, edgecolors="none")
        eixo.set_title(f"{quantidade} pontos na região escassa")
        eixo.grid(alpha=0.20)
        eixo.spines[["top", "right"]].set_visible(False)

    for eixo in eixos[:, 0]:
        eixo.set_ylabel("y")
    for eixo in eixos[-1, :]:
        eixo.set_xlabel("x")

    figura.legend(
        handles=[
            Patch(facecolor="#4C8BF5", alpha=0.25, label="CREDO-BART"),
            Patch(facecolor="#F4A261", alpha=0.25, label="CQR-QRF"),
            Patch(facecolor="#8E63B0", alpha=0.42, label="Interseção"),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.7,
                label="Quantis verdadeiros de 5% e 95%",
            ),
            Patch(facecolor="0.85", alpha=0.55, label="Região inicialmente escassa"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncols=5,
        frameon=False,
    )
    figura.tight_layout(rect=(0, 0.06, 1, 1))
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=250, bbox_inches="tight")
    plt.close(figura)
    print(f"Figura salva em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    criar_figura()
