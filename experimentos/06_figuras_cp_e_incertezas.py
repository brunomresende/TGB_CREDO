"""Gera duas ilustrações simples sobre predição conformal e incerteza.

Execute na raiz do projeto:
    uv run python experimentos/06_figuras_cp_e_incertezas.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor

from credal_cp_tgb.modelos.floresta_quantilica import RegressorQuantilicoFloresta

SEMENTE = 2026
ALPHA = 0.10
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_SAIDA = RAIZ_PROJETO / "resultados" / "figuras"


def quantil_conformal(escores: np.ndarray, alpha: float) -> float:
    """Calcula o quantil conformal finito, com correção para a amostra."""
    nivel = min(np.ceil((len(escores) + 1) * (1 - alpha)) / len(escores), 1.0)
    return float(np.quantile(escores, nivel, method="higher"))


def desenhar_faixa_difusa(
    eixo: plt.Axes,
    x: np.ndarray,
    centro: np.ndarray,
    largura: np.ndarray,
    cor: str,
) -> None:
    """Desenha uma faixa sem contorno, cuja opacidade desaparece nas bordas."""
    for proporcao in np.linspace(1.30, 0.15, 75):
        eixo.fill_between(
            x,
            centro - proporcao * largura,
            centro + proporcao * largura,
            color=cor,
            alpha=0.013,
            linewidth=0,
            edgecolor="none",
        )


def gerar_dados_heterocedasticos() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEMENTE)
    x = rng.uniform(0, 2 * np.pi, 1500)
    media = np.sin(x)
    desvio = 0.08 + 0.5 * x / (2 * np.pi)
    y = media + rng.normal(0, desvio)
    return x, y


def dividir_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEMENTE)
    indices = rng.permutation(n)
    tamanho = n // 3
    return indices[:tamanho], indices[tamanho : 2 * tamanho], indices[2 * tamanho :]


def figura_split_conformal_e_cqr() -> Path:
    """Compara a faixa constante do Split Conformal à faixa adaptativa do CQR."""
    x, y = gerar_dados_heterocedasticos()
    indice_treino, indice_calibracao, _ = dividir_indices(len(x))
    x_treino = x[indice_treino, None]
    y_treino = y[indice_treino]
    x_calibracao = x[indice_calibracao, None]
    y_calibracao = y[indice_calibracao]

    floresta_media = RandomForestRegressor(
        n_estimators=500,
        random_state=SEMENTE,
        n_jobs=-1,
    ).fit(x_treino, y_treino)
    residuos = np.abs(y_calibracao - floresta_media.predict(x_calibracao))
    ajuste_split = quantil_conformal(residuos, ALPHA)

    floresta_quantilica = RegressorQuantilicoFloresta(
        alpha=ALPHA,
        numero_arvores=500,
        maximo_atributos=1,
        minimo_amostras_folha=5,
        semente_aleatoria=SEMENTE,
    ).ajustar(x_treino, y_treino)
    quantis_calibracao = floresta_quantilica.prever_quantis(x_calibracao)
    escores_cqr = np.maximum(
        quantis_calibracao[:, 0] - y_calibracao,
        y_calibracao - quantis_calibracao[:, 1],
    )
    ajuste_cqr = quantil_conformal(escores_cqr, ALPHA)

    grade = np.linspace(-0.75, 2 * np.pi + 0.75, 300)
    matriz_grade = grade[:, None]
    media_grade = floresta_media.predict(matriz_grade)
    quantis_grade = floresta_quantilica.prever_quantis(matriz_grade)

    figura, eixos = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    configuracoes = (
        (
            "Resíduo absoluto",
            media_grade - ajuste_split,
            media_grade + ajuste_split,
            "#D4A017",
        ),
        (
            "CQR",
            quantis_grade[:, 0] - ajuste_cqr,
            quantis_grade[:, 1] + ajuste_cqr,
            "#6B8E23",
        ),
    )
    for eixo, (titulo, inferior, superior, cor) in zip(eixos, configuracoes, strict=True):
        eixo.fill_between(grade, inferior, superior, color=cor, alpha=0.5, label=titulo)
        eixo.scatter(x, y, color="black", s=9, alpha=0.35, edgecolors="none")
        eixo.set(title=titulo, xlabel="x", xlim=(-0.75, 2 * np.pi + 0.75))
        eixo.grid(alpha=0.25)
        eixo.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    eixos[0].set_ylabel("y")
    figura.tight_layout()

    caminho = PASTA_SAIDA / "split_conformal_vs_cqr.png"
    figura.savefig(caminho, dpi=250, bbox_inches="tight")
    plt.close(figura)
    return caminho


def figura_incertezas() -> Path:
    """Ilustra incerteza epistêmica numa lacuna e aleatória nas regiões observadas."""
    rng = np.random.default_rng(SEMENTE)
    n = 300
    n_meio = max(1, int(np.floor(n * 0.01)))
    n_lado = (n - n_meio) // 2
    x = np.concatenate(
        (
            rng.uniform(-1.0, -0.25, n_lado),
            rng.uniform(-0.25, 0.25, n_meio),
            rng.uniform(0.25, 1.0, n_lado),
        )
    )
    media = 1.2 * np.sin(1.3 * np.pi * x)
    desvio = np.where(x <= 0.5, 0.08, 0.08 + 0.25 * ((x - 0.5) / 0.5) ** 2)
    y = media + rng.normal(0, desvio)

    limite_dados = 1.0
    limite_figura = 1.35
    grade = np.linspace(-limite_figura, limite_figura, 1400)
    curva = 1.2 * np.sin(1.3 * np.pi * grade)
    mascara_extrapolacao_esquerda = grade < -limite_dados
    mascara_esquerda = (grade >= -limite_dados) & (grade <= -0.25)
    mascara_meio = (grade >= -0.25) & (grade <= 0.25)
    mascara_direita = (grade >= 0.25) & (grade <= limite_dados)
    mascara_extrapolacao_direita = grade > limite_dados

    # Perfil suave e assimétrico, conectado às faixas aleatórias dos dois lados.
    x_meio = grade[mascara_meio]
    posicao_meio = (x_meio + 0.25) / 0.5
    transicao_suave = 3 * posicao_meio**2 - 2 * posicao_meio**3
    base_epistemica = 0.19 + (0.10 - 0.19) * transicao_suave
    largura_epistemica = base_epistemica + 0.35 * np.sin(np.pi * posicao_meio) ** 2
    largura_esquerda = np.full(mascara_esquerda.sum(), 0.19)
    largura_direita = 0.10 + 0.40 * ((grade[mascara_direita] - 0.25) / 0.75) ** 2

    # Fora do suporte observado, a incerteza epistêmica cresce suavemente.
    distancia_esquerda = (-limite_dados - grade[mascara_extrapolacao_esquerda]) / (
        limite_figura - limite_dados
    )
    distancia_direita = (grade[mascara_extrapolacao_direita] - limite_dados) / (
        limite_figura - limite_dados
    )
    largura_extrapolacao_esquerda = 0.19 + 0.40 * distancia_esquerda**2
    largura_extrapolacao_direita = 0.50 + 0.40 * distancia_direita**2

    figura, eixo = plt.subplots(figsize=(8, 5))
    for mascara, largura in (
        (mascara_esquerda, largura_esquerda),
        (mascara_direita, largura_direita),
    ):
        desenhar_faixa_difusa(
            eixo,
            grade[mascara],
            curva[mascara],
            largura,
            "#E74C3C",
        )
    desenhar_faixa_difusa(
        eixo,
        grade[mascara_meio],
        curva[mascara_meio],
        largura_epistemica,
        "#3498DB",
    )
    for mascara, largura in (
        (mascara_extrapolacao_esquerda, largura_extrapolacao_esquerda),
        (mascara_extrapolacao_direita, largura_extrapolacao_direita),
    ):
        desenhar_faixa_difusa(
            eixo,
            grade[mascara],
            curva[mascara],
            largura,
            "#3498DB",
        )
    eixo.scatter(x, y, color="0.25", s=18)
    eixo.axvline(-limite_dados, color="0.45", linestyle="--", linewidth=1)
    eixo.axvline(limite_dados, color="0.45", linestyle="--", linewidth=1)
    eixo.set(xlabel="x", ylabel="y", xlim=(-limite_figura, limite_figura))
    eixo.grid(alpha=0.25)
    eixo.spines[["top", "right"]].set_visible(False)
    eixo.legend(
        handles=[
            Patch(facecolor="#E74C3C", alpha=0.5, label="Incerteza Aleatória"),
            Patch(facecolor="#3498DB", alpha=0.6, label="Incerteza Epistêmica"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncols=2,
        frameon=False,
    )
    figura.tight_layout()

    caminho = PASTA_SAIDA / "incerteza_aleatoria_e_epistemica.png"
    figura.savefig(caminho, dpi=250, bbox_inches="tight")
    plt.close(figura)
    return caminho


def principal() -> None:
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    caminhos = (figura_split_conformal_e_cqr(), figura_incertezas())
    for caminho in caminhos:
        print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    principal()
