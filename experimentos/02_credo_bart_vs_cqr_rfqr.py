"""Recria a figura histórica CREDO-BART versus CQR-RFQR.

Este experimento corresponde a ``credo_diferent_data_Figure1.py`` do
repositório original. Esta versão usa o cenário heteroscedástico com região
escassa em [0, 0.4] e RFQR (não CatBoost) como comparador.

Execute a partir da raiz do repositório:

    uv run --extra bart python experimentos/02_credo_bart_vs_cqr_rfqr.py

O BART usa amostragem MCMC e, por isso, esta execução pode demorar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from credal_cp_tgb.credal.gamma import GammaAdaptativo
from credal_cp_tgb.dados import (
    DivisaoDados,
    criar_dados_ruido_variavel,
)
from credal_cp_tgb.metodos.cqr import RegressorCQR
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.bart import BARTHeteroscedastico
from credal_cp_tgb.modelos.floresta_quantilica import RegressorQuantilicoFloresta
from credal_cp_tgb.utilitarios.aleatoriedade import fixar_todas_sementes

ALPHA = 0.1
SEMENTE_GLOBAL = 125
SEMENTE_DADOS_E_DIVISAO = 42
SEMENTE_CQR = 123
SEMENTE_AJUSTE_BART = 1250
SEMENTE_CALIBRACAO_CREDO = 0
SEMENTE_PREDICAO_CREDO = 45
NUMERO_AMOSTRAS = 1500
NUMERO_AMOSTRAS_POSTERIORES = 1000
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_SAIDA = RAIZ_PROJETO / "resultados" / "figuras" / "credo_bart_vs_cqr_rfqr.png"


def preparar_dados():
    """Gera e divide os dados na mesma ordem do código da figura original."""
    dados = criar_dados_ruido_variavel(
        numero_amostras=NUMERO_AMOSTRAS,
        probabilidade_regiao_escassa=0.02,
        desvio_base=0.2,
        semente_aleatoria=SEMENTE_DADOS_E_DIVISAO,
    )
    matriz_x = dados[["x"]].to_numpy(dtype=np.float32)
    vetor_y = dados["y"].to_numpy(dtype=np.float32)
    x_treino, x_restante, y_treino, y_restante = train_test_split(
        matriz_x,
        vetor_y,
        test_size=0.5,
        random_state=SEMENTE_DADOS_E_DIVISAO,
    )
    x_calibracao, x_teste, y_calibracao, y_teste = train_test_split(
        x_restante,
        y_restante,
        test_size=0.5,
        random_state=SEMENTE_DADOS_E_DIVISAO,
    )
    divisao = DivisaoDados(
        x_treino=x_treino,
        y_treino=y_treino,
        x_calibracao=x_calibracao,
        y_calibracao=y_calibracao,
        x_teste=x_teste,
        y_teste=y_teste,
    )
    return dados, divisao


def ajustar_credo_bart(divisao) -> RegressorCREDO:
    """Ajusta o BART e envolve suas amostras posteriores com o CREDO."""
    modelo_bart = BARTHeteroscedastico(
        alpha=ALPHA,
        amostras=NUMERO_AMOSTRAS_POSTERIORES,
        cadeias=4,
        nucleos=4,
        alfa_arvore=0.985,
        barra_progresso=True,
        semente_aleatoria=SEMENTE_AJUSTE_BART,
    )
    metodo = RegressorCREDO(
        modelo_base=modelo_bart,
        estrategia_gamma=GammaAdaptativo(gamma_minimo=0.05),
        alpha=ALPHA,
        numero_amostras_posteriores=NUMERO_AMOSTRAS_POSTERIORES,
        semente_aleatoria=SEMENTE_CALIBRACAO_CREDO,
    )
    return metodo.ajustar(divisao.x_treino, divisao.y_treino).calibrar(
        divisao.x_calibracao,
        divisao.y_calibracao,
    )


def ajustar_cqr_rfqr(divisao) -> RegressorCQR:
    """Treina a RFQR da figura oficial e aplica a calibração conformal."""
    modelo_rfqr = RegressorQuantilicoFloresta(
        alpha=ALPHA,
        numero_arvores=100,
        maximo_atributos="sqrt",
        minimo_amostras_folha=5,
        semente_aleatoria=SEMENTE_CQR,
    )
    return (
        RegressorCQR(modelo_rfqr, alpha=ALPHA)
        .ajustar(
            divisao.x_treino,
            divisao.y_treino,
        )
        .calibrar(
            divisao.x_calibracao,
            divisao.y_calibracao,
        )
    )


def criar_figura(divisao, credo_bart: RegressorCREDO, cqr_rfqr: RegressorCQR) -> None:
    """Desenha os intervalos e a decomposição da incerteza do CREDO."""
    grade_x = np.linspace(-1.15, 1.15, 500, dtype=np.float32).reshape(-1, 1)
    intervalos_credo, decomposicao = credo_bart.prever(
        grade_x,
        retornar_decomposicao=True,
        semente_aleatoria=SEMENTE_PREDICAO_CREDO,
    )
    intervalos_cqr = cqr_rfqr.prever(grade_x)
    eixo_x = grade_x[:, 0]

    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    figura, eixos = plt.subplots(2, 2, figsize=(14, 10))

    eixo = eixos[0, 0]
    eixo.scatter(
        divisao.x_teste[:, 0],
        divisao.y_teste,
        s=25,
        color="black",
        alpha=0.65,
        zorder=3,
        label="Dados de teste",
    )
    eixo.fill_between(
        eixo_x,
        intervalos_credo[:, 0],
        intervalos_credo[:, 1],
        color="C0",
        alpha=0.25,
        label="Intervalo preditivo",
    )
    eixo.plot(eixo_x, intervalos_credo.mean(axis=1), color="C0", linewidth=1)
    eixo.axvspan(0.0, 0.4, color="grey", alpha=0.12, label="Região escassa")
    eixo.set(title="CREDO-BART", xlabel="x", ylabel="y", ylim=(-1.35, 3.0))
    eixo.grid(True, alpha=0.3)
    eixo.legend(frameon=False)

    eixo = eixos[0, 1]
    eixo.scatter(
        divisao.x_teste[:, 0],
        divisao.y_teste,
        s=25,
        color="black",
        alpha=0.65,
        zorder=3,
    )
    eixo.fill_between(
        eixo_x,
        intervalos_cqr[:, 0],
        intervalos_cqr[:, 1],
        color="C1",
        alpha=0.25,
    )
    eixo.plot(eixo_x, intervalos_cqr.mean(axis=1), color="C1", linewidth=1)
    eixo.axvspan(0.0, 0.4, color="grey", alpha=0.12)
    eixo.set(title="CQR-QRF", xlabel="x", ylim=(-1.35, 3.0))
    eixo.grid(True, alpha=0.3)

    eixo = eixos[1, 0]
    eixo.plot(eixo_x, decomposicao["aleatoria"], label="Aleatória", color="C2")
    eixo.plot(eixo_x, decomposicao["epistemica"], label="Epistêmica", color="C1")
    eixo.axvspan(0.0, 0.4, color="grey", alpha=0.12, label="Região escassa")
    eixo.set(
        title="Decomposição da incerteza do CREDO-BART",
        xlabel="x",
        ylabel="Comprimento",
    )
    eixo.grid(True, alpha=0.3)
    eixo.legend(frameon=False)

    # O quarto painel permanece vazio para conservar a composição da figura original.
    eixos[1, 1].axis("off")
    figura.tight_layout()
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=300, bbox_inches="tight")
    plt.close(figura)


def principal() -> None:
    fixar_todas_sementes(SEMENTE_GLOBAL)
    _, divisao = preparar_dados()
    print("1/3 - Ajustando CREDO-BART...")
    credo_bart = ajustar_credo_bart(divisao)
    print("2/3 - Ajustando CQR-RFQR...")
    cqr_rfqr = ajustar_cqr_rfqr(divisao)
    print("3/3 - Criando a figura...")
    criar_figura(divisao, credo_bart, cqr_rfqr)
    print(f"Figura salva em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    principal()
