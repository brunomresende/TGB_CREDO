"""Introdução didática ao CQR e ao CREDO em um único experimento.

O arquivo deve ser lido de cima para baixo. Cada função representa uma etapa
conceitual: dados, modelo aleatório, métodos e visualização.

Execute:
    uv run python experimentos/01_introducao_cqr_e_credo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credal_cp_tgb.avaliacao.metricas import avaliar_intervalos
from credal_cp_tgb.credal.gamma import GammaAdaptativo, GammaFixo
from credal_cp_tgb.dados import criar_dados_lacuna_epistemica, dividir_treino_calibracao_teste
from credal_cp_tgb.metodos.cqr import RegressorCQR
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.rede_neural_quantilica import RedeNeuralQuantilica

ALPHA = 0.1
SEMENTE = 813920
NUMERO_AMOSTRAS = 2000
NUMERO_AMOSTRAS_POSTERIORES = 500
LIMITE_X_FIGURA = 1.4
NUMERO_PONTOS_GRADE = 500
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_SAIDA = RAIZ_PROJETO / "resultados" / "figuras" / "introducao_cqr_e_credo.png"


def gerar_e_dividir_dados():
    """Cria uma lacuna de dados e separa treino, calibração e teste."""
    dados = criar_dados_lacuna_epistemica(
        NUMERO_AMOSTRAS,
        semente_aleatoria=SEMENTE,
    )
    return dividir_treino_calibracao_teste(
        dados[["x"]].to_numpy(),
        dados["y"].to_numpy(),
        semente_aleatoria=SEMENTE,
    )


def ajustar_modelo_aleatorio(divisao):
    """Treina uma QNN; os métodos abaixo compartilharão os mesmos pesos."""
    return RedeNeuralQuantilica(
        tamanho_entrada=1,
        alpha=ALPHA,
        camadas_ocultas=(64,),
        epocas=2000,
        paciencia=50,
        tamanho_lote=32,
        dropout = 0.6,
        semente_aleatoria=SEMENTE,
    ).ajustar(divisao.x_treino, divisao.y_treino)


def construir_metodos(qnn, divisao):
    """Calibra CQR, CREDO com gamma fixo e CREDO com gamma adaptativo."""
    cqr = RegressorCQR(qnn, alpha=ALPHA, modelo_base_ajustado=True).calibrar(
        divisao.x_calibracao, divisao.y_calibracao
    )
    credo_fixo = RegressorCREDO(
        qnn,
        GammaFixo(0.1),
        alpha=ALPHA,
        modelo_base_ajustado=True,
        numero_amostras_posteriores=NUMERO_AMOSTRAS_POSTERIORES,
        semente_aleatoria=SEMENTE,
    )
    credo_adaptativo = RegressorCREDO(
        qnn,
        GammaAdaptativo(),
        alpha=ALPHA,
        modelo_base_ajustado=True,
        numero_amostras_posteriores=NUMERO_AMOSTRAS_POSTERIORES,
        semente_aleatoria=SEMENTE,
    )
    for metodo in (credo_fixo, credo_adaptativo):
        metodo.ajustar(divisao.x_treino, divisao.y_treino)
        metodo.calibrar(divisao.x_calibracao, divisao.y_calibracao)
    return cqr, credo_fixo, credo_adaptativo


def calcular_resultados(qnn, divisao, cqr, credo_fixo, credo_adaptativo):
    """Calcula métricas no teste e diagnósticos em grade com extrapolação."""
    linhas = []
    for nome, metodo in (
        ("CQR", cqr),
        ("CREDO fixo", credo_fixo),
        ("CREDO adaptativo", credo_adaptativo),
    ):
        intervalos_teste = metodo.prever(divisao.x_teste)
        linhas.append(
            {
                "método": nome,
                **avaliar_intervalos(divisao.y_teste, intervalos_teste, alpha=ALPHA),
            }
        )
    print("\nMétricas no conjunto de teste:")
    print(pd.DataFrame(linhas).to_string(index=False))

    grade = np.linspace(-LIMITE_X_FIGURA, LIMITE_X_FIGURA, NUMERO_PONTOS_GRADE)
    matriz_grade = grade[:, None]
    resultados = {"CQR": cqr.prever(matriz_grade)}
    resultados["Envelope credal"] = credo_adaptativo.prever(matriz_grade, conformalizar=False)
    resultados["CREDO fixo"] = credo_fixo.prever(matriz_grade)
    resultados["CREDO adaptativo"], decomposicao = credo_adaptativo.prever(
        matriz_grade, retornar_decomposicao=True
    )
    amostras_inferiores, amostras_superiores = qnn.amostrar_quantis(
        matriz_grade,
        numero_amostras=NUMERO_AMOSTRAS_POSTERIORES,
        semente_aleatoria=SEMENTE,
    )
    diagnostico_posterior = {
        "desvio_inferior": np.std(amostras_inferiores, axis=0),
        "desvio_superior": np.std(amostras_superiores, axis=0),
    }
    return grade, resultados, decomposicao, diagnostico_posterior


def criar_figura(divisao, grade, resultados, decomposicao, diagnostico_posterior):
    """Reúne a evolução do método e sua decomposição em uma única figura."""
    figura, eixos = plt.subplots(2, 4, figsize=(20, 8))

    paineis = (
        ("CQR", "CQR: quantis conformalizados"),
        ("Envelope credal", "CREDO antes da conformalização"),
        ("CREDO adaptativo", "CREDO depois da conformalização"),
        ("CREDO fixo", "CREDO com gamma fixo"),
        ("CREDO adaptativo", "CREDO com gamma adaptativo"),
    )
    for eixo, (chave, titulo) in zip(eixos.flat[:5], paineis, strict=True):
        intervalos = resultados[chave]
        eixo.scatter(divisao.x_teste[:, 0], divisao.y_teste, s=12, color="black", alpha=0.35)
        eixo.fill_between(grade, intervalos[:, 0], intervalos[:, 1], alpha=0.3)
        eixo.plot(grade, intervalos.mean(axis=1), linewidth=1)
        eixo.axvspan(-0.2, 0.2, color="grey", alpha=0.1)
        eixo.axvline(-1.0, color="grey", linestyle="--", linewidth=0.8)
        eixo.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
        eixo.set(
            title=titulo,
            xlabel="x",
            ylabel="y",
            xlim=(-LIMITE_X_FIGURA, LIMITE_X_FIGURA),
        )
        eixo.grid(alpha=0.25)

    eixo = eixos[1, 1]
    for chave, rotulo in (
        ("aleatoria", "Aleatória"),
        ("epistemica", "Epistêmica"),
        ("folga_calibracao", "Ajuste conformal (pode ser negativo)"),
    ):
        eixo.plot(grade, decomposicao[chave], label=rotulo)
    eixo.axvspan(-0.2, 0.2, color="grey", alpha=0.1, label="Região escassa")
    eixo.axvline(-1.0, color="grey", linestyle="--", linewidth=0.8)
    eixo.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
    eixo.set(
        title="Decomposição do CREDO",
        xlabel="x",
        ylabel="Contribuição para a largura",
        xlim=(-LIMITE_X_FIGURA, LIMITE_X_FIGURA),
    )
    eixo.grid(alpha=0.25)
    eixo.legend(frameon=False)

    eixo = eixos[1, 2]
    eixo.plot(grade, diagnostico_posterior["desvio_inferior"], label="Limite inferior")
    eixo.plot(grade, diagnostico_posterior["desvio_superior"], label="Limite superior")
    eixo.axvspan(-0.2, 0.2, color="grey", alpha=0.1, label="Região escassa")
    eixo.axvline(-1.0, color="grey", linestyle="--", linewidth=0.8)
    eixo.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
    eixo.set(
        title="Dispersão da posterior MC-dropout",
        xlabel="x",
        ylabel="Desvio-padrão",
        xlim=(-LIMITE_X_FIGURA, LIMITE_X_FIGURA),
    )
    eixo.grid(alpha=0.25)
    eixo.legend(frameon=False)

    eixo = eixos[1, 3]
    eixo.plot(grade, decomposicao["gamma"], color="C4")
    eixo.axvspan(-0.2, 0.2, color="grey", alpha=0.1, label="Região escassa")
    eixo.axvline(-1.0, color="grey", linestyle="--", linewidth=0.8)
    eixo.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
    eixo.set(
        title="Gamma adaptativo",
        xlabel="x",
        ylabel=r"$\gamma(x)$",
        xlim=(-LIMITE_X_FIGURA, LIMITE_X_FIGURA),
    )
    eixo.grid(alpha=0.25)
    eixo.legend(frameon=False)

    figura.tight_layout()
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=250, bbox_inches="tight")
    plt.close(figura)
    print(f"\nFigura salva em: {ARQUIVO_SAIDA}")


def principal():
    """Índice do experimento: dados → QNN → métodos → resultados → figura."""
    divisao = gerar_e_dividir_dados()
    qnn = ajustar_modelo_aleatorio(divisao)
    cqr, credo_fixo, credo_adaptativo = construir_metodos(qnn, divisao)
    grade, resultados, decomposicao, diagnostico_posterior = calcular_resultados(
        qnn, divisao, cqr, credo_fixo, credo_adaptativo
    )
    criar_figura(
        divisao,
        grade,
        resultados,
        decomposicao,
        diagnostico_posterior,
    )


if __name__ == "__main__":
    principal()
