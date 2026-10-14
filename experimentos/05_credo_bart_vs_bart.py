"""Compara CREDO-BART com CQR aplicado ao BART usual.

Os dois métodos compartilham exatamente o mesmo BART heteroscedástico ajustado.
O CQR-BART usa os quantis posteriores centrais do BART como limites iniciais e
os conformaliza. O CREDO constrói antes um envelope dos limites condicionais
amostrados e então aplica sua calibração conformal.

Execute a partir da raiz do repositório:

    uv run --extra bart python experimentos/05_credo_bart_vs_bart.py

O ajuste usa MCMC e, por isso, esta execução pode demorar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from credal_cp_tgb.avaliacao.metricas import avaliar_intervalos
from credal_cp_tgb.credal.gamma import GammaAdaptativo
from credal_cp_tgb.dados import DivisaoDados, criar_dados_ruido_variavel
from credal_cp_tgb.metodos.cqr import RegressorCQR
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.bart import BARTHeteroscedastico
from credal_cp_tgb.utilitarios.aleatoriedade import fixar_todas_sementes

ALPHA = 0.1
SEMENTE_GLOBAL = 125
SEMENTE_DADOS_E_DIVISAO = 42
SEMENTE_AJUSTE_BART = 1250
SEMENTE_CALIBRACAO_CREDO = 0
SEMENTE_PREDICAO = 45
NUMERO_AMOSTRAS = 1500
NUMERO_AMOSTRAS_POSTERIORES = 1000
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_SAIDA = RAIZ_PROJETO / "resultados" / "figuras" / "credo_bart_vs_bart.png"


def preparar_dados() -> DivisaoDados:
    """Gera e divide os dados como no experimento CREDO-BART versus CQR-RFQR."""
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
    return DivisaoDados(
        x_treino=x_treino,
        y_treino=y_treino,
        x_calibracao=x_calibracao,
        y_calibracao=y_calibracao,
        x_teste=x_teste,
        y_teste=y_teste,
    )


def ajustar_credo_bart(divisao: DivisaoDados) -> RegressorCREDO:
    """Ajusta uma vez o BART compartilhado e calibra somente o CREDO."""
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


def construir_cqr_bart(
    divisao: DivisaoDados,
    credo_bart: RegressorCREDO,
) -> RegressorCQR:
    """Conformaliza os quantis centrais do mesmo BART usado pelo CREDO."""
    return RegressorCQR(
        modelo_base=credo_bart.modelo_base,
        alpha=ALPHA,
        modelo_base_ajustado=True,
    ).calibrar(divisao.x_calibracao, divisao.y_calibracao)


def calcular_metricas(
    divisao: DivisaoDados,
    credo_bart: RegressorCREDO,
    cqr_bart: RegressorCQR,
) -> None:
    """Compara cobertura e eficiência no mesmo conjunto de teste."""
    intervalos_credo = credo_bart.prever(
        divisao.x_teste,
        semente_aleatoria=SEMENTE_PREDICAO,
    )
    intervalos_bart = cqr_bart.prever(divisao.x_teste)
    print("\nMétricas no conjunto de teste:")
    for nome, intervalos in (("CREDO-BART", intervalos_credo), ("CQR-BART", intervalos_bart)):
        metricas = avaliar_intervalos(divisao.y_teste, intervalos, alpha=ALPHA)
        valores = "  ".join(f"{chave}={valor:.4f}" for chave, valor in metricas.items())
        print(f"{nome:12s}  {valores}")


def criar_figura(
    divisao: DivisaoDados,
    credo_bart: RegressorCREDO,
    cqr_bart: RegressorCQR,
) -> None:
    """Desenha os dois intervalos, a decomposição e suas larguras."""
    grade_x = np.linspace(-1.15, 1.15, 500, dtype=np.float32).reshape(-1, 1)
    intervalos_credo, decomposicao = credo_bart.prever(
        grade_x,
        retornar_decomposicao=True,
        semente_aleatoria=SEMENTE_PREDICAO,
    )
    intervalos_bart = cqr_bart.prever(grade_x)
    eixo_x = grade_x[:, 0]

    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    figura, eixos = plt.subplots(2, 2, figsize=(14, 10))

    for eixo, intervalos, titulo, cor in (
        (eixos[0, 0], intervalos_credo, "CREDO-BART", "C0"),
        (eixos[0, 1], intervalos_bart, "CQR-BART", "C1"),
    ):
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
            intervalos[:, 0],
            intervalos[:, 1],
            color=cor,
            alpha=0.25,
            label="Intervalo preditivo",
        )
        eixo.plot(eixo_x, intervalos.mean(axis=1), color=cor, linewidth=1)
        eixo.axvspan(0.0, 0.4, color="grey", alpha=0.12, label="Região escassa")
        eixo.set(title=titulo, xlabel="x", ylabel="y", ylim=(-1.35, 3.0))
        eixo.grid(True, alpha=0.3)
        eixo.legend(frameon=False)

    eixo = eixos[1, 0]
    eixo.plot(eixo_x, decomposicao["aleatoria"], label="Aleatória", color="C2")
    eixo.plot(eixo_x, decomposicao["epistemica"], label="Epistêmica", color="C1")
    eixo.plot(
        eixo_x,
        decomposicao["folga_calibracao"],
        label="Ajuste conformal",
        color="C3",
    )
    eixo.axvspan(0.0, 0.4, color="grey", alpha=0.12, label="Região escassa")
    eixo.set(
        title="Decomposição da incerteza do CREDO-BART",
        xlabel="x",
        ylabel="Contribuição para a largura",
    )
    eixo.grid(True, alpha=0.3)
    eixo.legend(frameon=False)

    eixo = eixos[1, 1]
    eixo.plot(
        eixo_x,
        np.ptp(intervalos_credo, axis=1),
        label="CREDO-BART",
        color="C0",
    )
    eixo.plot(
        eixo_x,
        np.ptp(intervalos_bart, axis=1),
        label="CQR-BART",
        color="C1",
    )
    eixo.axvspan(0.0, 0.4, color="grey", alpha=0.12, label="Região escassa")
    eixo.set(title="Comparação das larguras", xlabel="x", ylabel="Comprimento")
    eixo.grid(True, alpha=0.3)
    eixo.legend(frameon=False)

    figura.tight_layout()
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=300, bbox_inches="tight")
    plt.close(figura)


def principal() -> None:
    fixar_todas_sementes(SEMENTE_GLOBAL)
    divisao = preparar_dados()
    print("1/3 - Ajustando e calibrando CREDO-BART...")
    credo_bart = ajustar_credo_bart(divisao)
    print("2/3 - Calibrando CQR-BART...")
    cqr_bart = construir_cqr_bart(divisao, credo_bart)
    calcular_metricas(divisao, credo_bart, cqr_bart)
    print("3/3 - Criando a figura...")
    criar_figura(divisao, credo_bart, cqr_bart)
    print(f"Figura salva em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    principal()
