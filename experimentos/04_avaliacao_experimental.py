"""Executa as comparações sintética, de modelos-base ou de dados reais.

Exemplos:
    uv run python experimentos/04_avaliacao_experimental.py --cenario sintetico
    uv run --extra bart python experimentos/04_avaliacao_experimental.py --cenario modelos-base
    uv run --extra completo python experimentos/04_avaliacao_experimental.py \
        --cenario reais --configuracao configuracoes/comparacao_rapida.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credal_cp_tgb.avaliacao.metricas import avaliar_intervalos
from credal_cp_tgb.credal.gamma import GammaAdaptativo, GammaFixo
from credal_cp_tgb.dados import criar_dados_lacuna_epistemica, dividir_treino_calibracao_teste
from credal_cp_tgb.metodos.cqr import RegressorCQR
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.bart import BARTHeteroscedastico
from credal_cp_tgb.modelos.rede_neural_quantilica import RedeNeuralQuantilica

ALPHA = 0.1
SEMENTE = 125
RAIZ_PROJETO = Path(__file__).resolve().parents[1]


def preparar_problema_sintetico():
    """Cria o mesmo problema controlado para todas as comparações didáticas."""
    dados = criar_dados_lacuna_epistemica(700, semente_aleatoria=SEMENTE)
    return dividir_treino_calibracao_teste(
        dados[["x"]].to_numpy(),
        dados["y"].to_numpy(),
        semente_aleatoria=SEMENTE,
    )


def ajustar_qnn(divisao):
    """Treina uma QNN que pode ser compartilhada entre vários métodos."""
    return RedeNeuralQuantilica(
        tamanho_entrada=1,
        alpha=ALPHA,
        camadas_ocultas=(64, 64),
        epocas=300,
        paciencia=25,
        tamanho_lote=32,
        semente_aleatoria=SEMENTE,
    ).ajustar(divisao.x_treino, divisao.y_treino)


def avaliar_metodo(nome, metodo, divisao):
    """Faz a calibração que falta, prevê e devolve uma linha de métricas."""
    if isinstance(metodo, RegressorCREDO):
        metodo.ajustar(divisao.x_treino, divisao.y_treino)
    metodo.calibrar(divisao.x_calibracao, divisao.y_calibracao)
    intervalos = metodo.prever(divisao.x_teste)
    return {
        "método": nome,
        **avaliar_intervalos(divisao.y_teste, intervalos, alpha=ALPHA),
    }


def comparar_metodos_sinteticos():
    """Compara CQR, CQR relativo e as duas versões do CREDO-QNN."""
    divisao = preparar_problema_sintetico()
    qnn = ajustar_qnn(divisao)
    metodos = {
        "CQR-QNN": RegressorCQR(qnn, modelo_base_ajustado=True),
        "CQR-r-QNN": RegressorCQR(qnn, relativo=True, modelo_base_ajustado=True),
        "CREDO-QNN": RegressorCREDO(
            qnn, GammaFixo(), modelo_base_ajustado=True, numero_amostras_posteriores=300
        ),
        "CREDO-QNN adaptativo": RegressorCREDO(
            qnn,
            GammaAdaptativo(),
            modelo_base_ajustado=True,
            numero_amostras_posteriores=300,
        ),
    }
    linhas = [avaliar_metodo(nome, metodo, divisao) for nome, metodo in metodos.items()]
    print(pd.DataFrame(linhas).to_string(index=False))


def comparar_modelos_base():
    """Compara CREDO-QNN e CREDO-BART no mesmo conjunto sintético."""
    divisao = preparar_problema_sintetico()
    print("1/2 - Ajustando QNN...")
    qnn = ajustar_qnn(divisao)
    print("2/2 - Ajustando BART...")
    bart = BARTHeteroscedastico(amostras=500, cadeias=2, nucleos=2).ajustar(
        divisao.x_treino, divisao.y_treino
    )
    modelos = {"QNN": qnn, "BART": bart}
    intervalos_por_modelo = {}
    linhas = []
    for nome, modelo in modelos.items():
        metodo = RegressorCREDO(
            modelo,
            GammaAdaptativo(),
            modelo_base_ajustado=True,
            numero_amostras_posteriores=300,
        )
        metodo.ajustar(divisao.x_treino, divisao.y_treino)
        metodo.calibrar(divisao.x_calibracao, divisao.y_calibracao)
        intervalos = metodo.prever(divisao.x_teste)
        intervalos_por_modelo[nome] = intervalos
        linhas.append(
            {
                "modelo-base": nome,
                **avaliar_intervalos(divisao.y_teste, intervalos, alpha=ALPHA),
            }
        )
    print(pd.DataFrame(linhas).to_string(index=False))
    criar_figura_modelos_base(divisao, intervalos_por_modelo)


def criar_figura_modelos_base(divisao, intervalos_por_modelo):
    """Mostra lado a lado os intervalos obtidos com QNN e BART."""
    ordem = np.argsort(divisao.x_teste[:, 0])
    figura, eixos = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for eixo, (nome, intervalos) in zip(eixos, intervalos_por_modelo.items(), strict=True):
        eixo.scatter(divisao.x_teste[:, 0], divisao.y_teste, s=12, color="black", alpha=0.35)
        eixo.fill_between(
            divisao.x_teste[ordem, 0], intervalos[ordem, 0], intervalos[ordem, 1], alpha=0.3
        )
        eixo.set(title=f"CREDO-{nome}", xlabel="x", ylabel="y")
        eixo.grid(alpha=0.25)
    figura.tight_layout()
    saida = RAIZ_PROJETO / "resultados" / "figuras" / "comparacao_qnn_bart.png"
    saida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(saida, dpi=250, bbox_inches="tight")
    plt.close(figura)
    print(f"Figura salva em: {saida}")


def comparar_dados_reais(caminho_configuracao):
    """Delega a comparação completa ao script reproduzível."""
    if caminho_configuracao is None:
        raise ValueError("O cenário 'reais' requer --configuracao ARQUIVO.yaml.")
    subprocess.run(
        [
            sys.executable,
            str(RAIZ_PROJETO / "scripts" / "executar_experimento.py"),
            "--configuracao",
            str(caminho_configuracao),
        ],
        check=True,
    )


def ler_argumentos():
    analisador = argparse.ArgumentParser(description="Comparações experimentais do CREDO.")
    analisador.add_argument(
        "--cenario",
        choices=("sintetico", "modelos-base", "reais"),
        default="sintetico",
        help="comparação a executar (padrão: sintetico)",
    )
    analisador.add_argument(
        "--configuracao",
        type=Path,
        help="arquivo YAML usado pelo cenário de dados reais",
    )
    return analisador.parse_args()


def principal():
    """Escolhe uma comparação sem executar obrigatoriamente as etapas caras."""
    argumentos = ler_argumentos()
    if argumentos.cenario == "sintetico":
        comparar_metodos_sinteticos()
    elif argumentos.cenario == "modelos-base":
        comparar_modelos_base()
    else:
        comparar_dados_reais(argumentos.configuracao)


if __name__ == "__main__":
    principal()
