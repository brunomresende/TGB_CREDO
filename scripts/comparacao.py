"""Orquestração das comparações; cada método permanece em seu próprio módulo."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from credal_cp_tgb.avaliacao.agregacao import agregar_metricas
from credal_cp_tgb.avaliacao.metricas import avaliar_intervalos
from credal_cp_tgb.avaliacao.valores_atipicos import detectar_grupos_valores_atipicos
from credal_cp_tgb.credal.gamma import GammaAdaptativo, GammaFixo
from credal_cp_tgb.dados import (
    carregar_conjunto_processado,
    dividir_treino_calibracao_teste,
)
from credal_cp_tgb.metodos.cqr import RegressorCQR
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.metodos.epicscore import RegressorEPICScore
from credal_cp_tgb.metodos.uacqr import RegressorUACQR
from credal_cp_tgb.modelos.catboost_quantilico import RegressorQuantilicoCatBoost
from credal_cp_tgb.modelos.rede_neural_quantilica import RedeNeuralQuantilica
from credal_cp_tgb.utilitarios.aleatoriedade import (
    fixar_todas_sementes,
    gerar_sementes_repeticoes,
)
from credal_cp_tgb.utilitarios.tempo import calcular_tempo_total_metodo


def carregar_configuracao(caminho: str | Path) -> dict:
    """Lê um arquivo YAML de configuração."""
    with Path(caminho).open(encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def _criar_qnn(
    configuracao: dict,
    tamanho_entrada: int,
    alpha: float,
    semente: int,
    numero_observacoes: int,
    conjunto: str,
) -> RedeNeuralQuantilica:
    valores = configuracao["qnn"]
    if conjunto == "WEC":
        tamanho_lote = valores.get(
            "tamanho_lote_wec",
            valores.get("tamanho_lote", 250),
        )
    elif numero_observacoes >= 10_000:
        tamanho_lote = valores.get(
            "tamanho_lote_grande",
            valores.get("tamanho_lote", 120),
        )
    else:
        tamanho_lote = valores.get(
            "tamanho_lote_pequeno",
            valores.get("tamanho_lote", 32),
        )
    return RedeNeuralQuantilica(
        tamanho_entrada=tamanho_entrada,
        alpha=alpha,
        camadas_ocultas=tuple(valores["camadas_ocultas"]),
        dropout=valores["dropout"],
        epocas=valores["epocas"],
        paciencia=valores["paciencia"],
        taxa_aprendizado=valores["taxa_aprendizado"],
        decaimento_pesos=valores["decaimento_pesos"],
        passo_agendador=valores.get("passo_agendador", 10),
        gamma_agendador=valores.get("gamma_agendador", 0.99),
        tamanho_lote=tamanho_lote,
        semente_aleatoria=semente,
    )


def _avaliar(
    metodo: str,
    intervalos,
    divisao,
    grupos,
    alpha: float,
    **informacoes_extras,
) -> dict:
    metricas = avaliar_intervalos(
        divisao.y_teste,
        intervalos,
        alpha=alpha,
        indices_atipicos=grupos.valores_atipicos,
        indices_internos=grupos.internos_ilr,
    )
    return {"metodo": metodo, **metricas, **informacoes_extras}


def executar_repeticao(
    X,
    y,
    *,
    conjunto: str,
    repeticao: int,
    semente: int,
    configuracao: dict,
    incluir_catboost: bool = True,
) -> list[dict]:
    """Executa todos os métodos para uma única divisão aleatória dos dados."""
    alpha = float(configuracao["alpha"])
    config_divisao = configuracao["divisao"]
    divisao = dividir_treino_calibracao_teste(
        X,
        y,
        fracao_treino=config_divisao["treino"],
        fracao_calibracao=config_divisao["calibracao"],
        fracao_teste=config_divisao["teste"],
        semente_aleatoria=semente,
    )
    config_atipicos = configuracao["valores_atipicos"]
    grupos = detectar_grupos_valores_atipicos(
        divisao.x_teste,
        semente_aleatoria=semente,
        **config_atipicos,
    )
    informacoes_comuns = {
        "conjunto": conjunto,
        "repeticao": repeticao,
        "semente": semente,
    }
    linhas: list[dict] = []

    # Uma única QNN compartilhada torna a comparação controlada.
    qnn = _criar_qnn(
        configuracao,
        divisao.x_treino.shape[1],
        alpha,
        semente,
        len(X),
        conjunto,
    )
    inicio = time.perf_counter()
    qnn.ajustar(divisao.x_treino, divisao.y_treino)
    tempo_ajuste_qnn = time.perf_counter() - inicio

    metodos_qnn = {
        "CQR-QNN": RegressorCQR(
            qnn,
            alpha=alpha,
            relativo=False,
            modelo_base_ajustado=True,
        ),
        "CQR-r-QNN": RegressorCQR(
            qnn,
            alpha=alpha,
            relativo=True,
            modelo_base_ajustado=True,
        ),
    }
    for nome, metodo in metodos_qnn.items():
        inicio_metodo = time.perf_counter()
        metodo.calibrar(divisao.x_calibracao, divisao.y_calibracao)
        inicio = time.perf_counter()
        intervalos = metodo.prever(divisao.x_teste)
        fim_metodo = time.perf_counter()
        tempo_predicao = fim_metodo - inicio
        linhas.append(
            {
                **informacoes_comuns,
                **_avaliar(nome, intervalos, divisao, grupos, alpha),
                "modelo_base": "QNN",
                "segundos_ajuste_compartilhado": tempo_ajuste_qnn,
                "segundos_predicao": tempo_predicao,
                "segundos_total_metodo": calcular_tempo_total_metodo(
                    tempo_ajuste_qnn,
                    inicio_metodo,
                    fim_metodo,
                ),
            }
        )

    config_gamma = configuracao["gamma"]
    metodos_credo = {
        "CREDO-QNN": RegressorCREDO(
            qnn,
            GammaFixo(config_gamma["fixo"]),
            alpha=alpha,
            numero_amostras_posteriores=configuracao["qnn"]["numero_amostras_mc"],
            modelo_base_ajustado=True,
            semente_aleatoria=semente,
        ),
        "CREDO-QNN-Adaptativo": RegressorCREDO(
            qnn,
            GammaAdaptativo(
                gamma_minimo=config_gamma["minimo"],
                gamma_maximo=config_gamma["maximo"],
                tau=config_gamma["tau"],
                epsilon=config_gamma.get("epsilon", 1e-6),
                constante_vizinhos=config_gamma.get("constante_vizinhos", 6.672),
            ),
            alpha=alpha,
            numero_amostras_posteriores=configuracao["qnn"]["numero_amostras_mc"],
            modelo_base_ajustado=True,
            semente_aleatoria=semente,
        ),
    }
    for nome, metodo in metodos_credo.items():
        inicio_metodo = time.perf_counter()
        metodo.ajustar(divisao.x_treino, divisao.y_treino)
        metodo.calibrar(divisao.x_calibracao, divisao.y_calibracao)
        inicio = time.perf_counter()
        intervalos, decomposicao = metodo.prever(
            divisao.x_teste,
            retornar_decomposicao=True,
        )
        fim_metodo = time.perf_counter()
        tempo_predicao = fim_metodo - inicio
        fracao_epistemica_atipicos = float(
            np.mean(decomposicao["fracao_epistemica"][grupos.valores_atipicos])
        )
        fracao_epistemica_internos = float(
            np.mean(decomposicao["fracao_epistemica"][grupos.internos_decomposicao])
        )
        linhas.append(
            {
                **informacoes_comuns,
                **_avaliar(nome, intervalos, divisao, grupos, alpha),
                "modelo_base": "QNN",
                "segundos_ajuste_compartilhado": tempo_ajuste_qnn,
                "segundos_predicao": tempo_predicao,
                "segundos_total_metodo": calcular_tempo_total_metodo(
                    tempo_ajuste_qnn,
                    inicio_metodo,
                    fim_metodo,
                ),
                "fracao_epistemica_atipicos": fracao_epistemica_atipicos,
                "fracao_epistemica_internos": fracao_epistemica_internos,
            }
        )

    if incluir_catboost:
        config_catboost = configuracao.get("catboost", {})
        catboost = RegressorQuantilicoCatBoost(
            alpha=alpha,
            semente_aleatoria=semente,
            **config_catboost,
        )
        inicio = time.perf_counter()
        catboost.ajustar(divisao.x_treino, divisao.y_treino)
        tempo_ajuste_catboost = time.perf_counter() - inicio

        for nome, relativo in (("CQR-CatBoost", False), ("CQR-r-CatBoost", True)):
            metodo = RegressorCQR(
                catboost,
                alpha=alpha,
                relativo=relativo,
                modelo_base_ajustado=True,
            )
            inicio_metodo = time.perf_counter()
            metodo.calibrar(divisao.x_calibracao, divisao.y_calibracao)
            intervalos = metodo.prever(divisao.x_teste)
            fim_metodo = time.perf_counter()
            linhas.append(
                {
                    **informacoes_comuns,
                    **_avaliar(nome, intervalos, divisao, grupos, alpha),
                    "modelo_base": "CatBoost",
                    "segundos_ajuste_compartilhado": tempo_ajuste_catboost,
                    "segundos_total_metodo": calcular_tempo_total_metodo(
                        tempo_ajuste_catboost,
                        inicio_metodo,
                        fim_metodo,
                    ),
                }
            )

        inicio_metodo = time.perf_counter()
        uacqr = RegressorUACQR(
            catboost,
            alpha=alpha,
            tamanho_ensemble=configuracao["uacqr"]["tamanho_ensemble"],
            modelo_base_ajustado=True,
            semente_aleatoria=semente,
        ).ajustar(divisao.x_treino, divisao.y_treino)
        uacqr.calibrar(divisao.x_calibracao, divisao.y_calibracao)
        intervalos_uacqr = uacqr.prever(divisao.x_teste)
        fim_metodo = time.perf_counter()
        tempo_total_uacqr = calcular_tempo_total_metodo(
            tempo_ajuste_catboost,
            inicio_metodo,
            fim_metodo,
        )
        for nome, intervalos in intervalos_uacqr.items():
            linhas.append(
                {
                    **informacoes_comuns,
                    **_avaliar(f"{nome}-CatBoost", intervalos, divisao, grupos, alpha),
                    "modelo_base": "CatBoost",
                    "segundos_ajuste_compartilhado": tempo_ajuste_catboost,
                    "segundos_total_metodo": tempo_total_uacqr,
                }
            )

        if "epicscore" in configuracao:
            config_epic = dict(configuracao["epicscore"])
            config_epic["camadas_ocultas"] = tuple(config_epic["camadas_ocultas"])
            if conjunto == "WEC":
                tamanho_lote_epic = config_epic.pop(
                    "tamanho_lote_wec",
                    config_epic.get("tamanho_lote", 250),
                )
            elif len(X) >= 10_000:
                tamanho_lote_epic = config_epic.pop(
                    "tamanho_lote_grande",
                    config_epic.get("tamanho_lote", 130),
                )
            else:
                tamanho_lote_epic = config_epic.pop(
                    "tamanho_lote_pequeno",
                    config_epic.get("tamanho_lote", 40),
                )
            config_epic.pop("tamanho_lote_pequeno", None)
            config_epic.pop("tamanho_lote_grande", None)
            config_epic.pop("tamanho_lote_wec", None)
            config_epic["tamanho_lote"] = tamanho_lote_epic
            inicio_metodo = time.perf_counter()
            epic = RegressorEPICScore(
                catboost,
                alpha=alpha,
                modelo_base_ajustado=True,
                semente_aleatoria=semente,
                **config_epic,
            ).ajustar(divisao.x_treino, divisao.y_treino)
            epic.calibrar(divisao.x_calibracao, divisao.y_calibracao)
            intervalos_epic = epic.prever(divisao.x_teste)
            fim_metodo = time.perf_counter()
            linhas.append(
                {
                    **informacoes_comuns,
                    **_avaliar(
                        "EPICScore-CatBoost-MDN",
                        intervalos_epic,
                        divisao,
                        grupos,
                        alpha,
                    ),
                    "modelo_base": "CatBoost+MDN",
                    "segundos_ajuste_compartilhado": tempo_ajuste_catboost,
                    "segundos_total_metodo": calcular_tempo_total_metodo(
                        tempo_ajuste_catboost,
                        inicio_metodo,
                        fim_metodo,
                    ),
                }
            )
    return linhas


def executar_comparacao(
    caminho_configuracao: str | Path,
    *,
    diretorio_dados: str | Path = "dados/processados",
    diretorio_resultados: str | Path = "resultados",
    incluir_catboost: bool = True,
) -> Path:
    """Executa todas as repetições e salva resultados brutos e agregados."""
    configuracao = carregar_configuracao(caminho_configuracao)
    fixar_todas_sementes(int(configuracao["semente"]))
    diretorio_dados = Path(diretorio_dados)
    diretorio_resultados = Path(diretorio_resultados)
    instante = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    diretorio_execucao = (
        diretorio_resultados
        / "brutos"
        / f"{configuracao['nome_experimento']}_{instante}"
    )
    diretorio_execucao.mkdir(parents=True, exist_ok=False)
    (diretorio_execucao / "configuracao.yaml").write_text(
        yaml.safe_dump(configuracao, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    ambiente = {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
    }
    (diretorio_execucao / "ambiente.json").write_text(
        json.dumps(ambiente, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sementes = gerar_sementes_repeticoes(
        configuracao["semente"],
        configuracao["numero_repeticoes"],
    )
    (diretorio_execucao / "sementes.json").write_text(
        json.dumps(sementes, indent=2),
        encoding="utf-8",
    )
    todas_linhas = []
    for conjunto in configuracao["conjuntos"]:
        X, y = carregar_conjunto_processado(diretorio_dados / f"{conjunto}.csv")
        for repeticao, semente in enumerate(sementes):
            linhas = executar_repeticao(
                X,
                y,
                conjunto=conjunto,
                repeticao=repeticao,
                semente=int(semente),
                configuracao=configuracao,
                incluir_catboost=incluir_catboost,
            )
            todas_linhas.extend(linhas)
            pd.DataFrame(todas_linhas).to_csv(
                diretorio_execucao / "metricas.csv",
                index=False,
            )
    resumo = agregar_metricas(pd.DataFrame(todas_linhas))
    resumo.to_csv(diretorio_execucao / "resumo.csv", index=False)
    return diretorio_execucao
