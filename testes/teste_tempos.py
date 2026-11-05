import numpy as np
import pandas as pd

from credal_cp_tgb.avaliacao.agregacao import agregar_metricas
from credal_cp_tgb.utilitarios.tempo import calcular_tempo_total_metodo


def teste_tempo_total_inclui_ajuste_compartilhado_e_etapas_especificas():
    total = calcular_tempo_total_metodo(
        segundos_ajuste_compartilhado=7.5,
        inicio_etapas_especificas=10.0,
        fim_etapas_especificas=12.25,
    )
    assert total == 9.75


def teste_resumo_calcula_tempo_medio_por_metodo():
    resultados = pd.DataFrame(
        {
            "conjunto": ["exemplo"] * 3,
            "metodo": ["metodo"] * 3,
            "repeticao": range(3),
            "semente": range(3),
            "modelo_base": ["modelo"] * 3,
            "segundos_total_metodo": [1.0, 2.0, 3.0],
        }
    )

    resumo = agregar_metricas(resultados)
    tempo = resumo.loc[resumo["metrica"] == "segundos_total_metodo"].iloc[0]

    assert tempo["media"] == 2.0
    assert tempo["numero_repeticoes"] == 3
    assert np.isclose(tempo["desvio_padrao"], 1.0)
