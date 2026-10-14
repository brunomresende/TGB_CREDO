import numpy as np

from credal_cp_tgb.avaliacao.agregacao import agregar_metricas
from credal_cp_tgb.utilitarios.aleatoriedade import gerar_sementes_repeticoes


def teste_sementes_reproduzem_codigo_oficial():
    sementes = gerar_sementes_repeticoes(125, 5)
    assert sementes == [28887997, 2070063261, 247092450, 840617815, 547951871]


def teste_resumo_registra_dobro_do_erro_padrao():
    resultados = {
        "conjunto": ["exemplo"] * 4,
        "metodo": ["metodo"] * 4,
        "repeticao": range(4),
        "semente": range(4),
        "modelo_base": ["modelo"] * 4,
        "smis": [1.0, 2.0, 3.0, 4.0],
    }
    import pandas as pd

    resumo = agregar_metricas(pd.DataFrame(resultados)).iloc[0]
    esperado = 2 * np.std(resultados["smis"], ddof=1) / np.sqrt(4)
    assert np.isclose(resumo["dobro_erro_padrao"], esperado)
