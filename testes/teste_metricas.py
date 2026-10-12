import numpy as np

from credal_cp_tgb.avaliacao.metricas import (
    cobertura_marginal,
    comprimento_medio_intervalo,
    escore_medio_intervalo_escalonado,
    razao_comprimento_atipicos_internos,
)


def teste_metricas_basicas_intervalos():
    y = np.array([0.0, 2.0])
    inferior = np.array([-1.0, 0.0])
    superior = np.array([1.0, 1.0])
    assert cobertura_marginal(y, inferior, superior) == 0.5
    assert comprimento_medio_intervalo(inferior, superior) == 1.5
    assert escore_medio_intervalo_escalonado(y, inferior, superior, 0.1) == 11.5


def teste_ilr():
    inferior = np.zeros(4)
    superior = np.array([1.0, 1.0, 2.0, 2.0])
    assert razao_comprimento_atipicos_internos(
        inferior,
        superior,
        [2, 3],
        [0, 1],
    ) == 2.0
