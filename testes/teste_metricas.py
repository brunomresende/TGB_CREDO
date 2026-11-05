import numpy as np

from credal_cp_tgb.avaliacao.metricas import (
    avaliar_intervalos,
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


def teste_avaliacao_remove_intervalos_infinitos_e_preserva_indices_dos_grupos():
    y = np.array([0.0, 10.0, 5.0, 3.0])
    intervalos = np.array(
        [
            [-0.5, 0.5],
            [-np.inf, np.inf],
            [1.0, 3.0],
            [2.0, 4.0],
        ]
    )

    resultado = avaliar_intervalos(
        y,
        intervalos,
        alpha=0.1,
        indices_atipicos=np.array([0, 1]),
        indices_internos=np.array([2, 3]),
    )

    assert all(np.isfinite(valor) for valor in resultado.values())
    assert resultado["amc"] == 2 / 3
    assert resultado["aco"] == 1.0
    assert resultado["il"] == 5 / 3
    assert resultado["ilr"] == 0.5
