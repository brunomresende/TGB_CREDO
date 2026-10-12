import numpy as np

from credal_cp_tgb.conformal.escores import (
    escores_quantilicos,
    escores_quantilicos_relativos,
    limiar_conformal,
)


def teste_escores_quantilicos_permitirem_valores_negativos():
    escores = escores_quantilicos(
        [0.5, 2.0, 4.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 3.0],
    )
    np.testing.assert_allclose(escores, [-0.5, 1.0, 1.0])


def teste_escores_relativos_usarem_largura_intervalo():
    escores = escores_quantilicos_relativos(
        [3.0, 3.0],
        [0.0, 0.0],
        [2.0, 1.0],
    )
    np.testing.assert_allclose(escores, [0.5, 2.0])


def teste_limiar_ser_estatistica_ordem_amostra_finita():
    escores = np.arange(1, 11)
    # ceil(11 * .8) = 9
    assert limiar_conformal(escores, 0.2) == 9
