import numpy as np
import pytest

from credal_cp_tgb.dados import criar_dados_ruido_variavel


def teste_dados_de_ruido_variavel_sao_reproduziveis_e_tem_regiao_escassa():
    primeiros = criar_dados_ruido_variavel(
        numero_amostras=1000,
        probabilidade_regiao_escassa=0.02,
        semente_aleatoria=7,
    )
    segundos = criar_dados_ruido_variavel(
        numero_amostras=1000,
        probabilidade_regiao_escassa=0.02,
        semente_aleatoria=7,
    )

    assert list(primeiros.columns) == ["x", "y", "media", "desvio_ruido"]
    assert len(primeiros) == 1000
    assert primeiros.equals(segundos)
    assert np.all(primeiros["x"].between(-1.0, 1.0))
    assert int(primeiros["x"].between(0.0, 0.4).sum()) == 20
    assert np.all(primeiros["desvio_ruido"] > 0)


@pytest.mark.parametrize(
    ("argumentos", "mensagem"),
    [
        ({"numero_amostras": 1}, "numero_amostras"),
        ({"desvio_base": 0}, "desvio_base"),
        ({"probabilidade_regiao_escassa": 1.1}, "probabilidade_regiao_escassa"),
    ],
)
def teste_dados_de_ruido_variavel_validam_parametros(argumentos, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        criar_dados_ruido_variavel(**argumentos)


def teste_dados_de_ruido_variavel_aceitam_apenas_regiao_escassa():
    dados = criar_dados_ruido_variavel(
        numero_amostras=10,
        probabilidade_regiao_escassa=1.0,
        semente_aleatoria=7,
    )
    assert np.all(dados["x"].between(0.0, 0.4))


def teste_dados_de_ruido_variavel_preservam_sequencia_da_figura_original():
    dados = criar_dados_ruido_variavel(
        numero_amostras=1500,
        semente_aleatoria=42,
    )
    np.testing.assert_allclose(
        dados.iloc[:3].to_numpy(),
        [
            [-0.7409701574343917, -0.3354508915047392, -0.40381813364643093, 0.1],
            [0.450515320830273, 1.8182738144535713, 1.837247330839805, 0.3421897032233009],
            [0.6574750183038587, 1.8014277133976413, 1.2132678511218118, 0.5068187312758281],
        ],
    )
