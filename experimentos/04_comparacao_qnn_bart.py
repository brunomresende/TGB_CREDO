"""Compara os modelos-base QNN e BART; requer o extra ``bart``."""

from _compartilhado import plotar_intervalos, problema_pequeno, qnn_ajustada

from credal_cp_tgb.credal.gamma import GammaAdaptativo
from credal_cp_tgb.metodos.credo import RegressorCREDO
from credal_cp_tgb.modelos.bart import BARTHeteroscedastico

divisao = problema_pequeno()
modelos = {
    "QNN": qnn_ajustada(divisao),
    "BART": BARTHeteroscedastico(amostras=500, cadeias=2, nucleos=2).ajustar(
        divisao.x_treino,
        divisao.y_treino,
    ),
}
for nome, modelo in modelos.items():
    metodo = RegressorCREDO(
        modelo,
        GammaAdaptativo(),
        modelo_base_ajustado=True,
        numero_amostras_posteriores=300,
    )
    metodo.ajustar(divisao.x_treino, divisao.y_treino).calibrar(
        divisao.x_calibracao,
        divisao.y_calibracao,
    )
    plotar_intervalos(
        divisao.x_teste,
        divisao.y_teste,
        metodo.prever(divisao.x_teste),
        f"CREDO-{nome}",
        f"credo_{nome.lower()}.png",
    )
