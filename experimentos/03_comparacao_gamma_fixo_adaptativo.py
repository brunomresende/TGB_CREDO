"""Compara CREDO-QNN fixo e adaptativo no mesmo modelo treinado."""

from _compartilhado import plotar_intervalos, problema_pequeno, qnn_ajustada

from credal_cp_tgb.credal.gamma import GammaAdaptativo, GammaFixo
from credal_cp_tgb.metodos.credo import RegressorCREDO

divisao = problema_pequeno()
qnn = qnn_ajustada(divisao)
estrategias = {"fixo": GammaFixo(0.1), "adaptativo": GammaAdaptativo()}
for nome, estrategia in estrategias.items():
    metodo = RegressorCREDO(
        qnn,
        estrategia,
        modelo_base_ajustado=True,
        numero_amostras_posteriores=300,
    )
    metodo.ajustar(divisao.x_treino, divisao.y_treino).calibrar(
        divisao.x_calibracao,
        divisao.y_calibracao,
    )
    intervalos = metodo.prever(divisao.x_teste)
    plotar_intervalos(
        divisao.x_teste,
        divisao.y_teste,
        intervalos,
        f"CREDO {nome}",
        f"credo_{nome}.png",
    )
