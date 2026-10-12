"""Exemplo mínimo do envelope credal e de sua conformalização."""

from _compartilhado import plotar_intervalos, problema_pequeno, qnn_ajustada

from credal_cp_tgb.credal.gamma import GammaFixo
from credal_cp_tgb.metodos.credo import RegressorCREDO

divisao = problema_pequeno()
qnn = qnn_ajustada(divisao)
credo = RegressorCREDO(
    qnn,
    GammaFixo(0.1),
    modelo_base_ajustado=True,
    numero_amostras_posteriores=300,
)
credo.ajustar(divisao.x_treino, divisao.y_treino).calibrar(
    divisao.x_calibracao,
    divisao.y_calibracao,
)
bruto = credo.prever(divisao.x_teste, conformalizar=False)
final = credo.prever(divisao.x_teste)
print(f"Limiar conformal: {credo.limiar_:.4f}")
plotar_intervalos(
    divisao.x_teste,
    divisao.y_teste,
    bruto,
    "Envelope credal",
    "credo_bruto.png",
)
plotar_intervalos(
    divisao.x_teste,
    divisao.y_teste,
    final,
    "CREDO conformalizado",
    "credo_final.png",
)
