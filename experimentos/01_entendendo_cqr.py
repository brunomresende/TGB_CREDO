"""Exemplo mínimo: quantis da QNN, escores e expansão conformal."""

from _compartilhado import plotar_intervalos, problema_pequeno, qnn_ajustada

from credal_cp_tgb.metodos.cqr import RegressorCQR

divisao = problema_pequeno()
qnn = qnn_ajustada(divisao)
cqr = RegressorCQR(qnn, modelo_base_ajustado=True).calibrar(
    divisao.x_calibracao,
    divisao.y_calibracao,
)
print(f"Limiar conformal: {cqr.limiar_:.4f}")
plotar_intervalos(
    divisao.x_teste,
    divisao.y_teste,
    cqr.prever(divisao.x_teste),
    "CQR-QNN",
    "cqr_qnn.png",
)
