"""Tabela pequena de CQR-QNN e CREDO-QNN no cenário sintético."""

import pandas as pd
from _compartilhado import problema_pequeno, qnn_ajustada

from credal_cp_tgb.avaliacao.metricas import avaliar_intervalos
from credal_cp_tgb.credal.gamma import GammaAdaptativo, GammaFixo
from credal_cp_tgb.metodos.cqr import RegressorCQR
from credal_cp_tgb.metodos.credo import RegressorCREDO

divisao = problema_pequeno()
qnn = qnn_ajustada(divisao)
metodos = {
    "CQR-QNN": RegressorCQR(qnn, modelo_base_ajustado=True),
    "CQR-r-QNN": RegressorCQR(qnn, relativo=True, modelo_base_ajustado=True),
    "CREDO-QNN": RegressorCREDO(
        qnn,
        GammaFixo(),
        modelo_base_ajustado=True,
        numero_amostras_posteriores=300,
    ),
    "CREDO-QNN-Adaptativo": RegressorCREDO(
        qnn,
        GammaAdaptativo(),
        modelo_base_ajustado=True,
        numero_amostras_posteriores=300,
    ),
}
linhas = []
for nome, metodo in metodos.items():
    if isinstance(metodo, RegressorCREDO):
        metodo.ajustar(divisao.x_treino, divisao.y_treino)
    metodo.calibrar(divisao.x_calibracao, divisao.y_calibracao)
    linhas.append(
        {
            "metodo": nome,
            **avaliar_intervalos(
                divisao.y_teste,
                metodo.prever(divisao.x_teste),
                alpha=0.1,
            ),
        }
    )
print(pd.DataFrame(linhas).to_string(index=False))
