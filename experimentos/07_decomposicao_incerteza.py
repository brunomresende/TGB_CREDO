"""Mostra as parcelas aleatória, epistêmica e conformal do CREDO."""

import matplotlib.pyplot as plt
import numpy as np
from _compartilhado import problema_pequeno, qnn_ajustada

from credal_cp_tgb.credal.gamma import GammaAdaptativo
from credal_cp_tgb.metodos.credo import RegressorCREDO

divisao = problema_pequeno()
qnn = qnn_ajustada(divisao)
metodo = RegressorCREDO(
    qnn,
    GammaAdaptativo(),
    modelo_base_ajustado=True,
    numero_amostras_posteriores=300,
)
metodo.ajustar(divisao.x_treino, divisao.y_treino).calibrar(
    divisao.x_calibracao,
    divisao.y_calibracao,
)
_, parcelas = metodo.prever(divisao.x_teste, retornar_decomposicao=True)
ordem = np.argsort(divisao.x_teste[:, 0])
for chave in ("aleatoria", "epistemica", "folga_calibracao"):
    plt.plot(
        divisao.x_teste[ordem, 0],
        parcelas[chave][ordem],
        label=chave,
    )
plt.xlabel("x")
plt.ylabel("comprimento")
plt.legend()
plt.tight_layout()
plt.savefig("decomposicao_incerteza.png", dpi=200)
