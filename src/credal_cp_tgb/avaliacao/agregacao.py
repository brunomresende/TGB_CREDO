"""Agregação das repetições sem esconder os resultados brutos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def agregar_metricas(resultados: pd.DataFrame) -> pd.DataFrame:
    colunas_identificadoras = {
        "conjunto",
        "repeticao",
        "semente",
        "metodo",
        "modelo_base",
    }
    colunas_metricas = [
        coluna
        for coluna in resultados.select_dtypes(include=[np.number]).columns
        if coluna not in colunas_identificadoras
    ]
    linhas = []
    for (conjunto, metodo), grupo in resultados.groupby(
        ["conjunto", "metodo"],
        sort=False,
    ):
        for metrica in colunas_metricas:
            valores = grupo[metrica].dropna().to_numpy(dtype=float)
            if not len(valores):
                continue
            desvio_padrao = float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0
            erro_padrao = desvio_padrao / np.sqrt(len(valores))
            linhas.append(
                {
                    "conjunto": conjunto,
                    "metodo": metodo,
                    "metrica": metrica,
                    "media": float(np.mean(valores)),
                    "desvio_padrao": desvio_padrao,
                    "erro_padrao": erro_padrao,
                    "dobro_erro_padrao": 2 * erro_padrao,
                    "semilargura_ic95": 1.96 * erro_padrao,
                    "numero_repeticoes": len(valores),
                }
            )
    return pd.DataFrame(linhas)
