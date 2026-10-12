"""Leitura uniforme dos CSVs processados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def carregar_conjunto_processado(caminho: str | Path, alvo: str = "alvo"):
    dados = pd.read_csv(caminho)
    if alvo not in dados:
        raise ValueError(f"A coluna alvo {alvo!r} não existe em {caminho}.")
    return (
        dados.drop(columns=[alvo]).to_numpy(dtype=float),
        dados[alvo].to_numpy(dtype=float),
    )
