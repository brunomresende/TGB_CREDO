"""Leitura uniforme dos CSVs processados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_processed_dataset(path: str | Path, target: str = "target"):
    data = pd.read_csv(path)
    if target not in data:
        raise ValueError(f"A coluna alvo {target!r} nao existe em {path}.")
    return data.drop(columns=[target]).to_numpy(dtype=float), data[target].to_numpy(dtype=float)

