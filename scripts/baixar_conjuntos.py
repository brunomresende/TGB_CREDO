#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

URLS_CONJUNTOS = {
    "airfoil": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    "protein": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
    "winered": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "winewhite": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}


def baixar_conjunto(nome: str, destino: str | Path) -> Path:
    """Baixa um conjunto público sem modificar arquivos já processados."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URLS_CONJUNTOS[nome], destino)
    return destino

analisador = argparse.ArgumentParser(description="Baixa um conjunto de dados público.")
analisador.add_argument("conjunto", choices=sorted(URLS_CONJUNTOS))
analisador.add_argument("--destino", default=None)
argumentos = analisador.parse_args()
destino = argumentos.destino or (
    f"dados/brutos/{argumentos.conjunto}/"
    f"{Path(URLS_CONJUNTOS[argumentos.conjunto]).name}"
)
print(baixar_conjunto(argumentos.conjunto, destino))
