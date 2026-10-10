"""Catalogo das fontes usadas no benchmark.

O download e o processamento permanecem separados para que arquivos brutos nunca
sejam alterados silenciosamente. Datasets com licenca ou fluxo manual levantam uma
mensagem explicativa.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


DATASET_URLS = {
    "airfoil": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    "protein": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
    "winered": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "winewhite": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}


def download_dataset(name: str, destination: str | Path) -> Path:
    if name not in DATASET_URLS:
        raise ValueError(
            f"{name!r} requer download/processamento manual; consulte data/README.md."
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATASET_URLS[name], destination)
    return destination

