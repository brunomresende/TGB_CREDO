#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from credal_cp_tgb.dados.baixar import URLS_CONJUNTOS, baixar_conjunto

analisador = argparse.ArgumentParser(description="Baixa um conjunto de dados público.")
analisador.add_argument("conjunto", choices=sorted(URLS_CONJUNTOS))
analisador.add_argument("--destino", default=None)
argumentos = analisador.parse_args()
destino = argumentos.destino or (
    f"dados/brutos/{argumentos.conjunto}/"
    f"{Path(URLS_CONJUNTOS[argumentos.conjunto]).name}"
)
print(baixar_conjunto(argumentos.conjunto, destino))
