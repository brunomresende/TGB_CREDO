#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from credal_cp_tgb.avaliacao.agregacao import agregar_metricas

analisador = argparse.ArgumentParser(description="Agrega as métricas das repetições.")
analisador.add_argument("arquivo_metricas")
analisador.add_argument("--saida", default=None)
argumentos = analisador.parse_args()
origem = Path(argumentos.arquivo_metricas)
saida = Path(argumentos.saida) if argumentos.saida else origem.with_name("resumo.csv")
agregar_metricas(pd.read_csv(origem)).to_csv(saida, index=False)
print(saida)
