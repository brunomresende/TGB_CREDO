from __future__ import annotations

import random

import numpy as np
import torch


def gerar_sementes_repeticoes(semente_inicial: int, numero_repeticoes: int) -> list[int]:
    """Reproduz exatamente a sequência de sementes usada no código oficial."""
    gerador = np.random.RandomState(semente_inicial)
    return gerador.randint(
        0,
        np.iinfo(np.int32).max,
        size=numero_repeticoes,
    ).tolist()


def fixar_todas_sementes(
    semente: int,
    *,
    torch_deterministico: bool = False,
) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)
    if torch_deterministico:
        torch.use_deterministic_algorithms(True)
