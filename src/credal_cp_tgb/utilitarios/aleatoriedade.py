from __future__ import annotations

import random

import numpy as np
import torch


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
