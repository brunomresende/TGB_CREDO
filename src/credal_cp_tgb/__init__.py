"""API pública do projeto credal_cp_tgb."""

from .credal.gamma import GammaAdaptativo, GammaFixo
from .metodos.cqr import RegressorCQR
from .metodos.credo import RegressorCREDO

__all__ = ["GammaAdaptativo", "GammaFixo", "RegressorCQR", "RegressorCREDO"]
__version__ = "0.1.0"
