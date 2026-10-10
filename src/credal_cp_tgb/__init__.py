"""API publica do projeto credal_cp_tgb."""

from .credal.gamma import AdaptiveGamma, FixedGamma
from .methods.cqr import CQRRegressor
from .methods.credo import CredalRegressor

__all__ = ["AdaptiveGamma", "CQRRegressor", "CredalRegressor", "FixedGamma"]
__version__ = "0.1.0"

