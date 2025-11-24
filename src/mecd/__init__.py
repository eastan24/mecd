"""
MECD: Mean–Event-Conditional Dispersion.

Public API:
    - compute_mecd_from_returns
    - DEFAULT_CONFIG
"""

from .config import DEFAULT_CONFIG, MECDConfig
from .mecd_core import compute_mecd_from_returns

__all__ = [
    "MECDConfig",
    "DEFAULT_CONFIG",
    "compute_mecd_from_returns",
]
