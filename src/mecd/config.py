from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MECDConfig:
    """
    Configuration for MECD model hyperparameters.

    H: forward horizon in days for mean & variance
    D: event window length in days for drawdown paths
    d_star: drawdown threshold (fraction, e.g. 0.10 = 10%)
    ewma_span: EWMA span in days for mean/variance
    lambda_sigma: penalty coefficient on expected variance
    lambda_EC: penalty coefficient on event-conditional dispersion
    """

    H: int = 21
    D: int = 63
    d_star: float = 0.10
    ewma_span: int = 60
    lambda_sigma: float = 1.0
    lambda_EC: float = 1.0


DEFAULT_CONFIG = MECDConfig()
