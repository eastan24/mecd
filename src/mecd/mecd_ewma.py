from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import MECDConfig


def compute_ewma_forecasts(
    returns: pd.DataFrame,
    config: MECDConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute EWMA-based forward mean and variance forecasts.

    Parameters
    ----------
    returns : DataFrame
        Daily simple returns, index=dates, columns=assets.
    config : MECDConfig
        Configuration with H (horizon) and ewma_span.

    Returns
    -------
    mu_H : DataFrame
        H-horizon EWMA mean forecast per asset.
    var_H : DataFrame
        H-horizon EWMA variance forecast per asset.
    """
    H = config.H
    span = config.ewma_span

    # Daily EWMA mean
    #  returns: dataset input
    #  ewm(span=span, adjust=False): calculates EWMA using recursive EWMA formula
        # includes: decay factor (lambda, get from span),  actual data point from returns at time t (x_t),  EWMA estimate at time t (m_t, smoothed value)
    mu_daily = returns.ewm(span=span, adjust=False).mean()

    # EWMA variance of residuals
    resid = returns - mu_daily
    # σt2​=λσt−12​+(1−λ)(residualt2​)
    var_daily = resid.pow(2).ewm(span=span, adjust=False).mean()

    # Horizon scaling
    mu_H = H * mu_daily
    var_H = H * var_daily

    return mu_H, var_H
