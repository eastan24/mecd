from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import MECDConfig
from .mecd_ewma import compute_ewma_forecasts
from .mecd_ecdisp import compute_event_conditional_dispersion
from .utils import safe_cross_sectional_zscore


def compute_raw_mecd(
    mu_H: pd.DataFrame,
    var_H: pd.DataFrame,
    ec_disp: pd.DataFrame,
    config: MECDConfig,
) -> pd.DataFrame:
    """Combine components into raw MECD score.

    MECD_{i,t} = μ^{(H)}_{i,t}
                 - λσ * σ^{2(H)}_{i,t}
                 - λEC * ECDisp_{i,t}
    """
    raw = (
        mu_H
        - config.lambda_sigma * var_H
        - config.lambda_EC * ec_disp
    )
    return raw


def compute_mecd_signal(
    returns: pd.DataFrame,
    config: MECDConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full MECD pipeline: from returns to raw MECD & Z-scored signal.

    Parameters
    ----------
    returns : DataFrame
        Daily simple returns, index=dates, columns=assets.
    config : MECDConfig
        MECD configuration.

    Returns
    -------
    raw_mecd : DataFrame
        Raw MECD per date & asset.
    mecd_z : DataFrame
        Cross-sectional Z-scored MECD per date & asset.
    """
    # 1) Forward mean and variance
    mu_H, var_H = compute_ewma_forecasts(returns, config)

    # 2) Event-conditional dispersion
    ec_disp = compute_event_conditional_dispersion(returns, config)

    # 3) Combine into raw MECD
    raw_mecd = compute_raw_mecd(mu_H, var_H, ec_disp, config)

    # 4) Cross-sectional Z-score per date
    mecd_z = safe_cross_sectional_zscore(raw_mecd)

    return raw_mecd, mecd_z
