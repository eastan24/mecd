from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import MECDConfig, DEFAULT_CONFIG
from .mecd_signal import compute_mecd_signal


def compute_mecd_from_returns(
    returns: pd.DataFrame,
    config: MECDConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """High-level entry point for MECD computation.

    Parameters
    ----------
    returns : DataFrame
        Daily simple returns, index=dates, columns=assets.
    config : MECDConfig, optional
        Configuration. If None, uses DEFAULT_CONFIG.

    Returns
    -------
    raw_mecd : DataFrame
        Raw MECD per date & asset.
    mecd_z : DataFrame
        Cross-sectional Z-scored MECD per date & asset.
    """
    if config is None:
        config = DEFAULT_CONFIG

    raw_mecd, mecd_z = compute_mecd_signal(returns, config)
    return raw_mecd, mecd_z
