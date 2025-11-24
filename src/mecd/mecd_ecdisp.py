from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MECDConfig
from .utils import build_wealth_from_returns, max_drawdown_from_wealth


def compute_event_conditional_dispersion_1d(
    returns: pd.Series,
    config: MECDConfig,
) -> pd.Series:
    """Event-conditional dispersion for a single asset.

    For each starting date t, look forward D days:
      1. Build wealth path from simple returns.
      2. Compute max drawdown (MDD) for that path.
      3. If MDD >= d_star, record variance of returns in that window.

    Parameters
    ----------
    returns : Series
        Daily returns for one asset.
    config : MECDConfig
        Contains D and d_star.

    Returns
    -------
    ec_disp : Series
        Event-conditional dispersion aligned at window start t.
        NaN if no drawdown event occurred in that window.
    """
    D = config.D
    d_star = config.d_star

    r = returns.values
    T = len(r)
    ec_disp = np.full(T, np.nan)

    if T < D:
        # Not enough data for even one window
        return pd.Series(ec_disp, index=returns.index)

    for t in range(T - D + 1):
        window = r[t : t + D]

        # Wealth path
        wealth = build_wealth_from_returns(window)

        # Max drawdown
        mdd = max_drawdown_from_wealth(wealth)

        if mdd >= d_star:
            # Use sample variance (ddof=1) for stability
            ec_disp[t] = np.var(window, ddof=1)

    return pd.Series(ec_disp, index=returns.index)


def compute_event_conditional_dispersion(
    returns: pd.DataFrame,
    config: MECDConfig,
) -> pd.DataFrame:
    """Vectorized wrapper: event-conditional dispersion for each asset.

    Parameters
    ----------
    returns : DataFrame
        Daily returns, index=dates, columns=assets.
    config : MECDConfig
        Contains D and d_star.

    Returns
    -------
    ec_disp : DataFrame
        Event-conditional dispersion per asset.
    """
    return returns.apply(lambda col: compute_event_conditional_dispersion_1d(col, config))
