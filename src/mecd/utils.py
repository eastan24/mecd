from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def build_wealth_from_returns(returns: np.ndarray, initial: float = 1.0) -> np.ndarray:
    """Convert simple returns into a wealth curve.

    Parameters
    ----------
    returns : np.ndarray
        Array of simple returns, shape (T,).
    initial : float
        Initial wealth level.

    Returns
    -------
    wealth : np.ndarray
        Wealth process, same length as returns.
    """
    wealth = initial * np.cumprod(1.0 + returns)
    return wealth


def max_drawdown_from_wealth(wealth: np.ndarray) -> float:
    """Compute maximum drawdown from a wealth curve.

    Drawdown(t) = (peak_to_date - wealth[t]) / peak_to_date
    """
    if wealth.size == 0:
        return np.nan

    peak = np.maximum.accumulate(wealth)
    # avoid division-by-zero if wealth starts at zero; unrealistic but guard anyway
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (peak - wealth) / peak
    dd = np.where(np.isfinite(dd), dd, 0.0)
    return float(dd.max())


def safe_cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional Z-score per timestamp (row-wise).

    Z_{i,t} = (x_{i,t} - mean_t) / std_t

    If std_t == 0 or nan, all Z-scores for that row are NaN.
    """
    mean = df.mean(axis=1, skipna=True)
    std = df.std(axis=1, ddof=0, skipna=True)

    z = df.sub(mean, axis=0)
    std_nonzero = std.replace(0.0, np.nan)
    z = z.div(std_nonzero, axis=0)
    return z


def load_returns_csv(path: str) -> pd.DataFrame:
    """Helper to load returns from a CSV file.

    Expects:
        - first column: date (parsed as datetime)
        - other columns: asset identifiers
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Ensure sorted by time
    df = df.sort_index()
    return df


def split_train_test_by_date(
    df: pd.DataFrame,
    split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple date-based split for examples/backtests."""
    split_ts = pd.to_datetime(split_date)
    train = df[df.index <= split_ts]
    test = df[df.index > split_ts]
    return train, test
