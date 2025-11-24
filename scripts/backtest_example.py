#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import pandas as pd

from mecd.config import DEFAULT_CONFIG
from mecd.mecd_core import compute_mecd_from_returns
from mecd.utils import load_returns_csv, split_train_test_by_date


def long_short_backtest(
    returns: pd.DataFrame,
    mecd_z: pd.DataFrame,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> pd.Series:
    """Toy long-short backtest:
    - Each day, long top q MECD assets, short bottom q
    - Equal-weight within long and short
    """
    aligned_returns = returns.reindex_like(mecd_z)

    daily_pnl = []
    index = []

    for date, scores in mecd_z.iterrows():
        r = aligned_returns.loc[date]
        scores = scores.dropna()
        r = r[scores.index].dropna()

        if len(scores) < 5:
            continue

        n = len(scores)
        k_long = max(1, int(n * top_quantile))
        k_short = max(1, int(n * bottom_quantile))

        scores_sorted = scores.sort_values(ascending=False)
        long_assets = scores_sorted.index[:k_long]
        short_assets = scores_sorted.index[-k_short:]

        r_long = r[long_assets].mean()
        r_short = r[short_assets].mean()

        pnl = 0.5 * r_long - 0.5 * r_short
        daily_pnl.append(pnl)
        index.append(date)

    pnl_series = pd.Series(daily_pnl, index=index).sort_index()
    return pnl_series


def main() -> None:
    # Example: assumes data/returns.csv exists
    returns = load_returns_csv("data/returns.csv")

    # Split for demonstration
    train, test = split_train_test_by_date(returns, "2020-12-31")

    # Fit & compute on full data (for now we don’t fit parameters)
    raw_mecd, mecd_z = compute_mecd_from_returns(returns, DEFAULT_CONFIG)

    pnl = long_short_backtest(returns, mecd_z)
    cum_pnl = (1.0 + pnl).cumprod()

    print("Backtest summary:")
    print(f"Total return: {cum_pnl.iloc[-1] - 1.0:.2%}")
    print(f"Annualized mean: {pnl.mean() * 252:.2%}")
    print(f"Annualized vol: {pnl.std(ddof=1) * np.sqrt(252):.2%}")
    print(f"Sharpe ~ {pnl.mean() / pnl.std(ddof=1) * np.sqrt(252):.2f}")


if __name__ == "__main__":
    main()
