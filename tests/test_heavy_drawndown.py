import numpy as np
import pandas as pd
from mecd.mecd_core import compute_mecd_from_returns
from mecd.config import MECDConfig

def test_heavy_drawdown():
    dates = pd.date_range("2020-01-01", periods=200)

    # Simulate a big crash
    # First 150 days normal, last 50 days -5% return each day
    r = np.concatenate([
        np.random.normal(0.0005, 0.01, 150),
        np.full(50, -0.05)
    ])

    df = pd.DataFrame({
        "CRASH_ASSET": r,
        "STEADY": np.zeros(200)
    }, index=dates)

    raw, z = compute_mecd_from_returns(df)

    # CRASH_ASSET should have more negative MECD than STEADY
    # (since only CRASH_ASSET has drawdown windows)
    both = raw.dropna()
    if len(both) > 0:
        assert (both["CRASH_ASSET"] < both["STEADY"]).any(), \
            "Crash asset should show more negative MECD"

