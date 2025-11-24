import numpy as np
import pandas as pd
from mecd.mecd_core import compute_mecd_from_returns
from mecd.config import MECDConfig

def test_constant_returns():
    dates = pd.date_range("2020-01-01", periods=200)
    data = np.full((200, 3), 0.001)     # constant +0.1% returns
    df = pd.DataFrame(data, index=dates, columns=["A", "B", "C"])

    config = MECDConfig(D=63, d_star=0.10)
    raw, z = compute_mecd_from_returns(df, config)

    # No drawdown ever occurs for constant positive returns
    assert raw.isna().all().all(), "Raw MECD should be NaN for constant upward returns"
    assert z.isna().all().all(), "Z-score should be NaN since raw is NaN"
