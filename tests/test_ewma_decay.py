import numpy as np
import pandas as pd
from mecd.mecd_ewma import compute_ewma_forecasts
from mecd.config import MECDConfig

def test_ewma_decay():
    # Construct a simple increasing return series
    dates = pd.date_range("2020-01-01", periods=100)
    r = np.linspace(0.0, 0.1, 100)  # rising returns

    df = pd.DataFrame({"A": r}, index=dates)

    mu_H, _ = compute_ewma_forecasts(df, MECDConfig())

    # Last expected return should be > earlier expected return
    assert mu_H["A"].iloc[-1] > mu_H["A"].iloc[10]
