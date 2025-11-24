import numpy as np
import pandas as pd
from mecd.mecd_ecdisp import compute_event_conditional_dispersion
from mecd.config import MECDConfig

def test_ecdisp_no_events():
    dates = pd.date_range("2020-01-01", periods=200)
    df = pd.DataFrame({
        "A": np.random.normal(0.001, 0.002, 200)  # small mild returns
    }, index=dates)

    config = MECDConfig(D=63, d_star=0.30)  # large threshold = almost no events
    ec = compute_event_conditional_dispersion(df, config)

    # All values should be NaN due to no event windows
    assert ec["A"].isna().all()
