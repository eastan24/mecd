import numpy as np
import pandas as pd
from mecd.utils import safe_cross_sectional_zscore

def test_zscore_correctness():
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0],
        "B": [2.0, 2.0, 2.0],
        "C": [3.0, 2.0, 1.0]
    })

    z = safe_cross_sectional_zscore(df)

    # Row 0 population z-score: [-1.2247, 0, 1.2247]
    assert np.allclose(
        z.iloc[0].values,
        [-1.22474487, 0.0, 1.22474487],
        atol=1e-6
    )

    # Row 1: std = 0 -> NaNs
    assert z.iloc[1].isna().all()

    # Row 2 population z-score: [1.2247, 0, -1.2247]
    assert np.allclose(
        z.iloc[2].values,
        [1.22474487, 0.0, -1.22474487],
        atol=1e-6
    )
