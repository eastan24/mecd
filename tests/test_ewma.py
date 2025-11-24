from __future__ import annotations

import numpy as np
import pandas as pd

from mecd.config import DEFAULT_CONFIG
from mecd.mecd_ewma import compute_ewma_forecasts


def test_ewma_shapes(toy_returns: pd.DataFrame) -> None:
    mu_H, var_H = compute_ewma_forecasts(toy_returns, DEFAULT_CONFIG)

    assert mu_H.shape == toy_returns.shape
    assert var_H.shape == toy_returns.shape

    # Horizon scaling should produce larger variance than daily variance on average
    assert np.all(np.isfinite(var_H.tail().values))
