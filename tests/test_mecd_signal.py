from __future__ import annotations

import numpy as np
import pandas as pd

from mecd.config import DEFAULT_CONFIG
from mecd.mecd_core import compute_mecd_from_returns


def test_mecd_shapes(toy_returns: pd.DataFrame) -> None:
    raw_mecd, mecd_z = compute_mecd_from_returns(toy_returns, DEFAULT_CONFIG)

    assert raw_mecd.shape == toy_returns.shape
    assert mecd_z.shape == toy_returns.shape

    # Some non-trivial finite values should exist
    assert np.isfinite(raw_mecd.values).sum() > 0
    # Z-scores may have NaNs at edges, but not all
    assert np.isfinite(mecd_z.values).sum() > 0
