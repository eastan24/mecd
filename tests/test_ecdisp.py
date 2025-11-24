from __future__ import annotations

import numpy as np
import pandas as pd

from mecd.config import DEFAULT_CONFIG
from mecd.mecd_ecdisp import (
    compute_event_conditional_dispersion_1d,
    compute_event_conditional_dispersion,
)


def test_ecdisp_1d_length(toy_returns: pd.DataFrame) -> None:
    asset = toy_returns.columns[0]
    series = toy_returns[asset]

    ec = compute_event_conditional_dispersion_1d(series, DEFAULT_CONFIG)
    assert isinstance(ec, pd.Series)
    assert len(ec) == len(series)


def test_ecdisp_df_shape(toy_returns: pd.DataFrame) -> None:
    ec_df = compute_event_conditional_dispersion(toy_returns, DEFAULT_CONFIG)
    assert ec_df.shape == toy_returns.shape
