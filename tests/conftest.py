from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_returns() -> pd.DataFrame:
    """Small toy returns DataFrame for unit tests."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    data = np.random.normal(loc=0.0005, scale=0.01, size=(200, 5))
    cols = [f"Asset_{i}" for i in range(5)]
    returns = pd.DataFrame(data, index=dates, columns=cols)
    return returns
