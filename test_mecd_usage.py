import pandas as pd
from mecd.mecd_core import compute_mecd_from_returns
from mecd.config import DEFAULT_CONFIG

# load your returns
returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)

# run MECD
raw_mecd, mecd_z = compute_mecd_from_returns(
    returns=returns,
    config=DEFAULT_CONFIG,
)

print(mecd_z.tail())
