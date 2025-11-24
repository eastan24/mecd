# MECD: Mean–Event-Conditional Dispersion

This repository implements the Mean–Event-Conditional Dispersion (MECD) risk
model as described in:

> Baoerjin (2025), "Beyond Mean–Variance: A Mean–Event-Conditional Dispersion
> Framework for Portfolio Construction and Trading."

## Features

- EWMA-based forward mean & variance forecasts
- Event-conditional dispersion based on drawdown windows
- Full MECD score: reward minus underwater instability penalties
- Cross-sectional Z-scored signal for portfolio ranking
- Modular, testable, production-style Python package

## Project Layout

```text
mecd-project/
├── src/mecd/          # core library code
├── scripts/           # command-line entry points
├── tests/             # unit tests
├── notebooks/         # optional research notebooks
├── data/              # sample data locations
└── pyproject.toml     # dependencies and project metadata

