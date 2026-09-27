# algo-trading

Notebooks and a small backtesting toolkit for demoing algorithmic trading techniques.

## Setup

```bash
cd algo-trading
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/download_data.py
```

The last step saves the price history to `.data/historical_data.csv`. You can also
download it by hand from
[Google Drive](https://drive.google.com/file/d/1kAzenUMGXyuP7rnKpqisIflU3nscz9O6/view?usp=sharing)
and put it there.

## Layout

| path | contents |
|---|---|
| `notebooks/` | the demos |
| `src/algo_trading/backtest/` | data loading, simulator, null tests, plots |
| `scripts/` | data download |
| `requirements/` | dependencies (read by `pyproject.toml`) |

## Usage

```python
from algo_trading.backtest import backtest as bt, data, plots

plots.apply_theme()
panel = data.daily_panel()
print(panel.describe())
```
