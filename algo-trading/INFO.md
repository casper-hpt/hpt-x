# algo-trading

Notebooks and a small backtesting toolkit for demoing algorithmic trading techniques.

## Setup

```bash
cd algo-trading
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/download_data.py --dataset all
```

The last step saves both price datasets to `.data/`. Use `--dataset historical` or
`--dataset ohlcv_15m` to fetch just one, and `--force` to re-download. You can also
download them by hand and put them in `.data/`:

| dataset | file | size | link |
|---|---|---|---|
| `historical` | `historical_data.csv` | ~8 MB | [Google Drive](https://drive.google.com/file/d/1kAzenUMGXyuP7rnKpqisIflU3nscz9O6/view?usp=sharing) |
| `ohlcv_15m` | `ohlcv_15m.csv` | ~290 MB | [Google Drive](https://drive.google.com/file/d/1Axd2JvTwBAZ2rUX-vLHRWLmhoeQ7HUX0/view?usp=sharing) |

The dataset the notebooks read is chosen in [config/config.yaml](config/config.yaml):
set `dataset:` to one of the entries under `datasets` and restart the kernel. Each
entry has a `path` and a `format` (`long` for `historical_data.csv`, `ohlcv` for
15-minute OHLCV bars, which are resampled to hourly and daily bars on load).

Optional: `make rust` builds the Rust engines for much faster backtests and videos,
see [rust/README.md](rust/README.md).

## Layout

| path | contents |
|---|---|
| `notebooks/` | the demos |
| `src/algo_trading/backtest/` | data loading, simulator, null tests, plots |
| `rust/` | optional Rust engines for the simulator and video renderer |
| `config/config.yaml` | which dataset to load |
| `scripts/` | data download |
| `requirements/` | dependencies (read by `pyproject.toml`) |

## Usage

```python
from algo_trading.backtest import backtest as bt, data, plots

plots.apply_theme()
panel = data.daily_panel()
print(panel.describe())
```
