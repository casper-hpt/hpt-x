"""A small, honest backtesting toolkit for long-only crypto portfolios.

Modules:
    config: Constants (fees, bar arithmetic, data path) and frozen config dataclasses.
    data: Load ``.data/historical_data.csv`` into gap-free price ``Panel``s.
    backtest: The band-rebalanced portfolio simulator and performance stats.
    nulls: Randomisation tests that decide whether a result means anything.
    plots: Matplotlib theme and composable charts.

Typical notebook setup::

    from algo_trading.backtest import backtest as bt, config, data, nulls, plots
    plots.apply_theme()
    panel = data.daily_panel()
"""
from . import backtest, config, data, nulls, plots

__all__ = ["backtest", "config", "data", "nulls", "plots"]
