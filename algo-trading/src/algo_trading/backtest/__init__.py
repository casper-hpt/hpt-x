"""A small, honest backtesting toolkit for long-only crypto portfolios.

Modules:
    config: Constants (fees, bar arithmetic, data path) and frozen config dataclasses.
    data: Load ``.data/historical_data.csv`` into gap-free price ``Panel``s.
    backtest: The band-rebalanced portfolio simulator and performance stats.
    signals: Entry/exit rules (momentum thresholds) for ``backtest.slot_weights``.
    nulls: Randomisation tests that decide whether a result means anything.
    plots: Matplotlib theme and composable charts.
    animate: Equity-race videos (mp4/gif) for sharing a result.

Typical notebook setup::

    from algo_trading.backtest import backtest as bt, config, data, nulls, plots
    plots.apply_theme()
    panel = data.daily_panel()
"""
from . import animate, backtest, config, data, nulls, plots, signals

__all__ = ["animate", "backtest", "config", "data", "nulls", "plots", "signals"]
