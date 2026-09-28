"""Entry and exit rules that feed ``backtest.slot_weights``.

Each rule returns boolean panels on the price grid: ``enter`` (may buy) and
``exit`` (must sell). They are evaluated on each bar's close; the simulator
trades on the next bar, so there is no look-ahead.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["momentum", "momentum_thresholds"]


def momentum(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Return over the last ``lookback`` bars: ``close / close.shift(lookback) - 1``.

    Args:
        close: Wide close-price panel.
        lookback: Window in bars.

    Returns:
        Momentum panel; NaN until a token has ``lookback`` bars of history.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1 bar")
    return close / close.shift(lookback) - 1.0


def momentum_thresholds(close: pd.DataFrame, lookback: int, buy: float,
                        sell: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Buy above one momentum level, sell below a lower one.

    The gap between ``buy`` and ``sell`` is a hysteresis band: a name bought at
    +20% is kept while it still has +0% (say), so a position isn't flipped every
    time momentum wobbles around a single level.

    Args:
        close: Wide close-price panel.
        lookback: Momentum window in bars.
        buy: Enter when momentum is above this (0.2 = +20% over the window).
        sell: Exit when momentum is below this. Must not exceed ``buy``.

    Returns:
        ``(enter, exit, mom)``: boolean entry and exit panels, and the momentum
        panel itself. Tokens without enough history can't enter; a held token
        whose price goes missing is sold.
    """
    if sell > buy:
        raise ValueError(f"sell threshold ({sell}) must not exceed buy threshold ({buy})")
    mom = momentum(close, lookback)
    enter = mom > buy
    exit = (mom < sell) | mom.isna()
    return enter, exit, mom
