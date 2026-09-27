"""Loading data: the historical CSV, the token universe, and gap-free price panels.

Everything is read from ``.data/historical_data.csv``; fetch it with
``python scripts/download_data.py``. The file is long-format, one row per
``(interval, ts, token_id)``:

    ========== ==============================================================
    column     meaning
    ========== ==============================================================
    interval   ``"1d"`` for true daily bars, ``"1h"`` for hourly bars
    ts         bar open time, UTC, ISO-8601
    token_id   lowercase token id, e.g. ``"link"``
    open       first price in the bar
    high       highest price in the bar
    low        lowest price in the bar
    close      last price in the bar
    volume     daily bars: traded volume; hourly bars: trailing 24h volume
    market_cap market cap at the bar close (hourly bars only)
    ========== ==============================================================

Hourly bars are resampled from one spot print per hour, so ``high`` and ``low``
carry no real intrabar range. That matters for any strategy defined on extremes
(a Donchian channel, say). Daily ``high``/``low`` are only populated from 2026.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import (
    BARS_PER_YEAR_DAILY,
    BARS_PER_YEAR_HOURLY,
    DATA_PATH,
    EXCLUDE,
    HOURLY_START,
    MAX_FFILL_DAYS,
    MAX_FFILL_HOURS,
    H,
)

__all__ = ["Panel", "load_bars", "load_universe", "to_wide", "hourly_panel",
           "daily_panel", "equal_weight_bh", "coverage"]

COLUMNS = ["interval", "ts", "token_id", "open", "high", "low", "close", "volume",
           "market_cap"]


@lru_cache(maxsize=2)
def _read_csv(path: str) -> pd.DataFrame:
    """Read and type the CSV once per path; callers get copies."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found. Run `python scripts/download_data.py` "
            f"(or see INFO.md for the manual link).")
    df = pd.read_csv(file, dtype={"interval": "category", "token_id": "string"})
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{file.name} is missing columns: {sorted(missing)}")
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    return df[COLUMNS].sort_values(["interval", "ts", "token_id"], ignore_index=True)


def load_bars(interval: str, tokens: list[str] | None = None, start: str | None = None,
              path: Path = DATA_PATH) -> pd.DataFrame:
    """Load long-format bars for one interval from the historical CSV.

    Args:
        interval: ``"1d"`` or ``"1h"``.
        tokens: Token ids to keep. None keeps every token in the file.
        start: Optional inclusive lower bound on ``ts`` (any pandas-parsable date).
        path: Location of the CSV.

    Returns:
        A DataFrame with the columns in ``COLUMNS`` (minus ``interval``), sorted by
        ``ts``.

    Raises:
        FileNotFoundError: If the CSV has not been downloaded yet.
        ValueError: If the CSV lacks expected columns or ``interval`` is unknown.
    """
    df = _read_csv(str(path))
    if interval not in set(df["interval"]):
        raise ValueError(f"no {interval!r} bars in {Path(path).name}; "
                         f"found {sorted(df['interval'].unique())}")
    mask = df["interval"] == interval
    if tokens is not None:
        mask &= df["token_id"].isin(tokens)
    if start is not None:
        mask &= df["ts"] >= pd.Timestamp(start, tz="UTC")
    return df.loc[mask].drop(columns="interval").reset_index(drop=True)


def load_universe(path: Path = DATA_PATH, exclude: frozenset[str] = EXCLUDE) -> list[str]:
    """List the tradeable tokens in the CSV.

    Args:
        path: Location of the CSV.
        exclude: Token ids to leave out (stablecoins, by default).

    Returns:
        Sorted token ids present in the daily bars, minus ``exclude``.
    """
    tokens = load_bars("1d", path=path)["token_id"].unique()
    return sorted(t for t in tokens if t not in exclude)


@dataclass
class Panel:
    """A price panel plus whatever else came with it, on a gap-free time grid.

    Attributes:
        close: Close prices, one column per token, indexed by bar timestamp.
        bars_per_year: Annualisation factor for this bar frequency.
        freq: Human-readable bar frequency, ``"1d"`` or ``"1h"``.
        volume: Volume panel aligned with ``close``, if available.
        mcap: Market-cap panel aligned with ``close``, if available.
    """
    close: pd.DataFrame
    bars_per_year: int
    freq: str
    volume: pd.DataFrame | None = None
    mcap: pd.DataFrame | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        """The shared bar timestamps."""
        return self.close.index

    @property
    def tokens(self) -> list[str]:
        """Token ids, in column order."""
        return list(self.close.columns)

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, lo: int, hi: int | None = None) -> Panel:
        """Return a positional slice of every panel.

        Args:
            lo: First bar to keep.
            hi: One past the last bar to keep; None keeps through the end.

        Returns:
            A new ``Panel`` covering bars ``[lo, hi)``.
        """
        sl = slice(lo, hi)
        return Panel(self.close.iloc[sl], self.bars_per_year, self.freq,
                     None if self.volume is None else self.volume.iloc[sl],
                     None if self.mcap is None else self.mcap.iloc[sl])

    def describe(self) -> str:
        """One-line summary: bar count, token count and date range."""
        return (f"{len(self):,} {self.freq} bars x {self.close.shape[1]} tokens  "
                f"{self.index[0]} -> {self.index[-1]}")


def to_wide(raw: pd.DataFrame, field: str, tokens: list[str], freq: str,
            ffill_limit: int) -> pd.DataFrame:
    """Pivot long bars into a gap-free ``ts x token`` panel.

    A token's history starts at its first print. Holes *inside* that history are
    stale-quote artifacts, so the last value is carried forward, but only for
    ``ffill_limit`` bars so an outage can't masquerade as a flat price.

    Args:
        raw: Long-format bars with ``ts``, ``token_id`` and ``field`` columns.
        field: Column to pivot, e.g. ``"close"``.
        tokens: Column order for the result; tokens absent from ``raw`` are dropped.
        freq: Pandas offset alias for the grid, e.g. ``"1D"`` or ``"1h"``.
        ffill_limit: Maximum consecutive bars to forward-fill.

    Returns:
        A wide DataFrame indexed by a regular UTC ``DatetimeIndex`` named ``ts``.
    """
    p = raw.pivot(index="ts", columns="token_id", values=field).sort_index()
    p = p.reindex(columns=[t for t in tokens if t in p.columns])
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq=freq, tz="UTC"))
    p.index.name = "ts"
    p.columns.name = None
    return p.ffill(limit=ffill_limit)


def hourly_panel(tokens: list[str] | None = None, start: str = HOURLY_START,
                 raw: pd.DataFrame | None = None) -> Panel:
    """Build the hourly panel (close, trailing-24h volume, market cap).

    Args:
        tokens: Token ids to include. None uses ``load_universe()``.
        start: First bar to include.
        raw: Pre-loaded long bars; skips reading the CSV when given.

    Returns:
        An hourly ``Panel``.
    """
    tokens = load_universe() if tokens is None else tokens
    raw = load_bars("1h", tokens, start) if raw is None else raw

    def wide(field: str) -> pd.DataFrame:
        return to_wide(raw, field, tokens, "1h", MAX_FFILL_HOURS)

    return Panel(close=wide("close"), bars_per_year=BARS_PER_YEAR_HOURLY, freq="1h",
                 volume=wide("volume"), mcap=wide("market_cap"))


def daily_panel(tokens: list[str] | None = None, raw: pd.DataFrame | None = None) -> Panel:
    """Build the daily panel (close and volume) from true 1d bars.

    Args:
        tokens: Token ids to include. None uses ``load_universe()``.
        raw: Pre-loaded long bars; skips reading the CSV when given.

    Returns:
        A daily ``Panel``.
    """
    tokens = load_universe() if tokens is None else tokens
    raw = load_bars("1d", tokens) if raw is None else raw
    return Panel(close=to_wide(raw, "close", tokens, "1D", MAX_FFILL_DAYS),
                 bars_per_year=BARS_PER_YEAR_DAILY, freq="1d",
                 volume=to_wide(raw, "volume", tokens, "1D", MAX_FFILL_DAYS))


def equal_weight_bh(close: pd.DataFrame) -> pd.Series:
    """Equal-weight buy & hold: buy every token once at its first print, never trade.

    Args:
        close: Wide close-price panel.

    Returns:
        The basket's growth-of-1 series.
    """
    live = close.dropna(axis=1, how="all")
    return live.apply(lambda s: s / s.loc[s.first_valid_index()]).mean(axis=1, skipna=True)


def coverage(close: pd.DataFrame, bars_per_day: int = H) -> pd.DataFrame:
    """Per-token history extent and interior gaps.

    Args:
        close: Wide close-price panel.
        bars_per_day: Bars per day at this frequency (24 hourly, 1 daily).

    Returns:
        One row per token with ``first_bar``, ``last_bar``, ``bars``, ``gaps``
        and ``days``, longest history first.
    """
    out = pd.DataFrame({
        "first_bar": close.apply(lambda s: s.first_valid_index()),
        "last_bar":  close.apply(lambda s: s.last_valid_index()),
        "bars":      close.notna().sum(),
        "gaps":      close.apply(lambda s: s.loc[s.first_valid_index():].isna().sum()),
    })
    out["days"] = (out.bars / bars_per_day).round(1)
    return out.sort_values("bars", ascending=False)
