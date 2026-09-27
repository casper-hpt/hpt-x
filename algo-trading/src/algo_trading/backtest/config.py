"""Shared constants and strategy/portfolio configuration objects.

Every config is a frozen dataclass so it can be ``replace()``-d safely inside a
sweep without any risk of a mutated object leaking between runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# ── Data ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # src/algo_trading/backtest -> root
DATA_DIR = PROJECT_ROOT / ".data"
DATA_PATH = DATA_DIR / "historical_data.csv"

# ── Cost ─────────────────────────────────────────────────────────────────────
FEE_BPS = 20.0                  # 0.2% of notional, charged per fill (buy AND sell)

# ── Universe ─────────────────────────────────────────────────────────────────
EXCLUDE = frozenset({"usdc"})   # stablecoin — it is the cash leg, not a position

# ── Bar arithmetic ───────────────────────────────────────────────────────────
H = 24                          # hourly bars per day
BARS_PER_YEAR_HOURLY = 24 * 365
BARS_PER_YEAR_DAILY = 365

HOURLY_START = "2026-05-25"
MAX_FFILL_HOURS = 6             # cap the carry so an outage can't fake a flat series
MAX_FFILL_DAYS = 3


@dataclass(frozen=True)
class PortfolioConfig:
    """Sizing and cost. Every field here is a cost decision.

    ``band`` is the single most important number: a position is left alone until
    its weight drifts more than ``band`` (relative to target) away, which is what
    keeps a 0.2%-per-fill strategy from becoming a fee-generation machine.

    Attributes:
        scheme: ``"invvol"`` for risk parity or ``"equal"`` for equal weight.
        band: No-trade band, relative to the target weight.
        check_bars: How often (in bars) drift is inspected at all.
        max_weight: Per-position cap as a fraction of equity.
        vol_span: EWM span, in bars, for the per-token volatility estimate.
        min_history: Bars a token needs before it becomes investable.
        min_trade_frac: Orders smaller than this share of equity are skipped.
        normalize: If True, re-spread weights to full investment (no cash).
    """
    scheme:         str   = "invvol"
    band:           float = 0.25
    check_bars:     int   = 6
    max_weight:     float = 0.15
    vol_span:       int   = 7 * H
    min_history:    int   = 14 * H
    min_trade_frac: float = 0.002
    normalize:      bool  = False

    @property
    def warmup(self) -> int:
        """Bars before the first weight is trustworthy."""
        return self.min_history + self.vol_span

    def daily(self, **overrides) -> PortfolioConfig:
        """Return the same config expressed in daily bars.

        Args:
            **overrides: Fields to set on top of the daily defaults.

        Returns:
            A new ``PortfolioConfig`` with daily-bar spans.
        """
        base = dict(vol_span=7, min_history=30, check_bars=1)
        base.update(overrides)
        return replace(self, **base)


@dataclass(frozen=True)
class DonchianConfig:
    """Channel breakout. Defaults are the canonical Turtle 20d/10d, in hours.

    Attributes:
        entry_lb: Lookback for the entry channel (break above the prior high).
        exit_lb: Lookback for the exit channel (break below the prior low).
        atr_span: EWM span for the ATR estimate.
        atr_stop: Exit at ``entry - k * ATR``; None means channel exit only.
        confirm: Bars the breakout must hold before acting.
        max_hold: Hard time exit in bars; None means channel exit only.
    """
    entry_lb: int = 20 * H
    exit_lb:  int = 10 * H
    atr_span: int = 7 * H
    atr_stop: float | None = None
    confirm:  int = 1
    max_hold: int | None = None

    @property
    def warmup(self) -> int:
        """Bars before the entry channel is defined."""
        return self.entry_lb + 1


@dataclass(frozen=True)
class EMAConfig:
    """The original adaptive-sigma EMA crossover, in hours. Kept as a benchmark.

    Attributes:
        base_span: Fast EMA span.
        slow_span_1: First slow EMA span.
        slow_span_2: Second (slowest) EMA span.
        mu_lookback: Window for the mean-return filter.
        sigma_lookback: Window for the volatility normaliser.
        momentum_lookback: Window for the momentum score.
        buy_threshold: Normalised score above which a token is bought.
        sell_threshold: Normalised score below which a token is sold.
        cooldown_bars: Bars to wait after an exit before re-entering.
        max_positions: Maximum concurrent holdings.
        require_mu_positive: Only buy when the trailing mean return is positive.
    """
    base_span:           int   = 6 * H
    slow_span_1:         int   = 24 * H
    slow_span_2:         int   = 48 * H
    mu_lookback:         int   = 6 * H
    sigma_lookback:      int   = 7 * H
    momentum_lookback:   int   = 6 * H
    buy_threshold:       float = 0.02
    sell_threshold:      float = -0.02
    cooldown_bars:       int   = 2 * H
    max_positions:       int   = 10
    require_mu_positive: bool  = True

    @property
    def warmup(self) -> int:
        """Bars before the slowest EMA is defined."""
        return self.slow_span_2 + 1


@dataclass(frozen=True)
class FeatureConfig:
    """Cross-sectional features. Retained because the IC gate is worth re-running.

    Attributes:
        mom_horizons: Momentum lookbacks, in bars.
        sigma_span: EWM span for volatility scaling.
        beta_window: Rolling window for market beta.
        er_window: Window for the efficiency ratio.
        rev_window: Window for short-term reversal.
        breadth_window: Window for market breadth.
        weights: Blend weights for the composite score.
    """
    mom_horizons:   tuple = (3 * H, 7 * H, 14 * H)
    sigma_span:     int   = 7 * H
    beta_window:    int   = 14 * H
    er_window:      int   = 7 * H
    rev_window:     int   = 12
    breadth_window: int   = 7 * H
    weights:        dict  = field(default_factory=lambda: {
        "resid_mom": 1.0, "trend": 1.0, "reversal": 0.5, "liquidity": 0.25})


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward refit protocol.

    Attributes:
        train: Bars in the trailing training window.
        test: Bars traded before the next refit.
        how: Selection objective, ``"sharpe"`` or ``"total"``.
        mode: ``"per_coin"``, ``"global"``, ``"random"`` or ``"fixed"``.
        seed: RNG seed for the ``"random"`` mode.
    """
    train: int = 480
    test:  int = 240
    how:   str = "sharpe"
    mode:  str = "per_coin"
    seed:  int = 1
