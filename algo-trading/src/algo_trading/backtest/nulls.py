"""Null tests: the gates that decide whether a result means anything.

The idea is the same in every case: keep the strategy, the costs and the
signal's own statistics exactly as they are, and destroy ONLY the thing being
claimed. If the real result sits inside the resulting distribution, that claim
is not supported.

``rotation_null``
    Destroys the alignment between signal and price, keeping duty cycle, run
    lengths and autocorrelation intact. Tests: "does the TIMING carry
    information?"

``sigma_rotation_null``
    Weights the book by the inverse of the WRONG token's volatility, keeping the
    weighting machinery intact. Tests: "is the vol ESTIMATE informative, or is
    any smooth weighting just as good?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import base_weights, risk_inputs, simulate, stats_from_result
from .config import FEE_BPS, PortfolioConfig
from .data import Panel

__all__ = ["rotation_null", "sigma_rotation_null", "null_summary"]


def _rotate(df: pd.DataFrame, rng: np.random.Generator, lo: int) -> pd.DataFrame:
    """Circularly rotate each column by an independent offset in ``[lo, len(df))``."""
    out = df.copy()
    n = len(df)
    for c in out.columns:
        out[c] = np.roll(out[c].to_numpy(float), int(rng.integers(lo, n)))
    return out


def rotation_null(panel: Panel, signal: pd.DataFrame, cfg: PortfolioConfig,
                  n_sims: int = 300, seed: int = 5, warmup: int | None = None,
                  fee_bps: float = FEE_BPS) -> pd.DataFrame:
    """Simulate the strategy with each token's signal randomly rotated in time.

    Args:
        panel: Price panel.
        signal: 0/1 (or fractional) signal panel aligned to ``panel.close``.
        cfg: Portfolio config.
        n_sims: Number of rotations.
        seed: RNG seed.
        warmup: Bars to skip; defaults to ``cfg.warmup``.
        fee_bps: Cost per fill, in basis points.

    Returns:
        One row of ``perf`` stats per simulation.
    """
    wu = cfg.warmup if warmup is None else warmup
    rng = np.random.default_rng(seed)
    w_all, _ = base_weights(panel.close, cfg)
    sig = signal.fillna(0.0)

    out = []
    for _ in range(n_sims):
        shuffled = _rotate(sig, rng, wu)
        res = simulate(panel.close, w_all * shuffled, cfg, fee_bps=fee_bps, warmup=wu)
        out.append(stats_from_result(res, "", wu, panel.bars_per_year,
                                     exposure=float(shuffled.iloc[wu:].mean().mean())))
    return pd.DataFrame(out)


def sigma_rotation_null(panel: Panel, cfg: PortfolioConfig, n_sims: int = 200,
                        seed: int = 7, warmup: int | None = None,
                        fee_bps: float = FEE_BPS) -> pd.DataFrame:
    """Simulate inverse-vol weights built from mis-assigned volatilities.

    Args:
        panel: Price panel.
        cfg: Portfolio config.
        n_sims: Number of rotations.
        seed: RNG seed.
        warmup: Bars to skip; defaults to ``cfg.warmup``.
        fee_bps: Cost per fill, in basis points.

    Returns:
        One row of ``perf`` stats per simulation.
    """
    wu = cfg.warmup if warmup is None else warmup
    rng = np.random.default_rng(seed)
    sigma, investable = risk_inputs(panel.close, cfg)

    def normalize(w: pd.DataFrame) -> pd.DataFrame:
        w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    out = []
    for _ in range(n_sims):
        shuffled = _rotate(sigma, rng, wu)
        w = normalize((1.0 / shuffled).where(investable, 0.0))
        w = normalize(w.clip(upper=cfg.max_weight))
        res = simulate(panel.close, w, cfg, fee_bps=fee_bps, warmup=wu)
        out.append(stats_from_result(res, "", wu, panel.bars_per_year, exposure=1.0))
    return pd.DataFrame(out)


def null_summary(null: pd.DataFrame, real: dict,
                 metrics: tuple[str, ...] = ("total_ret", "sharpe")) -> pd.DataFrame:
    """Compare a real result with its null distribution.

    ``p_value`` is the share of nulls that MATCHED OR BEAT the real result, so a
    small p means the real result is hard to reproduce by chance.

    Args:
        null: Output of a null function (one row per simulation).
        real: The real strategy's ``perf`` dict.
        metrics: Stat columns to compare.

    Returns:
        One row per metric with ``real``, ``null_p5``, ``null_median``,
        ``null_p95``, ``p_value`` and ``n_sims``.
    """
    rows = []
    for m in metrics:
        col = null[m].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"metric": m, "real": real[m], "null_p5": col.quantile(0.05),
                     "null_median": col.median(), "null_p95": col.quantile(0.95),
                     "p_value": float((col >= real[m]).mean()), "n_sims": len(col)})
    return pd.DataFrame(rows).set_index("metric")
