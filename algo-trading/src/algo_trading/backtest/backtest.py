"""The portfolio engine: target weights in, equity curve and fee bill out.

One simulator serves every strategy in this package. A strategy's only job is to
produce a target-weight panel; ``simulate`` decides what actually gets traded,
which is where the 0.2%-per-fill cost is either absorbed or squandered.

Two things here are load-bearing and easy to get wrong elsewhere:

* **No look-ahead.** A bar trades on the PREVIOUS bar's target at THIS bar's price.
* **The no-trade band.** A position is left alone until its weight drifts more
  than ``band`` (relative) from target, so drift costs nothing until it matters.
  Exits always execute, otherwise a delisted name could never be sold.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import FEE_BPS, PortfolioConfig
from .data import Panel, equal_weight_bh

__all__ = ["PERF_FMT", "PERF_COLS", "fmt_table", "show", "risk_inputs", "base_weights",
           "simulate", "perf", "stats_from_result", "run_weights", "run_flat", "exposure_matched",
           "benchmark_rows"]

PERF_FMT = {"total_ret": "{:+.1%}", "sharpe": "{:.2f}", "vol": "{:.0%}", "max_dd": "{:.1%}",
            "fees_pct": "{:.2%}", "turnover": "{:.1f}x", "gross": "{:.0%}",
            "exposure": "{:.0%}", "n_fills": "{:.0f}", "p_value": "{:.3f}"}
PERF_COLS = ["total_ret", "sharpe", "vol", "max_dd", "n_fills", "fees_pct",
             "turnover", "gross", "exposure"]


# ── Tables ───────────────────────────────────────────────────────────────────

def fmt_table(df: pd.DataFrame, fmts: dict[str, str]) -> pd.DataFrame:
    """Format numeric columns as plain strings.

    ``df.style`` needs jinja2, which isn't in every environment, so this returns
    an ordinary DataFrame of strings instead.

    Args:
        df: Table to format.
        fmts: ``{column: format spec}``; columns not in ``df`` are ignored.

    Returns:
        A copy of ``df`` with the listed columns rendered as strings (NaN -> "—").
    """
    out = df.copy()
    for col, spec in fmts.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v, s=spec: "—" if pd.isna(v) else s.format(v))
    return out


def show(rows: list[dict], cols: list[str] | None = None) -> pd.DataFrame:
    """Format a list of ``perf()`` dicts into a readable table.

    Args:
        rows: Stat dicts, each with a ``label`` key.
        cols: Columns to show, in order. Defaults to ``PERF_COLS``. Columns that
            are entirely NaN are dropped.

    Returns:
        A formatted DataFrame indexed by label.
    """
    df = pd.DataFrame(rows).set_index("label")
    keep = [c for c in (cols or PERF_COLS) if c in df.columns and df[c].notna().any()]
    return fmt_table(df[keep], PERF_FMT)


# ── Weights ──────────────────────────────────────────────────────────────────

def risk_inputs(close: pd.DataFrame, cfg: PortfolioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-token volatility and the investable mask.

    A token is investable once it has ``cfg.min_history`` bars of history and a
    positive, finite volatility estimate.

    Args:
        close: Wide close-price panel.
        cfg: Portfolio config (uses ``vol_span`` and ``min_history``).

    Returns:
        ``(sigma, investable)``: EWM return volatility and a boolean mask, both
        shaped like ``close``.
    """
    sigma = close.pct_change().ewm(span=cfg.vol_span, adjust=False).std()
    age = close.notna().cumsum()
    investable = close.notna() & (age >= cfg.min_history) & sigma.notna() & (sigma > 0)
    return sigma, investable


def _normalize_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Scale each row to sum to 1; all-zero rows stay zero."""
    raw = raw.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def base_weights(close: pd.DataFrame, cfg: PortfolioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Risk-parity (or equal) weights over the whole investable universe.

    Equal weight silently hands most of the *risk* to the wildest token;
    inverse-vol equalises risk contribution instead of capital. Weights are
    capped at ``cfg.max_weight`` with the excess redistributed pro rata.

    Args:
        close: Wide close-price panel.
        cfg: Portfolio config.

    Returns:
        ``(weights, sigma)``: target weights (rows sum to 1 wherever anything is
        investable) and the volatility estimate used to build them.
    """
    sigma, investable = risk_inputs(close, cfg)
    if cfg.scheme == "equal":
        raw = investable.astype(float)
    else:
        raw = (1.0 / sigma).where(investable, 0.0)
    w = _normalize_rows(raw)

    for _ in range(10):                       # iterative cap-and-renormalise
        over = w > cfg.max_weight
        if not over.values.any():
            break
        excess = (w[over] - cfg.max_weight).sum(axis=1)
        w = w.mask(over, cfg.max_weight)
        room = w.where(~over & (w > 0), 0.0)
        w = w.add(room.div(room.sum(axis=1).replace(0, np.nan), axis=0)
                  .mul(excess, axis=0).fillna(0.0))
    return w, sigma


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate(close: pd.DataFrame, w_tgt: pd.DataFrame, cfg: PortfolioConfig,
             fee_bps: float = FEE_BPS, initial: float = 100_000.0,
             warmup: int = 0) -> dict:
    """Band-rebalanced, long-only simulation with per-fill fees.

    On every ``cfg.check_bars``-th bar after warmup, each position is compared
    with the previous bar's target. Only positions outside the no-trade band (or
    being exited) are traded. Buys are capped at available cash; fees come out of
    each ticket.

    Args:
        close: Wide close-price panel.
        w_tgt: Target weights, aligned (or alignable) to ``close``.
        cfg: Portfolio config (uses ``band``, ``check_bars``, ``min_trade_frac``).
        fee_bps: Cost per fill, in basis points of notional.
        initial: Starting cash.
        warmup: Bars to skip before the first trade.

    Returns:
        A dict with:
            equity: Mark-to-market equity series.
            fills: One row per fill (``ts``, ``token``, ``side``, ``notional``, ``fee``).
            log: One row per rebalance check (``ts``, ``n_held``, ``gross``, ``cash_pct``).
            fees: Total fees paid.
            initial: Starting cash.
            turnover: Total traded notional divided by ``initial``.
            warmup, cfg, fee_bps: The inputs, for provenance.
    """
    fee = fee_bps / 1e4
    idx, toks = close.index, list(close.columns)
    px = close.to_numpy(float)
    W = w_tgt.reindex(idx).reindex(columns=toks).to_numpy(float)

    n = len(toks)
    qty = np.zeros(n)
    last = np.full(n, np.nan)
    cash = initial
    eq = np.empty(len(idx))
    fees = traded = 0.0
    fills, log = [], []
    first = max(warmup, 1)

    for t in range(len(idx)):
        p = px[t]
        last = np.where(np.isnan(p), last, p)          # mark on last known price
        equity = cash + float((qty * np.nan_to_num(last)).sum())

        if t >= first and (t - first) % cfg.check_bars == 0 and equity > 0:
            tgt = np.nan_to_num(W[t - 1])              # closed-bar target, traded now
            tradeable = ~np.isnan(p) & (p > 0)
            w_now = np.where(tradeable, qty * np.nan_to_num(p) / equity, 0.0)
            d = tgt - w_now

            for i in range(n):
                if not tradeable[i]:
                    continue
                exiting = tgt[i] <= 1e-9 and qty[i] > 0
                band_i = cfg.band * max(tgt[i], 0.01)
                if not exiting and (abs(d[i]) < band_i or abs(d[i]) < cfg.min_trade_frac):
                    continue                            # inside the band — leave it alone
                notional = d[i] * equity
                if notional > 0:
                    notional = min(notional, cash)
                    if notional <= 1e-8:
                        continue
                    c = notional * fee
                    qty[i] += (notional - c) / p[i]     # fee comes out of the ticket
                    cash -= notional
                    side = "BUY"
                else:
                    sell_qty = min(qty[i], -notional / p[i])
                    if sell_qty <= 1e-12:
                        continue
                    proceeds = sell_qty * p[i]
                    c = proceeds * fee
                    qty[i] -= sell_qty
                    cash += proceeds - c
                    notional, side = -proceeds, "SELL"
                fees += c
                traded += abs(notional)
                fills.append({"ts": idx[t], "token": toks[i], "side": side,
                              "notional": abs(notional), "fee": c})
            equity = cash + float((qty * np.nan_to_num(last)).sum())
            log.append({"ts": idx[t], "n_held": int((qty > 0).sum()),
                        "gross": float((qty * np.nan_to_num(p)).sum() / max(equity, 1e-9)),
                        "cash_pct": cash / max(equity, 1e-9)})
        eq[t] = equity

    return {"equity": pd.Series(eq, index=idx, name="equity"),
            "fills": pd.DataFrame(fills), "log": pd.DataFrame(log),
            "fees": fees, "initial": initial, "turnover": traded / initial,
            "warmup": warmup, "cfg": cfg, "fee_bps": fee_bps}


# ── Statistics ───────────────────────────────────────────────────────────────

def perf(equity: pd.Series, label: str = "", start: int = 0, fees: float = 0.0,
         initial: float = 100_000.0, n_fills: int = 0, turnover: float = 0.0,
         gross: float = np.nan, exposure: float = np.nan,
         bars_per_year: int = 24 * 365) -> dict:
    """Compute the standard stat block for one equity curve.

    Args:
        equity: Equity series.
        label: Row label for tables.
        start: Bars to clip from the front (the warmup), so rows stay comparable.
        fees: Total fees paid.
        initial: Starting equity, used to express fees as a fraction.
        n_fills: Number of fills.
        turnover: Traded notional over initial equity.
        gross: Average gross exposure.
        exposure: Average signal exposure (share of the book switched on).
        bars_per_year: Annualisation factor.

    Returns:
        A dict with ``label``, ``total_ret``, ``sharpe``, ``vol``, ``max_dd``,
        ``n_fills``, ``fees_pct``, ``turnover``, ``gross`` and ``exposure``.
    """
    e = equity.iloc[start:].dropna()
    r = e.pct_change().dropna()
    ann = math.sqrt(bars_per_year)
    return {
        "label": label,
        "total_ret": e.iloc[-1] / e.iloc[0] - 1.0 if len(e) else np.nan,
        "sharpe": (r.mean() / r.std() * ann) if r.std() > 0 else np.nan,
        "vol": r.std() * ann,
        "max_dd": (e / e.cummax() - 1.0).min() if len(e) else np.nan,
        "n_fills": n_fills,
        "fees_pct": fees / initial,
        "turnover": turnover,
        "gross": gross,
        "exposure": exposure,
    }


def stats_from_result(res: dict, label: str, start: int, bars_per_year: int,
                      exposure: float = np.nan) -> dict:
    """Run ``perf`` on a ``simulate`` result.

    Args:
        res: Output of ``simulate``.
        label: Row label.
        start: Bars to clip from the front.
        bars_per_year: Annualisation factor.
        exposure: Average signal exposure, if known.

    Returns:
        A ``perf`` stat dict.
    """
    return perf(res["equity"], label, start=start, fees=res["fees"], initial=res["initial"],
                n_fills=len(res["fills"]), turnover=res["turnover"],
                gross=res["log"].gross.mean() if len(res["log"]) else np.nan,
                exposure=exposure, bars_per_year=bars_per_year)


# ── Runners ──────────────────────────────────────────────────────────────────

def run_weights(panel: Panel, weights: pd.DataFrame, cfg: PortfolioConfig,
                label: str = "", fee_bps: float = FEE_BPS, warmup: int | None = None,
                exposure: float = np.nan) -> tuple[dict, dict]:
    """Simulate a ready-made target-weight panel.

    Args:
        panel: Price panel.
        weights: Target weights aligned to ``panel.close``.
        cfg: Portfolio config.
        label: Row label.
        fee_bps: Cost per fill, in basis points.
        warmup: Bars to skip; defaults to ``cfg.warmup``.
        exposure: Average signal exposure, if known.

    Returns:
        ``(stats, result)``: the ``perf`` dict and the raw ``simulate`` output.
    """
    wu = cfg.warmup if warmup is None else warmup
    res = simulate(panel.close, weights, cfg, fee_bps=fee_bps, warmup=wu)
    return stats_from_result(res, label, wu, panel.bars_per_year, exposure), res


def run_flat(panel: Panel, gross: float, label: str, cfg: PortfolioConfig,
             fee_bps: float = FEE_BPS, warmup: int | None = None) -> tuple[dict, dict]:
    """Risk parity held at a CONSTANT gross exposure, with no timing whatsoever.

    This is the benchmark that decides whether a timing signal is worth anything.
    A signal that is long 30% of the time will always look low-risk; the question
    is whether it beats simply *being* 30% invested and leaving it alone.

    Args:
        panel: Price panel.
        gross: Constant gross exposure, e.g. 0.3 for 30% invested.
        label: Row label.
        cfg: Portfolio config.
        fee_bps: Cost per fill, in basis points.
        warmup: Bars to skip; defaults to ``cfg.warmup``.

    Returns:
        ``(stats, result)``: the ``perf`` dict and the raw ``simulate`` output.
    """
    wu = cfg.warmup if warmup is None else warmup
    w_all, _ = base_weights(panel.close, cfg)
    res = simulate(panel.close, w_all * gross, cfg, fee_bps=fee_bps, warmup=wu)
    return stats_from_result(res, label, wu, panel.bars_per_year, exposure=gross), res


def exposure_matched(panel: Panel, stat: dict, cfg: PortfolioConfig,
                     fee_bps: float = FEE_BPS, warmup: int | None = None,
                     indent: str = "   ") -> dict:
    """Build the exposure-matched twin of a strategy row, ready to sit beneath it.

    Args:
        panel: Price panel.
        stat: A strategy's ``perf`` dict with a finite ``exposure``.
        cfg: Portfolio config.
        fee_bps: Cost per fill, in basis points.
        warmup: Bars to skip; defaults to ``cfg.warmup``.
        indent: Prefix for the label so the row reads as a child in tables.

    Returns:
        The ``perf`` dict of ``run_flat`` at the strategy's average exposure.

    Raises:
        ValueError: If ``stat`` has no finite ``exposure``.
    """
    g = stat.get("exposure", np.nan)
    if not np.isfinite(g):
        raise ValueError("stat has no finite 'exposure' to match")
    return run_flat(panel, g, f"{indent}flat {g:.0%} gross (no timing)", cfg,
                    fee_bps=fee_bps, warmup=warmup)[0]


def benchmark_rows(panel: Panel, cfg: PortfolioConfig, warmup: int | None = None,
                   fee_bps: float = FEE_BPS) -> list[dict]:
    """Buy & hold and always-on risk parity, scored on a common window.

    Args:
        panel: Price panel.
        cfg: Portfolio config.
        warmup: Bars to skip; defaults to ``cfg.warmup``.
        fee_bps: Cost per fill for the risk-parity row, in basis points.

    Returns:
        Two ``perf`` dicts: equal-weight buy & hold, then risk parity always on.
    """
    wu = cfg.warmup if warmup is None else warmup
    bh = equal_weight_bh(panel.close)
    bh = bh / bh.iloc[wu] * 100_000.0
    return [perf(bh, "buy & hold (equal wt)", start=wu, gross=1.0, exposure=1.0,
                 bars_per_year=panel.bars_per_year),
            run_flat(panel, 1.0, "risk parity, always on", cfg,
                     fee_bps=fee_bps, warmup=wu)[0]]
