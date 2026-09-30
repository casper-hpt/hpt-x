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
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import FEE_BPS, PortfolioConfig
from .data import Panel, equal_weight_bh

__all__ = ["PERF_FMT", "PERF_COLS", "fmt_table", "show", "risk_inputs", "slot_weights",
           "base_weights", "simulate", "simulate_many", "held_weights", "perf",
           "stats_from_result",
           "run_weights", "run_flat", "exposure_matched", "benchmark_rows"]

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


def slot_weights(signal: pd.DataFrame, n: int, priority: pd.DataFrame | None = None,
                 seed: int | None = 0, exit: pd.DataFrame | None = None,
                 min_hold: int = 0) -> pd.DataFrame:
    """Hold up to ``n`` names at ``1/n`` each, first come, first served.

    A name enters when its signal is on AND a slot is free, then stays until its
    own exit fires, however many stronger names appear meanwhile. A
    slot freed on bar ``t`` can be refilled on the same bar. Empty slots are
    cash. Pair with ``PortfolioConfig(rebalance=False)`` for true buy-and-hold
    per slot (no trimming as positions drift).

    Args:
        signal: Boolean (or 0/1) panel; True where a name may be bought (and,
            without ``exit``, where it may keep being held).
        n: Number of slots.
        priority: Optional panel ranking simultaneous candidates for a free
            slot (higher first), e.g. trend strength; ties go to column order.
            If None, candidates are chosen at random when there are more of
            them than free slots.
        seed: Seed for the random choice, so a run is reproducible. Try several
            seeds to see how much the result depends on the draw.
        exit: Optional boolean panel; True where a held name must be sold. Use it
            for entry/exit rules that differ (e.g. momentum thresholds with a
            gap). Defaults to ``~signal``.
        min_hold: Bars an exit is ignored for after each buy, so a name bought on
            bar ``t`` can first be sold on bar ``t + min_hold``. Filters whipsaw:
            signals that flip back and forth around a crossover. 0 and 1 both
            mean no filter, since the earliest exit is already the next bar.

    Returns:
        Target weights: ``1/n`` for each held name, 0 elsewhere.
    """
    on = signal.fillna(False).astype(bool).to_numpy()
    off = (~on if exit is None
           else exit.reindex_like(signal).fillna(False).astype(bool).to_numpy())
    rank = (None if priority is None
            else np.nan_to_num(priority.reindex_like(signal).to_numpy(float), nan=-np.inf))
    rng = np.random.default_rng(seed)
    held = np.zeros(on.shape[1], dtype=bool)
    age = np.zeros(on.shape[1], dtype=int)             # bars since each name was bought
    out = np.zeros(on.shape)
    for t in range(len(on)):
        age += held
        held &= ~(off[t] & (age >= min_hold))           # exits on each name's own rule
        free = n - int(held.sum())
        if free > 0:
            cand = np.flatnonzero(on[t] & ~held)
            if len(cand) > free:                        # more candidates than slots
                cand = (rng.choice(cand, free, replace=False) if rank is None
                        else cand[np.argsort(-rank[t, cand], kind="stable")[:free]])
            held[cand] = True
            age[cand] = 0
        out[t] = held / n
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


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

try:  # the optional Rust engine in rust/backtest; see rust/README.md
    import algo_backtest_rs as _rs
except ImportError:
    _rs = None

FILL_COLS = ["ts", "token", "side", "notional", "fee"]
LOG_COLS = ["ts", "n_held", "gross", "cash_pct"]


def _engine(engine: str) -> str:
    """Resolve ``"auto"`` to the fastest engine installed."""
    if engine == "auto":
        return "rust" if _rs is not None else "python"
    if engine == "rust" and _rs is None:
        raise ImportError("the Rust engine is not installed; build it with "
                          "`maturin develop --release -m rust/backtest/Cargo.toml`")
    if engine not in ("rust", "python"):
        raise ValueError(f"engine must be 'auto', 'rust' or 'python', not {engine!r}")
    return engine


def _arrays(close: pd.DataFrame, w_tgt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prices and targets as C-contiguous float arrays on ``close``'s grid."""
    px = np.ascontiguousarray(close.to_numpy(float))
    W = np.ascontiguousarray(
        w_tgt.reindex(close.index).reindex(columns=close.columns).to_numpy(float))
    return px, W


def _pack(close: pd.DataFrame, equity: np.ndarray, fills: pd.DataFrame, log: pd.DataFrame,
          fees: float, traded: float, cfg: PortfolioConfig, fee_bps: float, initial: float,
          warmup: int) -> dict:
    return {"equity": pd.Series(equity, index=close.index, name="equity"),
            "fills": fills, "log": log, "fees": fees, "initial": initial,
            "turnover": traded / initial, "warmup": warmup, "cfg": cfg, "fee_bps": fee_bps}


def _from_rust(close: pd.DataFrame, raw: dict, cfg: PortfolioConfig, fee_bps: float,
               initial: float, warmup: int) -> dict:
    """Turn the Rust engine's flat arrays into the usual result dict."""
    idx, toks = close.index, np.asarray(close.columns, dtype=object)
    fills = pd.DataFrame({"ts": idx[raw["fill_t"]], "token": toks[raw["fill_token"]],
                          "side": np.where(raw["fill_buy"], "BUY", "SELL"),
                          "notional": raw["fill_notional"], "fee": raw["fill_fee"]},
                         columns=FILL_COLS)
    log = pd.DataFrame({"ts": idx[raw["log_t"]], "n_held": raw["log_n_held"],
                        "gross": raw["log_gross"], "cash_pct": raw["log_cash_pct"]},
                       columns=LOG_COLS)
    return _pack(close, raw["equity"], fills, log, raw["fees"], raw["traded"], cfg, fee_bps,
                 initial, warmup)


def _simulate_py(close: pd.DataFrame, px: np.ndarray, W: np.ndarray, cfg: PortfolioConfig,
                 fee_bps: float, initial: float, warmup: int) -> dict:
    """Pure-Python reference loop. rust/backtest/src/sim.rs is a port of this; keep in step."""
    idx, toks = close.index, list(close.columns)
    fee = fee_bps / 1e4

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
            band = cfg.band                             # band_all: any breach opens it for all
            if cfg.band_all and (tradeable & (np.abs(d) >= cfg.band * np.maximum(tgt, 0.01))
                                 & (np.abs(d) >= cfg.min_trade_frac)).any():
                band = 0.0

            for i in np.argsort(d, kind="stable"):     # sells first, so swaps can fund buys
                if not tradeable[i]:
                    continue
                exiting = tgt[i] <= 1e-9 and qty[i] > 0
                if not cfg.rebalance and qty[i] > 0 and not exiting:
                    continue                            # buy-and-hold: never resize
                band_i = band * max(tgt[i], 0.01)
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
                    sell_qty = qty[i] if exiting else min(qty[i], -notional / p[i])
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

    return _pack(close, eq, pd.DataFrame(fills, columns=FILL_COLS),
                 pd.DataFrame(log, columns=LOG_COLS), fees, traded, cfg, fee_bps, initial, warmup)


def simulate(close: pd.DataFrame, w_tgt: pd.DataFrame, cfg: PortfolioConfig,
             fee_bps: float = FEE_BPS, initial: float = 100_000.0,
             warmup: int = 0, engine: str = "auto") -> dict:
    """Band-rebalanced, long-only simulation with per-fill fees.

    On every ``cfg.check_bars``-th bar after warmup, each position is compared
    with the previous bar's target. Only positions outside the no-trade band (or
    being exited) are traded. Sells go first, then buys, which are capped at
    available cash; fees come out of each ticket.

    Args:
        close: Wide close-price panel.
        w_tgt: Target weights, aligned (or alignable) to ``close``.
        cfg: Portfolio config (uses ``band``, ``band_all``, ``check_bars``,
            ``min_trade_frac``, ``rebalance``).
        fee_bps: Cost per fill, in basis points of notional.
        initial: Starting cash.
        warmup: Bars to skip before the first trade.
        engine: ``"rust"`` (compiled, see rust/README.md), ``"python"`` (the
            reference loop), or ``"auto"`` for Rust when it is installed.

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
    px, W = _arrays(close, w_tgt)
    if _engine(engine) == "python":
        return _simulate_py(close, px, W, cfg, fee_bps, initial, warmup)
    raw = _rs.simulate(px, W, float(fee_bps), float(initial), int(warmup), int(cfg.check_bars),
                       float(cfg.band), float(cfg.min_trade_frac), bool(cfg.rebalance),
                       bool(cfg.band_all))
    return _from_rust(close, raw, cfg, fee_bps, initial, warmup)


def simulate_many(close: pd.DataFrame, weights: Sequence[pd.DataFrame], cfg: PortfolioConfig,
                  fee_bps: float | Sequence[float] = FEE_BPS, initial: float = 100_000.0,
                  warmup: int = 0, engine: str = "auto") -> list[dict]:
    """Run ``simulate`` for many weight panels on the same prices.

    With the Rust engine the runs execute in parallel across CPU cores, which is
    where grids, sweeps and null tests spend their time.

    Args:
        close: Wide close-price panel.
        weights: Target-weight panels, one per run.
        cfg: Portfolio config shared by every run.
        fee_bps: One fee for every run, or one per run.
        initial: Starting cash.
        warmup: Bars to skip before the first trade.
        engine: As for ``simulate``.

    Returns:
        One ``simulate`` result dict per panel, in order.
    """
    fees = [float(fee_bps)] * len(weights) if np.isscalar(fee_bps) else [float(f) for f in fee_bps]
    if len(fees) != len(weights):
        raise ValueError("fee_bps needs one value per weights panel")
    if _engine(engine) == "python":
        return [simulate(close, w, cfg, f, initial, warmup, engine="python")
                for w, f in zip(weights, fees, strict=True)]
    px = np.ascontiguousarray(close.to_numpy(float))
    Ws = [_arrays(close, w)[1] for w in weights]
    raws = _rs.simulate_many(px, Ws, fees, float(initial), int(warmup), int(cfg.check_bars),
                             float(cfg.band), float(cfg.min_trade_frac),
                             bool(cfg.rebalance), bool(cfg.band_all))
    return [_from_rust(close, raw, cfg, f, initial, warmup)
            for raw, f in zip(raws, fees, strict=True)]


def held_weights(res: dict, close: pd.DataFrame) -> pd.DataFrame:
    """The weights a ``simulate`` run actually held at each close, after drift.

    Rebuilt from the fill log: a buy adds ``(notional - fee) / price`` units and a
    sell removes ``notional / price``, both at the fill bar's close, exactly as
    the simulator books them. Whatever the columns don't sum to is cash.

    Args:
        res: Output of ``simulate`` on ``close``.
        close: The price panel the run was simulated on.

    Returns:
        Weights shaped like ``close``: each position's value over equity.
    """
    f = res["fills"]
    px = close.to_numpy(float)[close.index.get_indexer(f.ts), close.columns.get_indexer(f.token)]
    units = np.where(f.side == "BUY", f.notional - f.fee, -f.notional) / px
    qty = (pd.DataFrame({"ts": f.ts, "token": f.token, "units": units})
           .pivot_table(index="ts", columns="token", values="units", aggfunc="sum")
           .reindex(index=close.index, columns=close.columns).fillna(0.0).cumsum())
    return (qty * close.ffill()).div(res["equity"], axis=0).fillna(0.0)


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
