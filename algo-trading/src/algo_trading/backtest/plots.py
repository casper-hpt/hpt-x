"""Matplotlib visuals: one theme, applied once, then small composable charts.

The theme is dark by default (black surface); ``apply_theme(dark=False)``
switches to a warm off-white one. Each mode has its own colour-vision-deficiency
validated steps of the same hues, not an automatic inversion. Series slots are
assigned in fixed order and never cycled, so a strategy keeps its colour across
every chart in a notebook. Every multi-series chart carries a legend and, where
there is room, direct labels, so colour is never the only cue.

Every chart function returns its figure/axes so notebooks can tweak further, and
accepts an ``ax`` where it makes sense to compose charts into a grid.

Typical use::

    from algo_trading.backtest import plots
    plots.apply_theme()
    plots.equity_drawdown({"strategy": eq, "buy & hold": bh})
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

__all__ = ["SERIES", "INK", "INK_MUTED", "GRID", "SURFACE", "BLUES", "PURPLES", "LIGHT", "DARK",
           "apply_theme", "is_dark", "color", "ramp", "equity_drawdown", "price_signal",
           "exposure", "weight_bars",
           "heatmap", "null_hist", "sweep", "fee_curve", "return_hist"]

# ── Tokens ───────────────────────────────────────────────────────────────────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# The brand violet (#a78bfa, the HPT icon) as a ramp: used for videos.
PURPLES = LinearSegmentedColormap.from_list(
    "purples", ["#ede9fe", "#c4b5fd", "#a78bfa", "#7c3aed", "#4c1d95"])
# Ramp positions (least -> most prominent) per hue and surface. Each run was
# validated as an ordinal ramp: monotone, visible steps, >= 2:1 at the faint end.
_RAMPS = {"blue": (BLUES, (0.3, 1.0), (0.75, 0.0)),
          "purple": (PURPLES, (0.42, 1.0), (0.95, 0.2))}   # dark: stays violet, never white

# Dark mode: the same hues re-stepped for a black surface (validated there), not
# an automatic inversion. Used by the video renderer.
DARK = {
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    "ink": "#ffffff", "ink_muted": "#c3c2b7", "grid": "#2e2e2c", "surface": "#000000",
}

LIGHT = {"series": SERIES, "ink": INK, "ink_muted": INK_MUTED, "grid": GRID, "surface": SURFACE}

_FONT = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
_ACTIVE = {"dark": True}                  # set by apply_theme; dark is the default


def is_dark() -> bool:
    """Whether the active theme is dark."""
    return _ACTIVE["dark"]


def _tok(name: str) -> str:
    """Look up a colour token (ink, ink_muted, grid, surface) in the active theme."""
    return (DARK if is_dark() else LIGHT)[name]


def apply_theme(dark: bool = True) -> None:
    """Apply the house style to matplotlib. Call once at the top of a notebook.

    Args:
        dark: Black surface with the dark-stepped palette (the default). Pass
            False for the light theme. Charts drawn afterwards follow this choice.
    """
    _ACTIVE["dark"] = dark
    plt.rcParams.update({
        "figure.facecolor": _tok("surface"), "axes.facecolor": _tok("surface"),
        "savefig.facecolor": _tok("surface"),
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": _FONT, "font.size": 10,
        "text.color": _tok("ink"), "axes.labelcolor": _tok("ink_muted"),
        "axes.titlecolor": _tok("ink"),
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 9.5,
        "axes.edgecolor": _tok("grid"), "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": _tok("grid"), "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "xtick.color": _tok("ink_muted"), "ytick.color": _tok("ink_muted"),
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": _tok("ink_muted"),
        "axes.prop_cycle": plt.cycler(color=_tok("series")),
    })


def color(i: int, dark: bool | None = None) -> str:
    """Return the ``i``-th series colour.

    Args:
        i: Series slot, starting at 0.
        dark: Return the dark-surface step of that slot. None follows the
            active theme.

    Returns:
        A hex colour.

    Raises:
        IndexError: If more slots are requested than the palette holds. Fold the
            extra series into "other" or use small multiples instead of cycling.
    """
    series = DARK["series"] if (is_dark() if dark is None else dark) else SERIES
    if i >= len(series):
        raise IndexError(f"only {len(series)} series colours; fold extras into 'other' "
                         f"or use small multiples")
    return series[i]


def ramp(n: int, dark: bool | None = None, hue: str = "blue") -> list[str]:
    """Return ``n`` evenly spaced steps of a one-hue ramp for an ordered sweep.

    Use this for an ordered sweep (1, 2, 4, 8, ...), where the value is a
    magnitude rather than an identity, so it can take any number of steps. The
    last step is always the most prominent against the surface: darkest on
    light, lightest on dark. Both ends clear 2:1 contrast against their surface.

    Args:
        n: Number of colours.
        dark: Step for a dark surface (mid tone to near-white) instead of light
            (light tint to deep). None follows the active theme.
        hue: ``"blue"`` (the chart default) or ``"purple"`` (the brand violet).

    Returns:
        Hex colours, least to most prominent.
    """
    if hue not in _RAMPS:
        raise ValueError(f"hue must be one of {sorted(_RAMPS)}, not {hue!r}")
    cmap, light, dark_run = _RAMPS[hue]
    lo, hi = dark_run if (is_dark() if dark is None else dark) else light
    return [to_hex(cmap(v)) for v in np.linspace(lo, hi, n)]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _date_axis(ax: Axes) -> None:
    """Concise, auto-spaced date ticks that read well from days to years."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _new_ax(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        return fig, ax
    return ax.figure, ax


def _end_labels(ax: Axes, ends: list[tuple], min_gap: float = 12.0) -> None:
    """Direct labels at the right end of lines, nudged apart so none collide.

    Args:
        ax: Axes holding the lines (limits and scale must already be final).
        ends: ``(x, y, text, colour)`` per line.
        min_gap: Minimum vertical spacing between labels, in points.
    """
    fig = ax.figure
    fig.canvas.draw()                        # settle limits/layout before measuring
    to_pt = 72.0 / fig.dpi
    ys = [ax.transData.transform((mdates.date2num(x) if isinstance(x, pd.Timestamp) else x, y))[1]
          * to_pt for x, y, *_ in ends]
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)
    for k in range(1, len(order)):           # push each label up until it clears the one below
        lo, hi = order[k - 1], order[k]
        placed[hi] = max(placed[hi], placed[lo] + min_gap)
    ax.autoscale(False)                      # the dots must not move the limits we measured
    for (x, y, text, col), y0, y1 in zip(ends, ys, placed):
        ax.scatter([x], [y], s=22, color=col, zorder=5, edgecolor=_tok("surface"), linewidth=1.5)
        ax.annotate(text, xy=(x, y), xytext=(7, y1 - y0), textcoords="offset points",
                    va="center", fontsize=8.5, color=_tok("ink_muted"))


def _log_axis(ax: Axes) -> None:
    """Log y-scale labelled in plain numbers at 1, 2, 5 x 10^k."""
    def label(v: float, _) -> str:
        if v <= 0:
            return ""
        mantissa = round(v / 10 ** np.floor(np.log10(v)), 6)
        return f"{v:g}" if mantissa in (1.0, 2.0, 5.0) else ""

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(label))
    ax.yaxis.set_minor_formatter(FuncFormatter(label))


def _price_axis(ax: Axes) -> None:
    """Thousands separators on a price axis; keeps small prices readable."""
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.4g}"))


def _pct_axis(ax: Axes, axis: str = "y", decimals: int = 0) -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ── Charts ───────────────────────────────────────────────────────────────────

def equity_drawdown(curves: dict[str, pd.Series], title: str = "Equity",
                    dd_for: str | None = None, log: bool = False,
                    colors: Sequence[str] | None = None,
                    figsize: tuple[float, float] = (11, 6.2)) -> tuple[Figure, np.ndarray]:
    """Growth-of-1 lines over a drawdown panel.

    Args:
        curves: ``{label: equity series}``. The first is the primary series.
            At most five unless ``colors`` is given.
        title: Chart title.
        dd_for: Label whose drawdown is drawn as a filled area; the rest are
            lines. Defaults to the first curve.
        log: Use a log scale for the growth panel.
        colors: One colour per curve, e.g. ``ramp(n)`` for an ordered sweep.
            Defaults to the categorical series colours.
        figsize: Figure size in inches.

    Returns:
        ``(fig, axes)`` where ``axes`` is ``[growth_ax, drawdown_ax]``.
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, layout="constrained",
                                  gridspec_kw={"height_ratios": [3, 1]})
    primary = dd_for or next(iter(curves))
    ends = []
    for i, (lab, s) in enumerate(curves.items()):
        s = s.dropna()
        g = s / s.iloc[0]
        col = colors[i] if colors is not None else color(i)
        ax.plot(g.index, g, color=col, label=lab, lw=2.2 if lab == primary else 1.5,
                zorder=3 if lab == primary else 2)
        ends.append((g.index[-1], g.iloc[-1], f"{lab}  {g.iloc[-1] - 1:+.0%}", col))

        dd = (s / s.cummax() - 1) * 100
        if lab == primary:
            ax2.fill_between(dd.index, dd, 0, color=col, alpha=0.18 if is_dark() else 0.3,
                             lw=0)
            ax2.plot(dd.index, dd, color=col, lw=1.3, zorder=3)
        else:
            ax2.plot(dd.index, dd, color=col, lw=1.1)

    ax.axhline(1.0, color=_tok("ink_muted"), lw=0.8, ls=":")
    if log:
        _log_axis(ax)
    ax.set_title(title)
    ax.set_ylabel("portfolio growth")
    if len(curves) > 3:     # a long legend goes above the plot, level with the title
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(curves),
                  borderaxespad=0.2, handlelength=1.4, columnspacing=1.2)
    else:
        ax.legend(loc="upper left")
    ax.margins(x=0.14 if len(curves) <= 4 else 0.02)
    ax2.set_ylabel("drawdown")
    _pct_axis(ax2)
    _date_axis(ax2)
    if len(curves) <= 4:
        _end_labels(ax, ends)
    return fig, np.array([ax, ax2])


def price_signal(close: pd.Series, signal: pd.Series, token: str,
                 bands: dict[str, pd.Series] | None = None, start: int = 0,
                 log: bool = False,
                 figsize: tuple[float, float] = (11, 5.4)) -> tuple[Figure, np.ndarray]:
    """Price with entries marked and the position underneath.

    Args:
        close: One token's close series.
        signal: That token's 0/1 position series.
        token: Token id, used in the title.
        bands: Optional overlays such as channel edges or moving averages,
            ``{label: series}``, at most four.
        start: Bars to skip from the front.
        log: Use a log price scale.
        figsize: Figure size in inches.

    Returns:
        ``(fig, axes)`` where ``axes`` is ``[price_ax, position_ax]``.
    """
    sl = slice(start, None)
    c, sig = close.iloc[sl], signal.reindex(close.index).fillna(0).iloc[sl]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, layout="constrained",
                                  gridspec_kw={"height_ratios": [4, 1]})
    ax.plot(c.index, c, color=_tok("ink"), lw=1.3, label="close")
    for i, (lab, s) in enumerate((bands or {}).items(), start=1):
        s = s.iloc[sl]
        ax.plot(s.index, s, color=color(i), lw=1.3, label=lab)

    entries = sig.index[(sig > 0) & (sig.shift(1, fill_value=0) <= 0)]
    if len(entries):
        ax.plot(entries, c.reindex(entries), ls="none", marker="o", ms=7, color=color(0),
                mec=_tok("surface"), mew=1.5, zorder=5, label="entry")
    if log:
        _log_axis(ax)
    else:
        _price_axis(ax)
    ax.set_title(f"{token.upper()} — price and position")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", ncol=5)

    ax2.fill_between(sig.index, sig, 0, step="post", color=color(0), alpha=0.85, lw=0)
    ax2.set_ylabel("long")
    ax2.set_yticks([0, 1])
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", visible=False)
    _date_axis(ax2)
    return fig, np.array([ax, ax2])


def exposure(signal: pd.DataFrame, start: int = 0, top: int = 12,
             figsize: tuple[float, float] = (11, 3.6)) -> tuple[Figure, np.ndarray]:
    """Concurrency over time, and share of bars long per token.

    Args:
        signal: 0/1 signal panel.
        start: Bars to skip from the front.
        top: Tokens to show in the bar chart.
        figsize: Figure size in inches.

    Returns:
        ``(fig, axes)`` where ``axes`` is ``[concurrency_ax, per_token_ax]``.
    """
    live = signal.iloc[start:].fillna(0)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize, layout="constrained",
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    n_long = live.sum(axis=1)
    ax.fill_between(n_long.index, n_long, 0, step="post", color=color(0), alpha=0.25, lw=0)
    ax.step(n_long.index, n_long, where="post", color=color(0), lw=1.4)
    ax.set_title("Names long at once")
    ax.set_ylabel("count")
    ax.set_ylim(bottom=0)
    _date_axis(ax)

    weight_bars(live, top=top, title="Share of bars long", xlabel="% of bars", ax=ax2)
    return fig, np.array([ax, ax2])


def weight_bars(weights: pd.DataFrame, start: int = 0, top: int = 10,
                title: str = "Mean target weight", xlabel: str = "weight",
                figsize: tuple[float, float] = (6, 3.8), ax: Axes | None = None) -> Axes:
    """Horizontal bars of the mean weight (or duty cycle) per token.

    Args:
        weights: Weight or 0/1 signal panel.
        start: Bars to skip from the front.
        top: Tokens to show, largest first.
        title: Chart title.
        xlabel: X-axis label.
        figsize: Figure size in inches, when creating a new figure.
        ax: Axes to draw into; a new figure is created if None.

    Returns:
        The axes drawn on.
    """
    _, ax = _new_ax(ax, figsize)
    mean_w = weights.iloc[start:].mean().sort_values(ascending=False).head(top)[::-1] * 100
    bars = ax.barh(mean_w.index, mean_w.values, color=color(0), height=0.7)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8, color=_tok("ink_muted"))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    _pct_axis(ax, "x")
    return ax


def heatmap(df: pd.DataFrame, title: str, fmt: str = "{:.2f}", ax: Axes | None = None,
            xlabel: str = "", ylabel: str = "", cmap=None,
            figsize: tuple[float, float] = (6, 4.2)) -> Axes:
    """Annotated matrix, for parameter surfaces and stability tables.

    Low values recede toward the surface and high values stand out from it, so
    the blue ramp runs light-to-navy on the light theme and navy-to-light on dark.

    Args:
        df: Values to show; index is rows, columns are columns.
        title: Chart title.
        fmt: Format spec for cell annotations.
        ax: Axes to draw into; a new figure is created if None.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        cmap: Sequential colormap (single hue). Defaults to the blue ramp,
            oriented for the active theme.
        figsize: Figure size in inches, when creating a new figure.

    Returns:
        The axes drawn on.
    """
    _, ax = _new_ax(ax, figsize)
    if cmap is None:
        cmap = BLUES.reversed() if is_dark() else BLUES
    strong_text = "#000000" if is_dark() else "#ffffff"   # on the most prominent cells
    vals = df.to_numpy(float)
    im = ax.imshow(vals, cmap=cmap, aspect="auto")
    ax.set_xticks(range(df.shape[1]), [str(c) for c in df.columns])
    ax.set_yticks(range(df.shape[0]), [str(i) for i in df.index])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_visible(False)

    lo, hi = im.get_clim()
    mid = lo + 0.55 * (hi - lo)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                        color=strong_text if v > mid else _tok("ink"))
    return ax


def null_hist(null: pd.DataFrame, real: dict, metrics: Sequence[str] = ("total_ret", "sharpe"),
              reference: dict | None = None,
              figsize: tuple[float, float] = (11, 3.6)) -> tuple[Figure, np.ndarray]:
    """Null distributions with the real result marked.

    Args:
        null: Output of a null function, one row per simulation.
        real: The real strategy's ``perf`` dict.
        metrics: Columns to plot, one panel each. ``total_ret`` is shown in %.
        reference: Optional second ``perf`` dict (e.g. an exposure-matched book)
            drawn as a dashed line; its ``label`` is used in the legend.
        figsize: Figure size in inches.

    Returns:
        ``(fig, axes)``, one axes per metric.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, layout="constrained")
    axes = np.atleast_1d(axes)
    for a, m in zip(axes, metrics):
        scale = 100 if m == "total_ret" else 1
        col = null[m].replace([np.inf, -np.inf], np.nan).dropna()
        p = float((col >= real[m]).mean())
        a.hist(col * scale, bins=30, color=color(0), alpha=0.55, edgecolor=_tok("surface"), lw=0.8)
        a.axvline(real[m] * scale, color=color(1), lw=2.2, label=f"real (p = {p:.3f})")
        if reference is not None:
            a.axvline(reference[m] * scale, color=color(3), lw=1.8, ls="--",
                      label=reference.get("label", "reference").strip())
        a.set_title(f"Null distribution — {m}")
        a.set_ylabel("simulations")
        a.grid(axis="x", visible=False)
        if scale == 100:
            _pct_axis(a, "x")
        a.legend(loc="upper left")
    return fig, axes


def sweep(x: Sequence, series: dict[str, Sequence[float]], title: str, xlabel: str,
          ylabel: str, hlines: dict[str, float] | None = None, ax: Axes | None = None,
          figsize: tuple[float, float] = (8, 3.8)) -> Axes:
    """Line chart for a parameter sweep, with optional reference lines.

    Args:
        x: Parameter values.
        series: ``{label: y values}``.
        title: Chart title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        hlines: ``{label: y}`` dashed horizontal references.
        ax: Axes to draw into; a new figure is created if None.
        figsize: Figure size in inches, when creating a new figure.

    Returns:
        The axes drawn on.
    """
    _, ax = _new_ax(ax, figsize)
    for i, (lab, ys) in enumerate(series.items()):
        ax.plot(x, ys, color=color(i), marker="o", ms=6, mec=_tok("surface"), mew=1.5, label=lab)
    for j, (lab, y) in enumerate((hlines or {}).items(), start=len(series)):
        ax.axhline(y, color=color(j), ls="--", lw=1.4, label=lab)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return ax


def fee_curve(fee_grid: Sequence[float], returns: Sequence[float],
              benchmark: float | None = None, benchmark_label: str = "benchmark",
              our_fee: float = 20.0, ax: Axes | None = None,
              figsize: tuple[float, float] = (8.5, 3.9)) -> Axes:
    """Total return as a function of cost per fill.

    Args:
        fee_grid: Fees in basis points per fill.
        returns: Total return (as a fraction) at each fee.
        benchmark: Optional benchmark total return, drawn as a dashed line.
        benchmark_label: Legend label for the benchmark.
        our_fee: Our assumed fee in bps, marked with a vertical line.
        ax: Axes to draw into; a new figure is created if None.
        figsize: Figure size in inches, when creating a new figure.

    Returns:
        The axes drawn on.
    """
    _, ax = _new_ax(ax, figsize)
    ax.plot(fee_grid, [r * 100 for r in returns], color=color(0), marker="o", ms=6,
            mec=_tok("surface"), mew=1.5, label="strategy")
    if benchmark is not None:
        ax.axhline(benchmark * 100, color=color(1), ls="--", lw=1.5, label=benchmark_label)
    ax.axvline(our_fee, color=_tok("ink_muted"), lw=1.0, ls=":")
    ax.annotate(f"our cost {our_fee / 100:.1f}% / fill", xy=(our_fee, 1), xytext=(4, -4),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                va="top", fontsize=8.5, color=_tok("ink_muted"))
    ax.set_title("Return vs cost per fill")
    ax.set_xlabel("fee (bps per fill)")
    ax.set_ylabel("total return")
    _pct_axis(ax)
    ax.legend(loc="upper right")
    return ax


def return_hist(returns: pd.Series, title: str = "Trade returns", bins: int = 40,
                ax: Axes | None = None, figsize: tuple[float, float] = (8, 3.6)) -> Axes:
    """Histogram of per-trade (or per-bar) returns, split at zero.

    Args:
        returns: Returns as fractions.
        title: Chart title.
        bins: Number of histogram bins.
        ax: Axes to draw into; a new figure is created if None.
        figsize: Figure size in inches, when creating a new figure.

    Returns:
        The axes drawn on.
    """
    _, ax = _new_ax(ax, figsize)
    r = returns.replace([np.inf, -np.inf], np.nan).dropna() * 100
    edges = np.histogram_bin_edges(r, bins=bins)
    ax.hist(r[r >= 0], bins=edges, color=color(0), edgecolor=_tok("surface"), lw=0.8,
            label=f"wins ({(r >= 0).mean():.0%})")
    ax.hist(r[r < 0], bins=edges, color=color(1), edgecolor=_tok("surface"), lw=0.8,
            label=f"losses ({(r < 0).mean():.0%})")
    ax.axvline(r.mean(), color=_tok("ink"), lw=1.2, ls="--")
    ax.annotate(f"mean {r.mean():+.1f}%", xy=(r.mean(), 1), xytext=(4, -4),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                va="top", fontsize=8.5, color=_tok("ink_muted"))
    ax.set_title(title)
    ax.set_xlabel("return")
    ax.set_ylabel("count")
    ax.grid(axis="x", visible=False)
    _pct_axis(ax, "x")
    ax.legend(loc="upper right")
    return ax
