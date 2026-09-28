"""Animated equity races for sharing a backtest as a short video.

Any set of equity curves works: fee levels, portfolio sizes, strategies against
a benchmark. The x-axis grows as the simulation runs, so the current bar is
always the right edge; each line's return is tagged at its current value,
drawdowns fill in underneath, and the last frame is held so a takeaway caption
can be read. Titles and captions are written into the frame
because social feeds autoplay muted.

Typical use::

    from algo_trading.backtest import animate, plots
    plots.apply_theme()
    animate.equity_race({"strategy": eq, "buy & hold": bh}, "race.mp4",
                        title="EMA 20/50 vs buy & hold", subtitle="daily, 0.2% fee per fill",
                        caption="Same signal, half the drawdown.")

Frames are drawn by the Rust renderer in ``rust/animate`` when it is installed
(many times faster, all cores), otherwise by matplotlib. ``.mp4`` needs
``ffmpeg`` on the PATH; ``.gif`` needs ffmpeg with the Rust renderer and works
with Pillow alone under matplotlib.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plots import _FONT, DARK, LIGHT, _date_axis, _log_axis, _pct_axis, color, is_dark

try:  # the optional Rust renderer in rust/animate; see rust/README.md
    import algo_animate_rs as _rs
except ImportError:
    _rs = None

__all__ = ["equity_race", "concat"]

# Sizes are in points at the figure's dpi; a 1080px frame is shown ~400px wide
# on a phone, so everything is larger than the notebook theme.
_TEXT = {"title": 17, "subtitle": 11, "label": 10.5, "tick": 10, "date": 13, "caption": 13}


def concat(parts: Sequence[str | Path], path: str | Path) -> Path:
    """Join videos end to end into one file, without re-encoding.

    Use it to stitch several ``equity_race`` renders into a longer sequence. The
    parts must share size, frame rate and codec, which renders with the same
    ``size``, ``fps`` and engine do.

    Args:
        parts: Video files, in playing order.
        path: Output file.

    Returns:
        The path written.

    Raises:
        RuntimeError: If ffmpeg is missing or fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("concat needs ffmpeg on the PATH (e.g. `brew install ffmpeg`)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "parts.txt"
        quoted = (Path(p).resolve().as_posix().replace("'", "'\\''") for p in parts)
        listing.write_text("".join(f"file '{q}'\n" for q in quoted))   # ffmpeg concat list
        run = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                              "-i", str(listing), "-c", "copy", "-movflags", "+faststart",
                              str(path)], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {run.stderr.strip()}")
    return path


def _theme(dark: bool) -> dict[str, str]:
    """Colour tokens for the chosen surface."""
    return DARK if dark else LIGHT


def _rc(t: dict[str, str]) -> dict:
    """Matplotlib overrides that recolour the house theme for ``t``."""
    return {"figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"], "text.color": t["ink"],
            "axes.labelcolor": t["ink_muted"], "axes.edgecolor": t["grid"],
            "grid.color": t["grid"], "xtick.color": t["ink_muted"],
            "ytick.color": t["ink_muted"]}


def _growth(curves: dict[str, pd.Series]) -> pd.DataFrame:
    """Align curves on one index and rebase each to 1 at its first value."""
    df = pd.concat(curves, axis=1).sort_index()
    return df / df.apply(lambda s: s.loc[s.first_valid_index()])


def _writer(path: Path, fps: int, dpi: int):
    if path.suffix.lower() == ".gif":
        return PillowWriter(fps=fps)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"writing {path.suffix} needs ffmpeg on the PATH "
                           "(e.g. `brew install ffmpeg`); or save as .gif instead")
    return FFMpegWriter(fps=fps, codec="libx264", bitrate=4500,
                        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def _width(fig: Figure, text: str, **kw) -> float:
    """Rendered width of ``text`` as a fraction of the figure width."""
    t = fig.text(0, 0, text, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    t.remove()
    return w


def _spread(ys: np.ndarray, gap: float) -> np.ndarray:
    """Push positions apart by at least ``gap`` while keeping their order."""
    order = np.argsort(ys)
    placed = ys.astype(float).copy()
    for a, b in zip(order[:-1], order[1:], strict=True):
        placed[b] = max(placed[b], placed[a] + gap)
    return placed - (placed - ys).mean()     # re-centre so labels stay near their lines


class _Tips:
    """A dot on each line's current value with its label beside it.

    Tags sit just right of the dots (the axes' right edge, which tracks the
    current bar), are nudged apart when lines bunch together, and keep a thin
    leader line back to their dot so a nudged tag is never ambiguous.

    Args:
        ax: Growth axes.
        labels: Curve labels.
        colors: One colour per curve.
        ink: Tag text colour.
        surface: Background colour, used for the ring around each dot.
    """

    OFFSET = 14                               # points from the dot to its tag

    def __init__(self, ax: Axes, labels: Sequence[str], colors: Sequence[str],
                 ink: str, surface: str):
        self.ax, self.labels = ax, list(labels)
        self.dots = [ax.plot([], [], "o", ms=6.5, color=c, mec=surface, mew=1.5, zorder=6,
                             clip_on=False)[0] for c in colors]
        self.tags = [ax.annotate("", xy=(0, 1), xytext=(self.OFFSET, 0),
                                 textcoords="offset points", va="center",
                                 fontsize=_TEXT["label"], color=ink, annotation_clip=False,
                                 arrowprops={"arrowstyle": "-", "color": c, "lw": 0.9,
                                             "relpos": (0.0, 0.5),  # from the tag's left
                                             "shrinkA": 3, "shrinkB": 5})
                     for c in colors]

    def update(self, x, ys: np.ndarray) -> None:
        """Move dots to ``(x, ys)`` and re-place the tags. Call after ``set_xlim``."""
        pt_per_px = 72.0 / self.ax.figure.dpi
        y_pt = self.ax.transData.transform(
            np.column_stack([np.zeros(len(ys)), np.nan_to_num(ys, nan=1.0)]))[:, 1] * pt_per_px
        placed = _spread(y_pt, _TEXT["label"] * 1.45)
        for i, y in enumerate(ys):
            self.dots[i].set_data([x], [y])
            self.tags[i].xy = (x, y)
            self.tags[i].set_position((self.OFFSET, placed[i] - y_pt[i]))
            self.tags[i].set_text(self.labels[i])


def equity_race(curves: dict[str, pd.Series], path: str | Path, title: str,
                subtitle: str = "", caption: str = "", log: bool = True,
                colors: Sequence[str] | None = None, dd_for: str | None = None,
                seconds: float = 20.0, hold: float = 4.0, fps: int = 30,
                size: tuple[int, int] = (1080, 1080), dpi: int = 200,
                dark: bool | None = None, font: str | Path | None = None,
                icon: str | Path | None = None, progress: bool = True,
                engine: str = "auto") -> Path:
    """Render equity curves drawing in over time, with drawdowns underneath.

    The x-axis extends each frame so the current bar sits at the right edge,
    with each line's return tagged beside its current value. The y-axis is fixed
    from the full history so the scale doesn't jump. The final state is held for
    ``hold`` seconds with ``caption``.

    Args:
        curves: ``{label: equity series}`` on a datetime index. Each curve is
            rebased to 1 at its own first value, so warmups can differ.
        path: Output file; ``.mp4`` (needs ffmpeg) or ``.gif``.
        title: Headline, top left.
        subtitle: One line of context under the title (universe, bars, fees).
        caption: Takeaway shown under the chart on the held final frame.
        log: Use a log scale for the growth panel.
        colors: One colour per curve, e.g. ``plots.ramp(n)`` for an
            ordered sweep. Defaults to the categorical series colours (at most
            five), stepped for the chosen surface.
        dd_for: Label whose drawdown is a filled area. Defaults to the first.
        seconds: Length of the draw-in, excluding ``hold``.
        hold: Seconds to hold the final frame.
        fps: Frames per second.
        size: Output size in pixels, ``(width, height)``. 1080x1080 suits X feeds;
            use 1920x1080 for landscape.
        dpi: Render resolution; with ``size`` this sets how large text appears.
        dark: Black background with the dark-surface colour steps. None follows
            the active ``plots`` theme, which is dark by default. Custom
            ``colors`` should come from the same mode (``plots.ramp(n)`` does).
        font: A font file (``.ttf``/``.otf``) used for every piece of text.
            Defaults to the house sans-serif as matplotlib resolves it.
        icon: An SVG or PNG drawn at the left of the title block, with the
            title and subtitle beside it. SVG needs the Rust renderer.
        progress: Print a frame counter to stderr while rendering.
        engine: ``"rust"`` (parallel renderer, see rust/README.md), ``"mpl"``
            (matplotlib), or ``"auto"`` for Rust when it is installed.

    Returns:
        The path written.

    Raises:
        RuntimeError: If ``.mp4`` is requested without ffmpeg installed.
        ImportError: If ``engine="rust"`` but the Rust renderer isn't installed.
        IndexError: If more than five curves are given without ``colors``.
    """
    dark = is_dark() if dark is None else dark
    if engine not in ("auto", "rust", "mpl"):
        raise ValueError(f"engine must be 'auto', 'rust' or 'mpl', not {engine!r}")
    if engine == "rust" and _rs is None:
        raise ImportError("the Rust renderer is not installed; build it with "
                          "`maturin develop --release -m rust/animate/Cargo.toml`")
    for f in (font, icon):
        if f is not None and not Path(f).is_file():
            raise FileNotFoundError(f)
    args = (curves, Path(path), title, subtitle, caption, log, colors, dd_for, seconds, hold,
            fps, size, dpi, dark, font, icon, progress)
    if engine != "mpl" and _rs is not None:
        return _render_rs(*args)
    rc = _rc(_theme(dark))
    if font is not None:                       # register the file and put it first in line
        font_manager.fontManager.addfont(str(font))
        name = font_manager.FontProperties(fname=str(font)).get_name()
        rc |= {"font.family": "sans-serif", "font.sans-serif": [name, *_FONT]}
    with plt.rc_context(rc):
        return _render_mpl(*args)


def _font_face(weight: str) -> tuple[str, int]:
    """The font file and face index matplotlib would use for the house font list."""
    fname = font_manager.findfont(font_manager.FontProperties(family=_FONT, weight=weight))
    fm = font_manager.fontManager
    faces = [e for e in fm.ttflist if e.fname == fname]
    if not faces:
        return fname, 0
    best = min(faces, key=lambda e: (fm.score_weight(weight, e.weight)
                                     + fm.score_stretch("normal", e.stretch)
                                     + (e.style != "normal")))
    return fname, int(getattr(best, "index", 0))


def _render_rs(curves, path, title, subtitle, caption, log, colors, dd_for, seconds, hold, fps,
               size, dpi, dark, font, icon, progress) -> Path:
    """Hand the prepared arrays to the Rust renderer."""
    t = _theme(dark)
    path.parent.mkdir(parents=True, exist_ok=True)
    growth = _growth(curves)
    labels = [str(c) for c in growth.columns]
    cols = list(colors) if colors is not None else [color(i, dark) for i in range(len(labels))]
    idx = growth.index
    epoch = pd.Timestamp(0, tz=idx.tz) if getattr(idx, "tz", None) else pd.Timestamp(0)
    x = ((idx - epoch) / pd.Timedelta(days=1)).to_numpy(float)
    label_size = font_manager.FontProperties(
        size=plt.rcParams["axes.labelsize"]).get_size_in_points()
    regular, bold = ((str(font), 0),) * 2 if font else (_font_face("normal"), _font_face("bold"))
    out = _rs.equity_race(
        str(path), np.ascontiguousarray(x), np.ascontiguousarray(growth.to_numpy(float)),
        labels, cols, labels.index(dd_for or labels[0]), title, subtitle, caption, bool(log),
        float(seconds), float(hold), int(fps), int(size[0]), int(size[1]), float(dpi),
        t["ink"], t["ink_muted"], t["grid"], t["surface"], regular, bold,
        0.22 if dark else 0.15, float(label_size), icon=None if icon is None else str(icon),
        progress=progress)
    return Path(out)


def _render_mpl(curves, path, title, subtitle, caption, log, colors, dd_for, seconds, hold,
                fps, size, dpi, dark, font, icon, progress) -> Path:
    """Matplotlib renderer; rust/animate/src/race.rs is a port of this. Keep in step."""
    t = _theme(dark)
    path.parent.mkdir(parents=True, exist_ok=True)
    growth = _growth(curves)
    labels = list(growth.columns)
    cols = list(colors) if colors is not None else [color(i, dark) for i in range(len(labels))]
    primary = dd_for or labels[0]
    dd = (growth / growth.cummax() - 1) * 100
    idx = growth.index
    lo, hi = np.nanmin(growth.to_numpy()), np.nanmax(growth.to_numpy())

    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    pt = 1.0 / (size[1] / dpi * 72)            # one point as a fraction of the height
    wpt = 1.0 / (size[0] / dpi * 72)           # one point as a fraction of the width
    tx = 0.04                                  # left edge of the title block
    if icon is not None and Path(icon).suffix.lower() == ".svg":
        warnings.warn("SVG icons need the Rust renderer; drawing without the icon",
                      stacklevel=3)
    elif icon is not None:                     # left of the title block, spanning it
        img = plt.imread(str(icon))
        ih = (_TEXT["title"] * 1.4 + _TEXT["subtitle"] if subtitle else _TEXT["title"] * 1.2) * pt
        iw = ih * img.shape[1] / img.shape[0] * size[1] / size[0]
        icon_ax = fig.add_axes((0.04, 1 - 16 * pt - ih, iw, ih))
        icon_ax.imshow(img)
        icon_ax.axis("off")
        tx += iw + 10 * wpt

    # Vertical layout in points, top down, so it holds at any size/aspect.
    y = 1 - 16 * pt
    fig.text(tx, y, title, fontsize=_TEXT["title"], fontweight="bold", color=t["ink"],
             va="top")
    y -= _TEXT["title"] * 1.4 * pt
    if subtitle:
        fig.text(tx, y, subtitle, fontsize=_TEXT["subtitle"], color=t["ink_muted"],
                 va="top")
        y -= _TEXT["subtitle"] * 1.5 * pt
    top = y - _TEXT["date"] * 2.0 * pt
    cap_y = 10 * pt
    bottom = cap_y + (_TEXT["caption"] * 1.6 + _TEXT["tick"] * 2.2) * pt
    dd_h = 0.26 * (top - bottom)

    # Horizontal layout: the right margin holds the widest tag.
    tag_w = max(_width(fig, str(lab), fontsize=_TEXT["label"]) for lab in labels)
    left = (0.03 + plt.rcParams["axes.labelsize"] * 1.6 * wpt
            + _width(fig, "-100%", fontsize=_TEXT["tick"]) + 8 * wpt)
    width = 0.97 - tag_w - _Tips.OFFSET * wpt - left
    ax2 = fig.add_axes((left, bottom, width, dd_h))
    ax = fig.add_axes((left, bottom + dd_h + 14 * pt, width, top - bottom - dd_h - 14 * pt),
                      sharex=ax2)
    tips = _Tips(ax, labels, cols, t["ink"], t["surface"])

    date_txt = ax.text(0.0, 1.0, "", transform=ax.transAxes, fontsize=_TEXT["date"],
                       color=t["ink_muted"], fontweight="bold", va="bottom")
    date_txt.set_position((0.0, 1.0 + _TEXT["date"] * 0.5 * pt / ax.get_position().height))
    cap_txt = fig.text(0.04, cap_y, caption, fontsize=_TEXT["caption"], color=t["ink"],
                       va="bottom", alpha=0.0)

    lines, dd_lines = [], []
    for lab, col in zip(labels, cols, strict=True):
        is_primary = lab == primary
        lines.append(ax.plot([], [], color=col, lw=2.4 if is_primary else 1.6,
                             zorder=3 if is_primary else 2)[0])
        dd_lines.append(ax2.plot([], [], color=col, lw=1.4 if is_primary else 1.0)[0])

    ax.axhline(1.0, color=t["ink_muted"], lw=0.8, ls=":")
    if log:
        _log_axis(ax)
        ax.set_ylim(lo / 1.15, hi * 1.15)
    else:
        pad = (hi - lo) * 0.06
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("portfolio growth")
    ax.tick_params(labelbottom=False, labelsize=_TEXT["tick"])
    ax2.set_ylim(min(np.nanmin(dd.to_numpy()) * 1.08, -1), 2)
    ax2.set_ylabel("drawdown")
    ax2.tick_params(labelsize=_TEXT["tick"])
    _pct_axis(ax2)
    _date_axis(ax2)
    ax2.xaxis.get_major_formatter().offset_formats = [""] * 6   # the date label says it
    fill = None

    draw_frames, hold_frames = int(seconds * fps), int(hold * fps)
    total = draw_frames + hold_frames
    g, d = growth.to_numpy(), dd.to_numpy()
    g_tip = growth.ffill().to_numpy()          # a curve that has ended keeps its last value
    p = labels.index(primary)
    k0 = max(2, len(idx) // 30)                # open on a short window, not a single bar

    writer = _writer(path, fps, dpi)
    with writer.saving(fig, str(path), dpi):
        for f in range(total):
            k = min(len(idx), k0 + int(np.ceil((len(idx) - k0) * (f + 1) / draw_frames)))
            x = idx[:k]
            ax.set_xlim(idx[0], x[-1])         # the current bar is always the right edge
            for i in range(len(labels)):
                lines[i].set_data(x, g[:k, i])
                dd_lines[i].set_data(x, d[:k, i])
            if fill is not None:
                fill.remove()
            fill = ax2.fill_between(x, np.nan_to_num(d[:k, p]), 0, color=cols[p],
                                    alpha=0.22 if dark else 0.15, lw=0)
            tips.update(x[-1], g_tip[k - 1])
            date_txt.set_text(x[-1].strftime("%b %Y"))
            if f >= draw_frames and caption:
                cap_txt.set_alpha(min(1.0, (f - draw_frames + 1) / (0.6 * fps)))
            writer.grab_frame(facecolor=t["surface"])
            if progress and (f % fps == 0 or f == total - 1):
                print(f"\r  rendering {path.name}: frame {f + 1}/{total}", end="",
                      file=sys.stderr)
    if progress:
        print(file=sys.stderr)
    plt.close(fig)
    return path
