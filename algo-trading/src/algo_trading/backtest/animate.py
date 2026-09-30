"""Equity races, allocation stills, risk bars, bell curves and explainers: backtests as video.

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

import logging
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .plots import _FONT, DARK, LIGHT, _date_axis, _log_axis, _pct_axis, color, is_dark, ramp

try:  # the optional Rust renderer in rust/animate; see rust/README.md
    import algo_animate_rs as _rs
except ImportError:
    _rs = None

__all__ = ["equity_race", "allocation_still", "risk_bars", "risk_contributions",
           "bell_morph", "vol_explainer", "concat", "add_audio"]

# Sizes are in points at the figure's dpi; a 1080px frame is shown ~400px wide
# on a phone, so everything is larger than the notebook theme.
_TEXT = {"title": 17, "subtitle": 11, "label": 10.5, "tick": 10, "date": 13, "caption": 13}


def _ffmpeg(args: list[str], what: str) -> None:
    """Run ffmpeg quietly; raise with its error output if it fails."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"{what} needs ffmpeg on the PATH (e.g. `brew install ffmpeg`)")
    run = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True,
                         text=True)
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg {what} failed: {run.stderr.strip()}")


def concat(parts: Sequence[str | Path], path: str | Path, audio: str | Path | None = None,
           audio_start: float = 0.0, fade: float = 1.5) -> Path:
    """Join videos end to end into one file, without re-encoding the video.

    Use it to stitch several ``equity_race`` renders into a longer sequence. The
    parts must share size, frame rate and codec, which renders with the same
    ``size``, ``fps`` and engine do. ``audio`` adds a soundtrack to the result;
    see ``add_audio``.

    Args:
        parts: Video files, in playing order.
        path: Output file.
        audio: Optional audio file (``.mp3``, ``.wav``, ...) played under the video.
        audio_start: Seconds into ``audio`` to start from.
        fade: Seconds of fade-out at the end of the video; 0 for none.

    Returns:
        The path written.

    Raises:
        RuntimeError: If ffmpeg is missing or fails.
        FileNotFoundError: If ``audio`` doesn't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "parts.txt"
        quoted = (Path(p).resolve().as_posix().replace("'", "'\\''") for p in parts)
        listing.write_text("".join(f"file '{q}'\n" for q in quoted))   # ffmpeg concat list
        joined = path if audio is None else Path(tmp) / f"joined{path.suffix}"
        _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                 "-movflags", "+faststart", str(joined)], "concat")
        if audio is not None:
            add_audio(joined, audio, path, start=audio_start, fade=fade)
    return path


def add_audio(video: str | Path, audio: str | Path, path: str | Path, start: float = 0.0,
              fade: float = 1.5) -> Path:
    """Put a soundtrack under a video; the video stream is copied, not re-encoded.

    The audio plays from ``start`` seconds into the file and is cut at the end of
    the video, fading out over the last ``fade`` seconds. Audio shorter than the
    video just ends early; the rest plays silent.

    Args:
        video: The video file (e.g. from ``equity_race`` or ``concat``).
        audio: Audio file, anything ffmpeg reads (``.mp3``, ``.wav``, ``.m4a``).
        path: Output file; may not be ``video`` itself.
        start: Seconds into ``audio`` to start from.
        fade: Seconds of fade-out at the end of the video; 0 for none.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If ``video`` or ``audio`` doesn't exist.
        ValueError: If ``start`` or ``fade`` is negative, or ``path`` is ``video``.
        RuntimeError: If ffmpeg is missing or fails.
    """
    video, audio, path = Path(video), Path(audio), Path(path)
    for f in (video, audio):
        if not f.is_file():
            raise FileNotFoundError(f)
    if start < 0 or fade < 0:
        raise ValueError("start and fade must be >= 0")
    if path.resolve() == video.resolve():
        raise ValueError("write the result to a new path; ffmpeg can't edit a file in place")
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffprobe") is None:
        raise RuntimeError("add_audio needs ffprobe on the PATH (it ships with ffmpeg)")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(video)], capture_output=True, text=True,
                           check=True)
    length = float(probe.stdout.strip())
    af = "apad"                                # silence after a short track, cut at `length`
    if fade > 0:
        af += f",afade=t=out:st={max(0.0, length - fade):.3f}:d={min(fade, length):.3f}"
    _ffmpeg(["-i", str(video), "-ss", f"{start:.3f}", "-i", str(audio),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", af,
             "-c:a", "aac", "-b:a", "192k", "-t", f"{length:.3f}", "-movflags", "+faststart",
             str(path)], "add_audio")
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
                engine: str = "auto", fees: dict[str, pd.Series] | None = None,
                drawdown: bool | None = None,
                widths: dict[str, float] | None = None) -> Path:
    """Render equity curves drawing in over time, with drawdowns and/or fees underneath.

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
        fees: ``{label: cumulative fees paid}`` in currency, one per curve. If
            given, a panel of fees paid is drawn under the growth chart.
        drawdown: Draw the drawdown panel. None draws it unless ``fees`` is
            given; True with ``fees`` stacks both, drawdowns above fees.
        widths: ``{label: line width}`` in points for the growth panel, overriding
            the default (2.4 for ``dd_for``, 1.6 for the rest), e.g. to thin a
            benchmark. The lower panels scale to match.

    Returns:
        The path written.

    Raises:
        RuntimeError: If ``.mp4`` is requested without ffmpeg installed.
        ImportError: If ``engine="rust"`` but the Rust renderer isn't installed.
        IndexError: If more than five curves are given without ``colors``.
    """
    dark = is_dark() if dark is None else dark
    use_rs = _use_rust(engine)
    for f in (font, icon):
        if f is not None and not Path(f).is_file():
            raise FileNotFoundError(f)
    drawdown = fees is None if drawdown is None else drawdown
    if not drawdown and fees is None:
        raise ValueError("drawdown=False needs fees, or there is nothing to draw underneath")
    args = (curves, Path(path), title, subtitle, caption, log, colors, dd_for, seconds, hold,
            fps, size, dpi, dark, font, icon, progress)
    if use_rs:
        return _render_rs(*args, fees=fees, drawdown=drawdown, widths=widths)
    with _quiet_fonts(), plt.rc_context(_style(dark, font)):
        return _render_mpl(*args, fees=fees, drawdown=drawdown, widths=widths)


def _use_rust(engine: str) -> bool:
    """Resolve ``engine`` to whether the Rust renderer draws the frames."""
    if engine not in ("auto", "rust", "mpl"):
        raise ValueError(f"engine must be 'auto', 'rust' or 'mpl', not {engine!r}")
    if engine == "rust" and _rs is None:
        raise ImportError("the Rust renderer is not installed; build it with `make rust-animate`")
    return engine != "mpl" and _rs is not None


def _rs_fonts(font: str | Path | None) -> tuple[tuple[str, int], tuple[str, int]]:
    """Regular and bold ``(file, face)`` for the Rust renderer: ``font`` for both, if given."""
    return ((str(font), 0),) * 2 if font else (_font_face("normal"), _font_face("bold"))


def _line_widths(labels: list, primary, widths: dict | None) -> list[tuple[float, float]]:
    """``(growth, lower panel)`` line width per label: the defaults unless ``widths`` names it."""
    out = []
    for lab in labels:
        g, lo = (2.4, 1.4) if lab == primary else (1.6, 1.0)
        if widths and lab in widths:
            g, lo = float(widths[lab]), float(widths[lab]) * 0.6
        out.append((g, lo))
    return out


def _fee_panel(fees: dict[str, pd.Series], index: pd.Index, labels: list) -> pd.DataFrame:
    """Cumulative fees on the curves' index, one column per label, carried across gaps."""
    return pd.concat(fees, axis=1).reindex(index).ffill().fillna(0.0)[labels]


def _style(dark: bool, font: str | Path | None) -> dict:
    """rc overrides for a matplotlib-rendered video: theme colours and ``font`` first."""
    rc = _rc(_theme(dark)) | {"text.parse_math": False}      # captions may quote dollars
    if font is not None:                       # register the file and put it first in line
        font_manager.fontManager.addfont(str(font))
        name = font_manager.FontProperties(fname=str(font)).get_name()
        rc |= {"font.family": "sans-serif", "font.sans-serif": [name, *_FONT]}
    return rc


def _title_block(fig: Figure, t: dict[str, str], title: str, subtitle: str,
                 icon: str | Path | None, size: tuple[int, int]) -> float:
    """Draw the icon, title and subtitle at the top left; return the y below them.

    Layout is in points, top down, so it holds at any size/aspect."""
    pt = 1.0 / (fig.get_figheight() * 72)      # one point as a fraction of the height
    wpt = 1.0 / (fig.get_figwidth() * 72)      # one point as a fraction of the width
    tx = 0.04                                  # left edge of the title block
    if icon is not None and Path(icon).suffix.lower() == ".svg":
        warnings.warn("SVG icons need the Rust renderer; drawing without the icon",
                      stacklevel=4)
    elif icon is not None:                     # left of the title block, spanning it
        img = plt.imread(str(icon))
        ih = (_TEXT["title"] * 1.4 + _TEXT["subtitle"] if subtitle else _TEXT["title"] * 1.2) * pt
        iw = ih * img.shape[1] / img.shape[0] * size[1] / size[0]
        icon_ax = fig.add_axes((0.04, 1 - 16 * pt - ih, iw, ih))
        icon_ax.imshow(img)
        icon_ax.axis("off")
        tx += iw + 10 * wpt
    y = 1 - 16 * pt
    fig.text(tx, y, title, fontsize=_TEXT["title"], fontweight="bold", color=t["ink"],
             va="top")
    y -= _TEXT["title"] * 1.4 * pt
    if subtitle:
        fig.text(tx, y, subtitle, fontsize=_TEXT["subtitle"], color=t["ink_muted"], va="top")
        y -= _TEXT["subtitle"] * 1.5 * pt
    return y


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
               size, dpi, dark, font, icon, progress, fees=None, drawdown=True,
               widths=None) -> Path:
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
    regular, bold = _rs_fonts(font)
    fee_arr = (None if fees is None
               else np.ascontiguousarray(_fee_panel(fees, idx, labels).to_numpy(float)))
    primary = labels.index(dd_for or labels[0])
    out = _rs.equity_race(
        str(path), np.ascontiguousarray(x), np.ascontiguousarray(growth.to_numpy(float)),
        labels, cols, primary, title, subtitle, caption, bool(log),
        float(seconds), float(hold), int(fps), int(size[0]), int(size[1]), float(dpi),
        t["ink"], t["ink_muted"], t["grid"], t["surface"], regular, bold,
        0.22 if dark else 0.15, float(label_size), icon=None if icon is None else str(icon),
        progress=progress, fees=fee_arr, drawdown=bool(drawdown),
        widths=_line_widths(labels, labels[primary], widths))
    return Path(out)


def _render_mpl(curves, path, title, subtitle, caption, log, colors, dd_for, seconds, hold,
                fps, size, dpi, dark, font, icon, progress, fees=None, drawdown=True,
                widths=None) -> Path:
    """Matplotlib renderer; rust/animate/src/race.rs is a port of this. Keep in step.

    ``fees`` (matplotlib only) adds a panel of cumulative fees paid, below the
    drawdown panel or, with ``drawdown=False``, in place of it."""
    t = _theme(dark)
    path.parent.mkdir(parents=True, exist_ok=True)
    growth = _growth(curves)
    labels = list(growth.columns)
    cols = list(colors) if colors is not None else [color(i, dark) for i in range(len(labels))]
    primary = dd_for or labels[0]
    panels = []                                # (kind, values), top to bottom
    if drawdown:
        panels.append(("drawdown", (growth / growth.cummax() - 1) * 100))
    if fees is not None:                       # cumulative, so carry across any gaps
        panels.append(("fees", _fee_panel(fees, growth.index, labels)))
    idx = growth.index
    lo, hi = np.nanmin(growth.to_numpy()), np.nanmax(growth.to_numpy())

    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    pt = 1.0 / (size[1] / dpi * 72)            # one point as a fraction of the height
    wpt = 1.0 / (size[0] / dpi * 72)           # one point as a fraction of the width
    y = _title_block(fig, t, title, subtitle, icon, size)
    top = y - _TEXT["date"] * 2.0 * pt
    cap_y = 10 * pt
    bottom = cap_y + (_TEXT["caption"] * 1.6 + _TEXT["tick"] * 2.2) * pt
    gap = 10 * pt                              # between stacked lower panels
    lower_h = (0.26 if len(panels) == 1 else 0.40) * (top - bottom)
    dd_h = (lower_h - gap * (len(panels) - 1)) / len(panels)

    # Horizontal layout: the right margin holds the widest tag.
    tag_w = max(_width(fig, str(lab), fontsize=_TEXT["label"]) for lab in labels)
    left = (0.03 + plt.rcParams["axes.labelsize"] * 1.6 * wpt
            + _width(fig, "-100%", fontsize=_TEXT["tick"]) + 8 * wpt)
    width = 0.97 - tag_w - _Tips.OFFSET * wpt - left
    lower = [fig.add_axes((left, bottom + (len(panels) - 1 - j) * (dd_h + gap), width, dd_h))
             for j in range(len(panels))]
    ax2 = lower[-1]                            # the bottom panel carries the date axis
    for a in lower[:-1]:
        a.sharex(ax2)
    ax = fig.add_axes((left, bottom + lower_h + 14 * pt, width, top - bottom - lower_h - 14 * pt),
                      sharex=ax2)
    tips = _Tips(ax, labels, cols, t["ink"], t["surface"])

    date_txt = ax.text(0.0, 1.0, "", transform=ax.transAxes, fontsize=_TEXT["date"],
                       color=t["ink_muted"], fontweight="bold", va="bottom")
    date_txt.set_position((0.0, 1.0 + _TEXT["date"] * 0.5 * pt / ax.get_position().height))
    cap_txt = fig.text(0.04, cap_y, caption, fontsize=_TEXT["caption"], color=t["ink"],
                       va="bottom", alpha=0.0)

    lines, dd_lines = [], []                   # dd_lines: per panel, one line per curve
    lws = _line_widths(labels, primary, widths)
    for lab, col, (lw, _) in zip(labels, cols, lws, strict=True):
        lines.append(ax.plot([], [], color=col, lw=lw, zorder=3 if lab == primary else 2)[0])
    for a in lower:
        dd_lines.append([a.plot([], [], color=col, lw=lw)[0]
                         for col, (_, lw) in zip(cols, lws, strict=True)])

    ax.axhline(1.0, color=t["ink_muted"], lw=0.8, ls=":")
    if log:
        _log_axis(ax)
        ax.set_ylim(lo / 1.15, hi * 1.15)
    else:
        pad = (hi - lo) * 0.06
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("portfolio growth")
    ax.tick_params(labelbottom=False, labelsize=_TEXT["tick"])
    for a, (kind, vals) in zip(lower, panels, strict=True):
        a.tick_params(labelsize=_TEXT["tick"], labelbottom=a is ax2)
        if kind == "drawdown":
            a.set_ylim(min(np.nanmin(vals.to_numpy()) * 1.08, -1), 2)
            a.set_ylabel("drawdown")
            _pct_axis(a)
        else:
            a.set_ylim(0, max(np.nanmax(vals.to_numpy()), 1e-9) * 1.08)
            a.set_ylabel("fees paid")
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _date_axis(ax2)
    if len(panels) > 1:                        # keep the stacked panels' labels in one column
        fig.align_ylabels([ax, *lower])
    ax2.xaxis.get_major_formatter().offset_formats = [""] * 6   # the date label says it
    fills = [None] * len(panels)

    draw_frames, hold_frames = int(seconds * fps), int(hold * fps)
    total = draw_frames + hold_frames
    g, ds = growth.to_numpy(), [vals.to_numpy() for _, vals in panels]
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
            for j, (a, d) in enumerate(zip(lower, ds, strict=True)):
                for i in range(len(labels)):
                    dd_lines[j][i].set_data(x, d[:k, i])
                if fills[j] is not None:
                    fills[j].remove()
                fills[j] = a.fill_between(x, np.nan_to_num(d[:k, p]), 0, color=cols[p],
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


def allocation_still(weights: dict[str, pd.DataFrame], fees: dict[str, pd.Series],
                     equity: dict[str, pd.Series], target: pd.Series, path: str | Path,
                     title: str, subtitle: str = "", caption: str = "",
                     threshold: float | None = None, colors: Sequence[str] | None = None,
                     seconds: float = 5.0, fps: int = 30, size: tuple[int, int] = (1080, 1080),
                     dpi: int = 200, dark: bool | None = None,
                     font: str | Path | None = None, engine: str = "auto") -> Path:
    """Render portfolios' full allocation histories as one frame held for ``seconds``.

    One panel per portfolio, in a 2-column grid: the held weights as a stacked
    area over the whole period, dashed lines at the target split, the total
    return, and two meters underneath. The drift meter shows the largest gap
    between any position and its target, at its worst over the period; the fee
    meter shows the fees paid in total. Encoded like ``equity_race`` (same
    ``size`` and ``fps``), so the two can be joined with ``concat``.

    Args:
        weights: ``{label: held weights}`` (e.g. ``backtest.held_weights``), one
            column per asset, all on the same datetime index.
        fees: ``{label: cumulative fees paid}`` in currency, on that index.
        equity: ``{label: equity}`` on that index.
        target: Target weight per asset (the ``weights`` columns), in order.
        path: Output file; ``.mp4`` (needs ffmpeg) or ``.gif``.
        title: Headline, top left.
        subtitle: One line of context under the title.
        caption: Takeaway along the bottom.
        threshold: Drift limit (as a weight, e.g. 0.05) marked on every drift meter.
        colors: One colour per asset. Defaults to the categorical series colours.
        seconds: How long the frame is held.
        fps: Frames per second.
        size: Output size in pixels, ``(width, height)``.
        dpi: Render resolution; with ``size`` this sets how large text appears.
        dark: Black background. None follows the active ``plots`` theme.
        font: A font file (``.ttf``/``.otf``) used for every piece of text.
        engine: Who encodes the held frame (it is always drawn once by
            matplotlib): ``"rust"``, ``"mpl"``, or ``"auto"`` for Rust when it
            is installed. Rust writes the frame once per video frame instead of
            redrawing it, and encodes like the other Rust renders.

    Returns:
        The path written.
    """
    dark = is_dark() if dark is None else dark
    use_rs = _use_rust(engine)
    if font is not None and not Path(font).is_file():
        raise FileNotFoundError(font)
    with _quiet_fonts(), plt.rc_context(_style(dark, font)):
        return _render_still(weights, fees, equity, target, Path(path), title, subtitle,
                             caption, threshold, colors, seconds, fps, size, dpi, dark, use_rs)


@contextmanager
def _quiet_fonts():
    """Silence font lookups while rendering; a one-weight font file warns on every bold."""
    fm_log = logging.getLogger("matplotlib.font_manager")
    level = fm_log.level
    fm_log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        fm_log.setLevel(level)


def _render_still(weights, fees, equity, target, path, title, subtitle, caption, threshold,
                  colors, seconds, fps, size, dpi, dark, use_rs=False) -> Path:
    t = _theme(dark)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(weights)
    assets = list(target.index)
    idx = weights[labels[0]].index
    cols = list(colors) if colors is not None else [color(i, dark) for i in range(len(assets))]
    W = {lab: weights[lab].reindex(columns=assets).to_numpy(float) * 100 for lab in labels}
    drift = {lab: np.abs(W[lab] - target.to_numpy(float) * 100).max(axis=1).max()
             for lab in labels}
    fee = {lab: float(fees[lab].iloc[-1]) for lab in labels}
    ret = {lab: (equity[lab].iloc[-1] / equity[lab].iloc[0] - 1) * 100 for lab in labels}
    drift_max = max(max(drift.values()), (threshold or 0) * 100) * 1.05
    fee_max = max(max(fee.values()), 1e-9) * 1.05

    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    pt = 1.0 / (size[1] / dpi * 72)            # one point as a fraction of the height
    wpt = 1.0 / (size[0] / dpi * 72)           # one point as a fraction of the width

    # Vertical layout in points, top down, as in equity_race.
    y = 1 - 16 * pt
    fig.text(0.04, y, title, fontsize=_TEXT["title"], fontweight="bold", color=t["ink"],
             va="top")
    y -= _TEXT["title"] * 1.4 * pt
    if subtitle:
        fig.text(0.04, y, subtitle, fontsize=_TEXT["subtitle"], color=t["ink_muted"], va="top")
        y -= _TEXT["subtitle"] * 1.5 * pt
    x = 0.04                                   # asset key: a swatch and name per asset
    y -= 4 * pt
    for a, c in zip(assets, cols, strict=True):
        fig.text(x, y, "■", fontsize=_TEXT["label"], color=c, va="top")
        x += 11 * wpt
        fig.text(x, y, a.upper(), fontsize=_TEXT["label"], color=t["ink"], va="top")
        x += _width(fig, a.upper(), fontsize=_TEXT["label"]) + 12 * wpt
    fig.text(0.96, y, f"{idx[0]:%b %Y} – {idx[-1]:%b %Y}", fontsize=_TEXT["label"],
             color=t["ink_muted"], va="top", ha="right")
    y -= _TEXT["date"] * 1.9 * pt
    cap_y = 10 * pt
    bottom = cap_y + _TEXT["caption"] * 2.2 * pt

    # Panels: header, stacked weights, then the two meters.
    ncol = 2
    nrow = int(np.ceil(len(labels) / ncol))
    gap_x, gap_y = 22 * wpt, 18 * pt
    pw = (0.92 - gap_x * (ncol - 1)) / ncol
    ph = (y - bottom - gap_y * (nrow - 1)) / nrow
    meter_fs = _TEXT["tick"] * 0.9
    bar_h = 5 * pt
    row_h = meter_fs * 1.25 * pt + 3 * pt + bar_h + 7 * pt   # label, gap, bar, gap
    head_h = _TEXT["label"] * 1.9 * pt
    stack_h = ph - head_h - 2 * row_h - 4 * pt
    cut = np.cumsum(target.to_numpy(float))[:-1] * 100

    for n, lab in enumerate(labels):
        r, c = divmod(n, ncol)
        px = 0.04 + c * (pw + gap_x)
        py = y - r * (ph + gap_y) - ph         # panel bottom
        fig.text(px, py + ph, lab, fontsize=_TEXT["label"], fontweight="bold", color=t["ink"],
                 va="top")
        fig.text(px + pw, py + ph, f"{ret[lab]:+.0f}%", fontsize=_TEXT["label"],
                 color=t["ink"], va="top", ha="right")
        ax = fig.add_axes((px, py + 2 * row_h + 4 * pt, pw, stack_h))
        ax.stackplot(idx, W[lab].T, colors=cols, lw=0)
        for v in cut:                          # the target split
            ax.axhline(v, color=t["surface"], lw=0.9, ls=(0, (3, 2)), zorder=3)
        ax.set_xlim(idx[0], idx[-1])
        ax.set_ylim(0, 100)
        ax.set_xticks([])                      # the period is in the header
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)

        for k, (name, v, scale, text) in enumerate((
                ("worst drift from target", drift[lab], drift_max, f"{drift[lab]:.1f}pp"),
                ("fees paid", fee[lab], fee_max, f"${fee[lab]:,.0f}"))):
            yb = py + (1 - k) * row_h + 7 * pt     # bar bottom; its label sits above it
            m = fig.add_axes((px, yb, pw, bar_h))
            m.set_xlim(0, 1)
            m.set_ylim(0, 1)
            m.axis("off")
            m.add_patch(plt.Rectangle((0, 0), 1, 1, color=t["grid"], lw=0))
            m.add_patch(plt.Rectangle((0, 0), min(v / scale, 1.0), 1, color=t["ink_muted"],
                                      lw=0))
            if k == 0 and threshold is not None:
                m.plot([threshold * 100 / scale] * 2, [-0.6, 1.6], color=t["ink"], lw=1.6,
                       clip_on=False, solid_capstyle="butt")
            ly = yb + bar_h + 3 * pt
            fig.text(px, ly, name, fontsize=meter_fs, color=t["ink_muted"], va="bottom")
            fig.text(px + pw, ly, text, fontsize=meter_fs, color=t["ink"], va="bottom",
                     ha="right")

    if caption:
        fig.text(0.04, cap_y, caption, fontsize=_TEXT["caption"], color=t["ink"], va="bottom")
    frames = max(1, int(seconds * fps))
    if use_rs:                                 # draw once; Rust repeats it into the encoder
        fig.set_facecolor(t["surface"])
        canvas = FigureCanvasAgg(fig)          # exact pixels: a HiDPI GUI backend doubles
        fig.set_dpi(dpi)                       # fig.dpi, which Agg would then render at
        canvas.draw()
        rgba = np.ascontiguousarray(np.asarray(canvas.buffer_rgba()))
        plt.close(fig)
        return Path(_rs.hold_frame(str(path), rgba, int(fps), frames))
    writer = _writer(path, fps, dpi)
    with writer.saving(fig, str(path), dpi):
        for _ in range(frames):
            writer.grab_frame(facecolor=t["surface"])
    plt.close(fig)
    return path


def risk_contributions(w: Sequence[float], cov: np.ndarray) -> np.ndarray:
    """Each asset's share of portfolio variance: ``w_i (cov @ w)_i / (w' cov w)``. Sums to 1."""
    w = np.asarray(w, float)
    m = np.asarray(cov, float) @ w
    return w * m / (w @ m)


def risk_bars(stages: dict[str, pd.Series], cov: pd.DataFrame, path: str | Path, title: str,
              subtitle: str = "", captions: Sequence[str] | None = None,
              colors: Sequence[str] | None = None, move: float = 1.2, hold: float = 3.0,
              fps: int = 30, size: tuple[int, int] = (1080, 1080), dpi: int = 200,
              dark: bool | None = None, font: str | Path | None = None,
              icon: str | Path | None = None, engine: str = "auto") -> Path:
    """Render capital weights beside risk contributions, morphing from one portfolio to the next.

    Two bar charts share the frame: how the money is split, and how the risk is
    split (each asset's share of portfolio variance under ``cov``). The first
    stage grows in from zero; each later stage morphs from the one before over
    ``move`` seconds, with risk recomputed from the in-between weights at every
    frame, so the bars move as the real portfolio would. Every stage is held for
    ``hold`` seconds with its caption. Encoded like ``equity_race`` (same
    ``size`` and ``fps``), so the two can be joined with ``concat``.

    Args:
        stages: ``{name: weights}`` in playing order; each Series is indexed by
            the ``cov`` assets and sums to 1.
        cov: Annualised covariance of asset returns (sets the risk bars and the
            portfolio volatility shown).
        path: Output file; ``.mp4`` (needs ffmpeg) or ``.gif``.
        title: Headline, top left.
        subtitle: One line of context under the title.
        captions: One takeaway per stage, shown while it is held.
        colors: One colour per asset. Defaults to the categorical series colours.
        move: Seconds for each grow-in or morph.
        hold: Seconds each stage is held.
        fps: Frames per second.
        size: Output size in pixels, ``(width, height)``.
        dpi: Render resolution; with ``size`` this sets how large text appears.
        dark: Black background. None follows the active ``plots`` theme.
        font: A font file (``.ttf``/``.otf``) used for every piece of text.
        icon: A PNG (or, with the Rust renderer, SVG) drawn at the left of the
            title block.
        engine: ``"rust"``, ``"mpl"``, or ``"auto"`` for Rust when it is installed.

    Returns:
        The path written.
    """
    dark = is_dark() if dark is None else dark
    use_rs = _use_rust(engine)
    for f in (font, icon):
        if f is not None and not Path(f).is_file():
            raise FileNotFoundError(f)
    if captions is not None and len(captions) != len(stages):
        raise ValueError("captions needs one entry per stage")
    if use_rs:
        t = _theme(dark)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assets = list(cov.index)
        W = np.array([stages[n].reindex(assets).to_numpy(float) for n in stages])
        cols = list(colors) if colors is not None else [color(i, dark) for i in range(len(assets))]
        regular, bold = _rs_fonts(font)
        return Path(_rs.risk_bars(
            str(path), [str(n) for n in stages], np.ascontiguousarray(W),
            np.ascontiguousarray(cov.loc[assets, assets].to_numpy(float)),
            [str(a) for a in assets], cols, title, subtitle,
            list(captions) if captions is not None else [""] * len(stages), float(move),
            float(hold), int(fps), int(size[0]), int(size[1]), float(dpi), t["ink"],
            t["ink_muted"], t["grid"], t["surface"], regular, bold,
            icon=None if icon is None else str(icon), progress=False))
    with _quiet_fonts(), plt.rc_context(_style(dark, font)):
        return _render_bars(stages, cov, Path(path), title, subtitle, captions, colors, move,
                            hold, fps, size, dpi, dark, icon)


def _render_bars(stages, cov, path, title, subtitle, captions, colors, move, hold, fps, size,
                 dpi, dark, icon) -> Path:
    t = _theme(dark)
    path.parent.mkdir(parents=True, exist_ok=True)
    assets = list(cov.index)
    names = list(stages)
    S = cov.loc[assets, assets].to_numpy(float)
    W = np.array([stages[n].reindex(assets).to_numpy(float) for n in names])
    cols = list(colors) if colors is not None else [color(i, dark) for i in range(len(assets))]

    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    pt = 1.0 / (size[1] / dpi * 72)            # one point as a fraction of the height
    wpt = 1.0 / (size[0] / dpi * 72)           # one point as a fraction of the width
    y = _title_block(fig, t, title, subtitle, icon, size)

    # The stage name, and the portfolio's volatility on the right.
    y -= 14 * pt
    stage_txt = fig.text(0.04, y, "", fontsize=_TEXT["title"] * 1.25, fontweight="bold",
                         color=t["ink"], va="top")
    vol_txt = fig.text(0.96, y - _TEXT["title"] * 0.25 * pt, "", fontsize=_TEXT["label"],
                       color=t["ink_muted"], va="top", ha="right")
    y -= _TEXT["title"] * 1.25 * 1.5 * pt + _TEXT["label"] * 2.2 * pt   # name, panel titles
    cap_y = 12 * pt
    bottom = cap_y + _TEXT["caption"] * 2.6 * pt + _TEXT["tick"] * 1.8 * pt
    gap = 64 * wpt                             # room for the equal-risk label
    pw = (0.92 - gap) / 2
    axes = [fig.add_axes((0.04 + k * (pw + gap), bottom, pw, y - bottom)) for k in range(2)]
    x = np.arange(len(assets))
    bars, tags = [], []
    for ax, head in zip(axes, ("capital", "risk"), strict=True):
        bars.append(ax.bar(x, np.zeros(len(assets)), width=0.72, color=cols, lw=0))
        tags.append([ax.text(i, 0, "", ha="center", va="bottom", fontsize=_TEXT["label"] * 1.3,
                             fontweight="bold", color=t["ink"]) for i in x])
        ax.set_ylim(0, 108)                   # headroom for a tag on a near-100% bar
        ax.set_xlim(-0.6, len(assets) - 0.4)
        ax.set_xticks(x, assets, fontsize=_TEXT["tick"], color=t["ink"])
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["ink_muted"])
        ax.text(0.0, 1.0 + _TEXT["label"] * 0.8 * pt / (y - bottom),
                "Capital: where the money is" if head == "capital"
                else "Risk: what moves the portfolio",
                transform=ax.transAxes, fontsize=_TEXT["label"], color=t["ink_muted"],
                va="bottom")
    eq = 100 / len(assets)                     # the equal-risk line
    axes[1].axhline(eq, color=t["ink_muted"], lw=1.1, ls=(0, (4, 3)), zorder=0)
    axes[1].text(-0.7, eq, "equal\nrisk", ha="right", va="center", linespacing=1.1,
                 fontsize=_TEXT["tick"] * 0.9, color=t["ink_muted"])   # in the gap
    cap_txt = fig.text(0.04, cap_y, "", fontsize=_TEXT["caption"], color=t["ink"], va="bottom",
                       wrap=True)

    def ease(u: float) -> float:
        return u * u * (3 - 2 * u)             # smoothstep: no jolt at either end

    move_f, hold_f = max(1, int(move * fps)), int(hold * fps)
    writer = _writer(path, fps, dpi)
    with writer.saving(fig, str(path), dpi):
        for s, name in enumerate(names):
            start = np.zeros(len(assets)) if s == 0 else W[s - 1]
            for f in range(move_f + hold_f):
                e = ease(min(1.0, (f + 1) / move_f))
                w = start + (W[s] - start) * e if s else W[s]
                grow = e if s == 0 else 1.0     # the first stage grows in from zero
                heights = (w * 100 * grow, risk_contributions(w, S) * 100 * grow)
                for b, tg, h in zip(bars, tags, heights, strict=True):
                    for rect, txt, v in zip(b, tg, h, strict=True):
                        rect.set_height(v)
                        txt.set_position((txt.get_position()[0], v + 1.5))
                        txt.set_text(f"{v:.0f}%" if grow > 0.95 or s else "")
                stage_txt.set_text(name)
                stage_txt.set_alpha(e if s == 0 else 1.0)
                vol_txt.set_text(f"portfolio volatility {np.sqrt(w @ S @ w) * 100:.0f}%/yr"
                                 if s or e > 0.95 else "")
                if captions:
                    cap_txt.set_text(captions[s])
                    cap_txt.set_alpha(min(1.0, max(0.0, (f - move_f + 1) / (0.4 * fps))))
                writer.grab_frame(facecolor=t["surface"])
    plt.close(fig)
    return path


def bell_morph(means: dict[str, pd.DataFrame], vols: dict[str, pd.DataFrame],
               weights: dict[str, pd.DataFrame], path: str | Path, title: str,
               subtitle: str = "", caption: str = "", colors: Sequence[str] | None = None,
               xlim: tuple[float, float] | None = None, seconds: float = 16.0,
               hold: float = 4.0, fps: int = 30, size: tuple[int, int] = (1080, 1080),
               dpi: int = 200, dark: bool | None = None, font: str | Path | None = None,
               icon: str | Path | None = None, progress: bool = True) -> Path:
    """Render each asset's estimated return distribution as a bell curve, re-estimated bar by bar.

    One panel per label (e.g. one per lookback window), each holding one normal
    curve per asset: centred on the estimated mean return, as wide as the
    estimated volatility, all drawn to the same height so a volatile asset stays
    as visible as a steady one. This is all a mean-variance optimiser sees of an asset,
    so a curve that slides around is an input the optimiser will chase. Each
    panel's header carries a bar of the weights chosen from those estimates.
    Axes are fixed from the whole history so the scale doesn't jump. Drawn by
    matplotlib and encoded like ``equity_race`` (same ``size`` and ``fps``), so
    the two can be joined with ``concat``.

    Args:
        means: ``{label: expected returns}``, annualised, as fractions; one
            column per asset, all on the same datetime index. NaN before the
            first estimate.
        vols: ``{label: volatilities}``, annualised, shaped like ``means``.
        weights: ``{label: weights}`` chosen from those estimates, shaped like
            ``means``.
        path: Output file; ``.mp4`` (needs ffmpeg) or ``.gif``.
        title: Headline, top left.
        subtitle: One line of context under the title.
        caption: Takeaway shown under the chart on the held final frame.
        colors: One colour per asset. Defaults to the categorical series colours.
        xlim: Return axis range, as fractions. Defaults to the 1st-99th
            percentile of every mean plus or minus two volatilities.
        seconds: Length of the draw-in, excluding ``hold``.
        hold: Seconds to hold the final frame.
        fps: Frames per second.
        size: Output size in pixels, ``(width, height)``.
        dpi: Render resolution; with ``size`` this sets how large text appears.
        dark: Black background. None follows the active ``plots`` theme.
        font: A font file (``.ttf``/``.otf``) used for every piece of text.
        icon: A PNG drawn at the left of the title block.
        progress: Print a frame counter to stderr while rendering.

    Returns:
        The path written.
    """
    dark = is_dark() if dark is None else dark
    for f in (font, icon):
        if f is not None and not Path(f).is_file():
            raise FileNotFoundError(f)
    if not (set(means) == set(vols) == set(weights)):
        raise ValueError("means, vols and weights need the same labels")
    with _quiet_fonts(), plt.rc_context(_style(dark, font)):
        return _render_bells(means, vols, weights, Path(path), title, subtitle, caption,
                             colors, xlim, seconds, hold, fps, size, dpi, dark, icon, progress)


def _render_bells(means, vols, weights, path, title, subtitle, caption, colors, xlim, seconds,
                  hold, fps, size, dpi, dark, icon, progress) -> Path:
    t = _theme(dark)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(means)
    idx = means[labels[0]].index
    assets = [str(a) for a in means[labels[0]].columns]
    cols = list(colors) if colors is not None else [color(i, dark) for i in range(len(assets))]
    M = {lab: means[lab].reindex(index=idx).to_numpy(float) * 100 for lab in labels}   # percent
    S = {lab: vols[lab].reindex(index=idx).to_numpy(float) * 100 for lab in labels}
    W = {lab: weights[lab].reindex(index=idx).to_numpy(float) for lab in labels}
    if xlim is None:
        lo = np.concatenate([(M[lab] - 2 * S[lab]).ravel() for lab in labels])
        hi = np.concatenate([(M[lab] + 2 * S[lab]).ravel() for lab in labels])
        xlim = (np.nanpercentile(lo, 1), np.nanpercentile(hi, 99))
    else:
        xlim = (xlim[0] * 100, xlim[1] * 100)
    xs = np.linspace(*xlim, 400)
    peak = 1.0                                 # every curve peaks at 1; its width is the vol

    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    pt = 1.0 / (size[1] / dpi * 72)            # one point as a fraction of the height
    wpt = 1.0 / (size[0] / dpi * 72)           # one point as a fraction of the width
    y = _title_block(fig, t, title, subtitle, icon, size)
    x = 0.04                                   # asset key: a swatch and name per asset
    y -= 4 * pt
    sw = _TEXT["label"] * 0.75                 # swatch side, in points
    for a, c in zip(assets, cols, strict=True):
        fig.add_artist(plt.Rectangle((x, y - _TEXT["label"] * 0.95 * pt), sw * wpt, sw * pt,
                                     color=c, lw=0, transform=fig.transFigure))
        x += sw * wpt + 5 * wpt
        fig.text(x, y, a, fontsize=_TEXT["label"], color=t["ink"], va="top")
        x += _width(fig, a, fontsize=_TEXT["label"]) + 14 * wpt
    date_txt = fig.text(0.96, y + 2 * pt, "", fontsize=_TEXT["date"], color=t["ink_muted"],
                        fontweight="bold", va="top", ha="right")
    top = y - _TEXT["label"] * 2.2 * pt
    cap_y = 10 * pt
    bottom = cap_y + (_TEXT["caption"] * 1.6 + _TEXT["tick"] * 1.6
                      + plt.rcParams["axes.labelsize"] * 2.2) * pt
    left, right = 0.04, 0.96
    inset = _width(fig, "-600%", fontsize=_TEXT["tick"]) / 2   # room for the end tick labels

    # Panels, top down; each has a header row: label on the left, weight bar on the right.
    head = _TEXT["label"] * 2.0 * pt
    gap = 12 * pt
    ph = (top - bottom - len(labels) * head - gap * (len(labels) - 1)) / len(labels)
    bar_w = 0.42 * (right - left)
    panels = []
    for j, lab in enumerate(labels):
        pb = top - (j + 1) * (head + ph) - j * gap         # panel bottom
        ax = fig.add_axes((left + inset, pb, right - left - 2 * inset, ph))
        ax.set_xlim(*xlim)
        ax.set_ylim(0, peak * 1.08)
        ax.set_yticks([])
        ax.axvline(0, color=t["ink_muted"], lw=0.8, ls=":")
        ax.grid(False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["ink_muted"])
        ax.tick_params(labelsize=_TEXT["tick"], labelbottom=j == len(labels) - 1)
        _pct_axis(ax, "x")
        hy = pb + ph + 4 * pt
        fig.text(left, hy, lab, fontsize=_TEXT["label"], fontweight="bold", color=t["ink"],
                 va="bottom")
        wb = fig.add_axes((right - bar_w, hy + 1 * pt, bar_w, _TEXT["label"] * 0.8 * pt))
        wb.set_xlim(0, 1)
        wb.set_ylim(0, 1)
        wb.axis("off")
        segs = [wb.add_patch(plt.Rectangle((0, 0), 0, 1, color=c, lw=0)) for c in cols]
        top_txt = fig.text(right - bar_w - 8 * wpt, hy, "", fontsize=_TEXT["label"],
                           color=t["ink_muted"], va="bottom", ha="right")
        curves = [ax.plot([], [], color=c, lw=1.8, zorder=3)[0] for c in cols]
        ticks = [ax.plot([], [], color=c, lw=2.2, solid_capstyle="butt", zorder=4)[0]
                 for c in cols]                # each mean, marked on the baseline
        panels.append((ax, curves, ticks, segs, top_txt, []))
    panels[-1][0].set_xlabel("annual return: mean ± volatility, estimated from the window")
    cap_txt = fig.text(0.04, cap_y, caption, fontsize=_TEXT["caption"], color=t["ink"],
                       va="bottom", alpha=0.0)

    first = max(int(np.argmax(np.isfinite(M[lab]).all(axis=1))) for lab in labels)
    n = len(idx) - first
    draw_frames, hold_frames = int(seconds * fps), int(hold * fps)
    total = draw_frames + hold_frames
    writer = _writer(path, fps, dpi)
    with writer.saving(fig, str(path), dpi):
        for f in range(total):
            i = first + min(n, int(np.ceil(n * (f + 1) / draw_frames))) - 1
            for lab, (ax, curves, ticks, segs, top_txt, fills) in zip(labels, panels,
                                                                     strict=True):
                for fl in fills:
                    fl.remove()
                fills.clear()
                for a in range(len(assets)):
                    m, s = M[lab][i, a], S[lab][i, a]
                    dens = np.exp(-0.5 * ((xs - m) / s) ** 2)   # a normal curve, peak 1
                    curves[a].set_data(xs, dens)
                    ticks[a].set_data([m, m], [0, peak * 0.06])
                    fills.append(ax.fill_between(xs, dens, 0, color=cols[a],
                                                 alpha=0.16 if dark else 0.12, lw=0, zorder=2))
                w = np.nan_to_num(W[lab][i])
                edges = np.concatenate([[0.0], np.cumsum(w)])
                for a, seg in enumerate(segs):
                    seg.set_x(edges[a])
                    seg.set_width(w[a])
                k = int(np.argmax(w))
                top_txt.set_text(f"holds {assets[k]} {w[k]:.0%}")
            date_txt.set_text(idx[i].strftime("%b %Y"))
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


_VOL_FORMULAS = (
    (r"$r_t = P_t \,/\, P_{t-1} - 1$", "daily return"),
    (r"$\hat\sigma_t^2 = \lambda\,\hat\sigma_{t-1}^2 + (1-\lambda)\,r_t^2$",
     "estimated volatility, annualised"),
    (r"$w_t = \min\left(\sigma_{\mathrm{target}} \,/\, \hat\sigma_t,\ 1\right)$",
     "share in the asset, rest in cash"),
)


def vol_explainer(returns: pd.Series, vol: pd.Series, weight: pd.Series, target: float,
                  path: str | Path, title: str, subtitle: str = "", caption: str = "",
                  formulas: Sequence[tuple[str, str]] = _VOL_FORMULAS,
                  colors: Sequence[str] | None = None, seconds: float = 3.5,
                  hold: float = 1.5, fps: int = 30, size: tuple[int, int] = (1080, 1080),
                  dpi: int = 200, dark: bool | None = None, font: str | Path | None = None,
                  icon: str | Path | None = None, progress: bool = True) -> Path:
    """Render how volatility targeting works: returns, the volatility estimate, the position.

    Three stacked panels on one date axis, drawn in left to right together:
    daily returns as up/down bars, the volatility estimate against the target
    (shaded where it runs above), and the share of equity held in the asset
    against cash. Each panel is headed by its formula, so the clip reads as the
    rule's derivation. Drawn by matplotlib and encoded like ``equity_race``
    (same ``size`` and ``fps``), so it can open a video joined with ``concat``.

    Args:
        returns: Daily returns, as fractions, on a datetime index.
        vol: Annualised volatility estimate, as a fraction, on the same index.
        weight: Share of equity held in the asset, 0 to 1, on the same index.
        target: Target volatility, as a fraction; drawn as a dashed line.
        path: Output file; ``.mp4`` (needs ffmpeg) or ``.gif``.
        title: Headline, top left.
        subtitle: One line of context under the title.
        caption: Takeaway shown under the chart on the held final frame.
        formulas: ``(formula, note)`` per panel, top down. ``formula`` is
            matplotlib mathtext (``$...$``); ``note`` is a short description
            drawn beside it, which may also use mathtext (e.g. for a Greek
            letter the ``font`` lacks), so escape a literal dollar as ``\\$``.
        colors: ``(up day, down day, strategy)``. Defaults to the green and
            orange series colours and the brand violet.
        seconds: Length of the draw-in, excluding ``hold``.
        hold: Seconds to hold the final frame.
        fps: Frames per second.
        size: Output size in pixels, ``(width, height)``.
        dpi: Render resolution; with ``size`` this sets how large text appears.
        dark: Black background. None follows the active ``plots`` theme.
        font: A font file (``.ttf``/``.otf``) used for every piece of text
            except the formulas, which use a matching sans maths font.
        icon: A PNG drawn at the left of the title block.
        progress: Print a frame counter to stderr while rendering.

    Returns:
        The path written.
    """
    dark = is_dark() if dark is None else dark
    for f in (font, icon):
        if f is not None and not Path(f).is_file():
            raise FileNotFoundError(f)
    if len(formulas) != 3:
        raise ValueError("formulas needs one (formula, note) pair per panel, three in all")
    if colors is None:
        colors = (color(2, dark), color(1, dark), ramp(3, dark, hue="purple")[-1])
    with _quiet_fonts(), plt.rc_context(_style(dark, font) | {"mathtext.fontset": "stixsans"}):
        return _render_vol(returns, vol, weight, target, Path(path), title, subtitle, caption,
                           formulas, colors, seconds, hold, fps, size, dpi, dark, icon,
                           progress)


def _render_vol(returns, vol, weight, target, path, title, subtitle, caption, formulas, colors,
                seconds, hold, fps, size, dpi, dark, icon, progress) -> Path:
    t = _theme(dark)
    path.parent.mkdir(parents=True, exist_ok=True)
    up, down, strat = colors
    idx = returns.index
    r = returns.to_numpy(float) * 100
    v = vol.reindex(idx).to_numpy(float) * 100
    w = weight.reindex(idx).to_numpy(float) * 100
    tgt = target * 100

    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    pt = 1.0 / (size[1] / dpi * 72)            # one point as a fraction of the height
    wpt = 1.0 / (size[0] / dpi * 72)           # one point as a fraction of the width
    top = _title_block(fig, t, title, subtitle, icon, size) - 6 * pt
    cap_y = 10 * pt
    bottom = cap_y + (_TEXT["caption"] * 1.6 + _TEXT["tick"] * 2.2) * pt
    left = (0.03 + plt.rcParams["axes.labelsize"] * 1.6 * wpt
            + _width(fig, "100%", fontsize=_TEXT["tick"]) + 8 * wpt)
    right = 0.96
    f_size = _TEXT["label"] * 1.3                # formulas lead each panel
    head = f_size * 2.3 * pt
    gap = 8 * pt
    ph = (top - bottom - 3 * head - 2 * gap) / 3

    axes = []
    for j, (formula, note) in enumerate(formulas):
        pb = top - (j + 1) * (head + ph) - j * gap
        ax = fig.add_axes((left, pb, right - left, ph), sharex=axes[0] if axes else None)
        hy = pb + ph + 6 * pt
        fig.text(left, hy, formula, fontsize=f_size, color=t["ink"], va="bottom",
                 parse_math=True)
        fig.text(right, hy, note, fontsize=_TEXT["label"], color=t["ink_muted"], va="bottom",
                 ha="right", parse_math=True)
        ax.tick_params(labelsize=_TEXT["tick"], labelbottom=j == 2)
        _pct_axis(ax)
        axes.append(ax)
    ax_r, ax_v, ax_w = axes
    ax_r.set_xlim(idx[0], idx[-1])
    _date_axis(ax_w)
    ax_w.xaxis.get_major_formatter().offset_formats = [""] * 6

    lim = np.nanmax(np.abs(r)) * 1.05
    ax_r.set_ylim(-lim, lim)
    ax_r.axhline(0, color=t["ink_muted"], lw=0.6)
    ax_r.set_ylabel("return")
    ax_v.set_ylim(0, max(np.nanmax(v), tgt) * 1.1)
    ax_v.axhline(tgt, color=t["ink"], lw=0.9, ls=(0, (4, 3)))
    ax_v.annotate(f"target {target:.0%}", (1.0, tgt), xycoords=("axes fraction", "data"),
                  xytext=(-4, 4), textcoords="offset points", ha="right", va="bottom",
                  fontsize=_TEXT["tick"], color=t["ink"])
    ax_v.set_ylabel("volatility")
    ax_w.set_ylim(0, 100)
    ax_w.set_ylabel("held")
    fig.align_ylabels(axes)

    # One vertical bar per day, as wide as a day on screen.
    ax_pt = (right - left) * size[0] / dpi * 72
    bars = ax_r.vlines([], [], [], lw=max(0.3, 0.85 * ax_pt / len(idx)),
                       capstyle="butt")
    bar_cols = np.where(r < 0, down, up)
    x_num = ax_r.convert_xunits(idx)
    v_line = ax_v.plot([], [], color=strat, lw=1.6, zorder=3)[0]
    w_line = ax_w.plot([], [], color=strat, lw=1.0, zorder=3)[0]
    cap_txt = fig.text(0.04, cap_y, caption, fontsize=_TEXT["caption"], color=t["ink"],
                       va="bottom", alpha=0.0)
    fills = []

    draw_frames, hold_frames = int(seconds * fps), int(hold * fps)
    total = draw_frames + hold_frames
    writer = _writer(path, fps, dpi)
    with writer.saving(fig, str(path), dpi):
        for f in range(total):
            k = min(len(idx), max(2, int(np.ceil(len(idx) * (f + 1) / draw_frames))))
            x = idx[:k]
            ok = np.isfinite(r[:k])
            bars.set_segments([[(xi, 0), (xi, ri)] for xi, ri in zip(x_num[:k][ok], r[:k][ok])])
            bars.set_color(bar_cols[:k][ok])
            v_line.set_data(x, v[:k])
            w_line.set_data(x, w[:k])
            for fl in fills:
                fl.remove()
            vk, wk = np.nan_to_num(v[:k]), np.nan_to_num(w[:k])
            fills[:] = [
                ax_v.fill_between(x, tgt, vk, where=vk > tgt, color=strat, lw=0,
                                  alpha=0.3 if dark else 0.25, interpolate=True),
                ax_w.fill_between(x, 0, wk, color=strat, lw=0, alpha=0.85),
                ax_w.fill_between(x, wk, 100, color=t["grid"], lw=0),
            ]
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
