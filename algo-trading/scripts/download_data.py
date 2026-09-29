"""Download a price dataset from Google Drive into ``.data/``.

The ``backtest`` package reads the dataset chosen in ``config/config.yaml``. Run
from the ``algo-trading`` directory::

    python scripts/download_data.py                        # historical_data.csv
    python scripts/download_data.py --dataset ohlcv_15m    # ohlcv_15m.csv
    python scripts/download_data.py --dataset all          # both
    python scripts/download_data.py --force                # re-download and overwrite

The dataset names match the entries under ``datasets`` in ``config/config.yaml``:

* ``historical``: ``historical_data.csv`` (~8 MB), long format, one row per
  ``(interval, ts, token_id)`` with columns
  ``interval, ts, token_id, open, high, low, close, volume, market_cap``.
* ``ohlcv_15m``: ``ohlcv_15m.csv`` (~290 MB), 15-minute bars, one row per
  ``(dt, symbol)`` with columns
  ``<row index>, dt, timestamp, symbol, open, high, low, close, volume, trades``.

See ``src/algo_trading/backtest/data.py`` for what each column means.

Standard library only, so it runs before ``pip install -e .``. Colour is turned
off when output is not a terminal or ``NO_COLOR`` is set.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / ".data"
CHUNK = 64 * 1024


@dataclass(frozen=True)
class Dataset:
    """One downloadable file: its Drive id, local name, and expected CSV header."""
    file_id: str
    filename: str
    header: str

    @property
    def share_url(self) -> str:
        return f"https://drive.google.com/file/d/{self.file_id}/view?usp=sharing"

    @property
    def download_url(self) -> str:
        return (f"https://drive.usercontent.google.com/download"
                f"?id={self.file_id}&export=download&confirm=t")


DATASETS = {
    "historical": Dataset("1kAzenUMGXyuP7rnKpqisIflU3nscz9O6", "historical_data.csv",
                          "interval,ts,token_id,open,high,low,close,volume,market_cap"),
    "ohlcv_15m": Dataset("1Axd2JvTwBAZ2rUX-vLHRWLmhoeQ7HUX0", "ohlcv_15m.csv",
                         ",dt,timestamp,symbol,open,high,low,close,volume,trades"),
}


# ── Terminal styling ─────────────────────────────────────────────────────────

class Style:
    """ANSI styles that collapse to plain text when colour isn't wanted."""

    def __init__(self, stream=sys.stdout):
        self.tty = stream.isatty()
        self.on = self.tty and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def blue(self, t: str) -> str:
        return self._wrap("38;5;33", t)

    def green(self, t: str) -> str:
        return self._wrap("38;5;35", t)

    def yellow(self, t: str) -> str:
        return self._wrap("38;5;178", t)

    def red(self, t: str) -> str:
        return self._wrap("38;5;203", t)


S = Style()


def _mb(n: float) -> str:
    return f"{n / 1e6:.1f} MB"


def _eta(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


class ProgressBar:
    """A single-line download bar: ``━━━━━━╺──── 62%  4.7/7.6 MB  3.1 MB/s  0:01``.

    With an unknown total it shows a sliding segment and the bytes received.
    Redraws are throttled, and nothing is drawn when output is not a terminal.

    Args:
        total: Expected size in bytes, or None if the server didn't say.
        width: Bar width in characters.
    """

    def __init__(self, total: int | None, width: int = 32):
        self.total = total if total and total > 0 else None
        self.width = width
        self.done = 0
        self.start = time.monotonic()
        self._last_draw = 0.0

    def update(self, n: int) -> None:
        """Record ``n`` more bytes and redraw if enough time has passed."""
        self.done += n
        now = time.monotonic()
        if now - self._last_draw >= 0.05:
            self._last_draw = now
            self._draw()

    def close(self) -> None:
        """Draw the final state and clear the line."""
        if S.tty:
            self._draw(final=True)
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()

    def _bar(self, final: bool) -> str:
        w = self.width
        if self.total:
            frac = 1.0 if final else min(self.done / self.total, 1.0)
            filled = int(frac * w)
            head = "" if filled >= w else "╺"
            rest = "─" * (w - filled - len(head))
            return S.blue("━" * filled) + S.dim(head + rest)
        seg, pos = 8, int((time.monotonic() - self.start) * 20) % (w + 8) - 8
        cells = ["━" if pos <= i < pos + seg else "─" for i in range(w)]
        return "".join(S.blue(c) if c == "━" else S.dim(c) for c in cells)

    def _draw(self, final: bool = False) -> None:
        if not S.tty:
            return
        elapsed = max(time.monotonic() - self.start, 1e-6)
        rate = self.done / elapsed
        parts = [f"  {self._bar(final)}"]
        if self.total:
            pct = 100 if final else min(100, int(self.done / self.total * 100))
            parts.append(S.bold(f"{pct:>3d}%"))
            parts.append(f"{self.done / 1e6:.1f}/{_mb(self.total)}")
        else:
            parts.append(_mb(self.done))
        parts.append(S.dim(f"{rate / 1e6:.1f} MB/s"))
        if self.total and rate > 0 and not final:
            parts.append(S.dim(f"eta {_eta((self.total - self.done) / rate)}"))
        line = "  ".join(parts)
        sys.stdout.write("\r\033[2K" + line)
        sys.stdout.flush()


# ── Download ─────────────────────────────────────────────────────────────────

def download(ds: Dataset, out: Path, timeout: int = 120) -> Path:
    """Stream the CSV to ``out`` with a progress bar, writing to a temp file first.

    Args:
        ds: The dataset to fetch.
        out: Destination path.
        timeout: Socket timeout in seconds.

    Returns:
        The path written.

    Raises:
        RuntimeError: If the response is not the expected CSV (for example, a
            Google Drive HTML page because the link is private or rate-limited).
        urllib.error.URLError: If the request fails.
    """
    out.parent.mkdir(parents=True, exist_ok=True)  # create .data/ on a fresh clone
    tmp = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(ds.download_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
            length = resp.headers.get("Content-Length")
            bar = ProgressBar(int(length) if length and length.isdigit() else None)
            try:
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
                    bar.update(len(chunk))
            finally:
                bar.close()
        with tmp.open(encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        if first != ds.header:
            raise RuntimeError(
                f"download did not return the expected CSV (got {first[:80]!r}).\n"
                f"Download it manually from {ds.share_url} and save it as {out}")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _rel(path: Path) -> str:
    """Show ``path`` relative to the cwd when that's shorter."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def fetch(ds: Dataset, out: Path, force: bool) -> int:
    """Download one dataset unless it is already there; return an exit code."""
    if out.exists() and not force:
        print(f"{S.yellow('•')} {S.bold(_rel(out))} already exists "
              f"{S.dim('(' + _mb(out.stat().st_size) + ')')} — pass "
              f"{S.bold('--force')} to re-download")
        return 0

    print(f"{S.blue('↓')} {S.bold(ds.filename)}  {S.dim('from Google Drive')}")
    start = time.monotonic()
    try:
        path = download(ds, out)
    except (RuntimeError, urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        print(f"{S.red('✗')} {S.bold('Download failed:')} {reason}", file=sys.stderr)
        if not isinstance(exc, RuntimeError):
            print(S.dim(f"  Try again, or download manually from {ds.share_url}"),
                  file=sys.stderr)
        return 1

    took = time.monotonic() - start
    print(f"{S.green('✓')} Saved {S.bold(_mb(path.stat().st_size))} to "
          f"{S.bold(_rel(path))} {S.dim(f'in {took:.1f}s')}")
    return 0


def main() -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="historical",
                    help="which file to fetch; names match config/config.yaml")
    ap.add_argument("--out", type=Path,
                    help="destination file (default: .data/<filename>); single dataset only")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    if args.out is not None and len(names) > 1:
        ap.error("--out needs a single --dataset")
    codes = [fetch(DATASETS[n], args.out or DATA_DIR / DATASETS[n].filename, args.force)
             for n in names]
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())
