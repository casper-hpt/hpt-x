# Rust engines

Two optional Python extensions, built with [maturin](https://www.maturin.rs/). The
Python package works without them and uses them automatically when installed.

| crate | Python module | speeds up |
|---|---|---|
| `backtest/` | `algo_backtest_rs` | `backtest.simulate` (~50x) and `backtest.simulate_many` (parallel, ~90x) |
| `animate/` | `algo_animate_rs` | `animate.equity_race` (all cores, ~15x) |

## Build

Needs a Rust toolchain ([rustup](https://rustup.rs/)). From `algo-trading/`:

```bash
make rust     # installs maturin into .venv if needed, builds both extensions
make test     # cargo tests + a Python/Rust parity check
make help     # every target
```

Rebuild after changing Rust code. Video rendering also needs `ffmpeg` on the PATH.

## Choosing an engine

Both entry points take `engine=`: `"auto"` (default) uses Rust when installed,
`"python"` / `"mpl"` forces the reference implementation. The Python versions stay
the reference: `rust/backtest/src/sim.rs` and `rust/animate/src/race.rs` are ports,
so change both together.

## Tests

```bash
make test
```
