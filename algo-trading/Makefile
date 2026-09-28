# Build the optional Rust engines (rust/) into the project venv.
#
#   make rust        build and install both extensions (release)
#   make test        cargo tests + Python/Rust parity check
#   make help        list every target

VENV    ?= .venv
PYTHON  := $(VENV)/bin/python
MATURIN := $(VENV)/bin/maturin
CARGO   ?= cargo
PROFILE ?= --release

export VIRTUAL_ENV := $(abspath $(VENV))

.DEFAULT_GOAL := help
.PHONY: help rust rust-backtest rust-animate test rust-test parity lint rust-lint fmt clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

$(PYTHON):
	@echo "no venv at $(VENV); create it first (see INFO.md)" && exit 1

$(MATURIN): | $(PYTHON)
	$(PYTHON) -m pip install -q "maturin>=1.7,<2"

rust: rust-backtest rust-animate  ## Build and install both Rust extensions

rust-backtest: | $(MATURIN)  ## Build and install the simulator (algo_backtest_rs)
	$(MATURIN) develop $(PROFILE) -m rust/backtest/Cargo.toml

rust-animate: | $(MATURIN)  ## Build and install the video renderer (algo_animate_rs)
	$(MATURIN) develop $(PROFILE) -m rust/animate/Cargo.toml

test: rust-test parity  ## Run the Rust tests and the Python/Rust parity check

rust-test:  ## cargo test for both crates
	cd rust && $(CARGO) test --workspace

parity: | $(PYTHON)  ## Check the Rust simulator matches the Python reference
	$(PYTHON) -c "import numpy as np; \
	from algo_trading.backtest import backtest as bt, data; \
	from algo_trading.backtest.config import PortfolioConfig; \
	c = data.daily_panel().close; w = c.notna().astype(float); w = w.div(w.sum(axis=1), axis=0); \
	cfg = PortfolioConfig(check_bars=1); \
	a, b = (bt.simulate(c, w, cfg, 20, 1e4, 50, engine=e) for e in ('python', 'rust')); \
	assert len(a['fills']) == len(b['fills']); \
	assert np.allclose(a['equity'], b['equity'], rtol=1e-12); print('parity OK')"

lint: rust-lint  ## Lint Rust (clippy) and Python (ruff)
	$(VENV)/bin/ruff check src scripts

rust-lint:  ## cargo clippy for both crates, warnings as errors
	cd rust && $(CARGO) clippy --workspace --all-targets -- -D warnings

fmt:  ## Format the Rust code
	cd rust && $(CARGO) fmt --all

clean:  ## Remove Rust build output
	cd rust && $(CARGO) clean
