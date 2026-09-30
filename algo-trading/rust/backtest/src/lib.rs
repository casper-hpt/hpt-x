//! Python bindings for the portfolio simulator.
//!
//! `algo_trading.backtest.simulate` calls into this module when it is installed
//! and falls back to its pure-Python loop otherwise. Results come back as flat
//! NumPy arrays; the Python side turns them into the usual DataFrames.

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

pub mod sim;
use sim::{Grid, Params, SimOutput};

/// A panel copied out of NumPy in row-major order, so it can cross threads.
struct Owned {
    data: Vec<f64>,
    n_bars: usize,
    n_tokens: usize,
}

impl Owned {
    fn from(a: &PyReadonlyArray2<f64>) -> Self {
        let v = a.as_array();
        let (n_bars, n_tokens) = v.dim();
        Owned { data: v.iter().copied().collect(), n_bars, n_tokens }
    }

    fn grid(&self) -> Grid<'_> {
        Grid { data: &self.data, n_bars: self.n_bars, n_tokens: self.n_tokens }
    }
}

#[allow(clippy::too_many_arguments)]
fn params(fee_bps: f64, initial: f64, warmup: usize, check_bars: usize, band: f64,
          min_trade_frac: f64, rebalance: bool, band_all: bool) -> Params {
    Params { fee: fee_bps / 1e4, initial, warmup, check_bars, band, min_trade_frac, rebalance,
             band_all }
}

fn check_shape(close: &Owned, w: &Owned) -> PyResult<()> {
    if (close.n_bars, close.n_tokens) != (w.n_bars, w.n_tokens) {
        return Err(PyValueError::new_err(format!(
            "weights shape {:?} does not match close shape {:?}",
            (w.n_bars, w.n_tokens), (close.n_bars, close.n_tokens))));
    }
    Ok(())
}

fn to_dict(py: Python<'_>, out: SimOutput) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    let fills = &out.fills;
    let log = &out.log;
    d.set_item("fill_t", fills.iter().map(|f| f.t as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("fill_token",
               fills.iter().map(|f| f.token as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("fill_buy", fills.iter().map(|f| f.buy).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("fill_notional",
               fills.iter().map(|f| f.notional).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("fill_fee", fills.iter().map(|f| f.fee).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("log_t", log.iter().map(|r| r.t as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("log_n_held",
               log.iter().map(|r| r.n_held as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("log_gross", log.iter().map(|r| r.gross).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("log_cash_pct",
               log.iter().map(|r| r.cash_pct).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("fees", out.fees)?;
    d.set_item("traded", out.traded)?;
    d.set_item("equity", out.equity.into_pyarray(py))?;
    Ok(d)
}

/// Simulate one target-weight panel. Runs with the GIL released.
///
/// Args:
///     close: (bars, tokens) float64 close prices; NaN where a token has no print.
///     weights: (bars, tokens) float64 target weights, same shape.
///     fee_bps: Cost per fill in basis points.
///     initial: Starting cash.
///     warmup: Bars to skip before the first trade.
///     check_bars: Inspect drift every this many bars.
///     band: No-trade band, relative to the target weight.
///     min_trade_frac: Orders smaller than this share of equity are skipped.
///     rebalance: If False, held positions are never resized (bought once, sold in full).
///     band_all: If True, one position outside its band trades every position to target.
///
/// Returns:
///     A dict of NumPy arrays: ``equity``, ``fill_*`` and ``log_*`` columns, plus
///     ``fees`` and ``traded`` totals.
#[pyfunction]
#[pyo3(signature = (close, weights, fee_bps, initial, warmup, check_bars, band, min_trade_frac,
                    rebalance=true, band_all=false))]
#[allow(clippy::too_many_arguments)]
fn simulate<'py>(py: Python<'py>, close: PyReadonlyArray2<'py, f64>,
                 weights: PyReadonlyArray2<'py, f64>, fee_bps: f64, initial: f64,
                 warmup: usize, check_bars: usize, band: f64, min_trade_frac: f64,
                 rebalance: bool, band_all: bool) -> PyResult<Bound<'py, PyDict>> {
    let (px, w) = (Owned::from(&close), Owned::from(&weights));
    check_shape(&px, &w)?;
    let p = params(fee_bps, initial, warmup, check_bars, band, min_trade_frac, rebalance,
                   band_all);
    let out = py.detach(|| sim::simulate(px.grid(), w.grid(), &p));
    to_dict(py, out)
}

/// Simulate many weight panels against one price panel, in parallel.
///
/// Same arguments as ``simulate`` except ``weights`` is a list of panels and
/// ``fee_bps`` is one value per panel. Returns one dict per panel, in order.
#[pyfunction]
#[pyo3(signature = (close, weights, fee_bps, initial, warmup, check_bars, band, min_trade_frac,
                    rebalance=true, band_all=false))]
#[allow(clippy::too_many_arguments)]
fn simulate_many<'py>(py: Python<'py>, close: PyReadonlyArray2<'py, f64>,
                      weights: Vec<PyReadonlyArray2<'py, f64>>, fee_bps: Vec<f64>,
                      initial: f64, warmup: usize, check_bars: usize, band: f64,
                      min_trade_frac: f64, rebalance: bool, band_all: bool)
                      -> PyResult<Vec<Bound<'py, PyDict>>> {
    if fee_bps.len() != weights.len() {
        return Err(PyValueError::new_err("fee_bps needs one value per weights panel"));
    }
    let px = Owned::from(&close);
    let ws: Vec<Owned> = weights.iter().map(Owned::from).collect();
    for w in &ws {
        check_shape(&px, w)?;
    }
    let outs: Vec<SimOutput> = py.detach(|| {
        ws.par_iter()
            .zip(fee_bps.par_iter())
            .map(|(w, &fee)| {
                let p = params(fee, initial, warmup, check_bars, band, min_trade_frac, rebalance,
                               band_all);
                sim::simulate(px.grid(), w.grid(), &p)
            })
            .collect()
    });
    outs.into_iter().map(|o| to_dict(py, o)).collect()
}

#[pymodule]
fn algo_backtest_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_many, m)?)?;
    Ok(())
}
