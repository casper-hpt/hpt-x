//! The simulator itself, free of any Python types so it can be tested with
//! `cargo test` and run on worker threads.
//!
//! This is a line-for-line port of the Python reference in
//! `algo_trading/backtest/backtest.py::_simulate_py`; keep the two in step.

/// Inputs that don't vary per bar.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Cost per fill as a fraction of notional (bps / 1e4).
    pub fee: f64,
    /// Starting cash.
    pub initial: f64,
    /// Bars to skip before the first trade.
    pub warmup: usize,
    /// Inspect drift every this many bars.
    pub check_bars: usize,
    /// No-trade band, relative to the target weight.
    pub band: f64,
    /// Orders smaller than this share of equity are skipped.
    pub min_trade_frac: f64,
    /// If false, held positions are never resized: bought once, sold in full.
    pub rebalance: bool,
}

/// One executed order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fill {
    pub t: usize,
    pub token: usize,
    pub buy: bool,
    pub notional: f64,
    pub fee: f64,
}

/// Book state after one rebalance check.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogRow {
    pub t: usize,
    pub n_held: usize,
    pub gross: f64,
    pub cash_pct: f64,
}

/// Everything one run produces.
#[derive(Clone, Debug, Default)]
pub struct SimOutput {
    pub equity: Vec<f64>,
    pub fills: Vec<Fill>,
    pub log: Vec<LogRow>,
    pub fees: f64,
    pub traded: f64,
}

/// Row-major 2-D view: `data[t * n + i]` is bar `t`, token `i`.
#[derive(Clone, Copy)]
pub struct Grid<'a> {
    pub data: &'a [f64],
    pub n_bars: usize,
    pub n_tokens: usize,
}

impl Grid<'_> {
    #[inline]
    fn row(&self, t: usize) -> &[f64] {
        &self.data[t * self.n_tokens..(t + 1) * self.n_tokens]
    }
}

/// `np.nan_to_num`: NaN -> 0, +/-inf -> +/-f64::MAX.
#[inline]
fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v.is_infinite() {
        if v > 0.0 { f64::MAX } else { f64::MIN }
    } else {
        v
    }
}

/// Mark-to-market value of the book at the last known prices.
#[inline]
fn book_value(cash: f64, qty: &[f64], last: &[f64]) -> f64 {
    cash + qty.iter().zip(last).map(|(q, p)| q * nan_to_num(*p)).sum::<f64>()
}

/// Band-rebalanced, long-only simulation with per-fill fees.
///
/// Bar `t` trades on bar `t - 1`'s target at bar `t`'s price (no look-ahead).
/// On each check, sells are placed before buys so rotations fund themselves;
/// buys are capped at available cash and fees come out of each ticket.
pub fn simulate(px: Grid, w: Grid, p: &Params) -> SimOutput {
    let (n_bars, n) = (px.n_bars, px.n_tokens);
    let mut qty = vec![0.0; n];
    let mut last = vec![f64::NAN; n];
    let mut cash = p.initial;
    let mut out = SimOutput { equity: Vec::with_capacity(n_bars), ..Default::default() };
    let first = p.warmup.max(1);
    let check = p.check_bars.max(1);

    let mut tgt = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut tradeable = vec![false; n];
    let mut order: Vec<usize> = (0..n).collect();

    for t in 0..n_bars {
        let price = px.row(t);
        for i in 0..n {
            if !price[i].is_nan() {
                last[i] = price[i];
            }
        }
        let mut equity = book_value(cash, &qty, &last);

        if t >= first && (t - first) % check == 0 && equity > 0.0 {
            let target = w.row(t - 1);
            for i in 0..n {
                tgt[i] = nan_to_num(target[i]);
                tradeable[i] = !price[i].is_nan() && price[i] > 0.0;
                let w_now = if tradeable[i] { qty[i] * nan_to_num(price[i]) / equity } else { 0.0 };
                d[i] = tgt[i] - w_now;
            }
            // np.argsort(d, kind="stable"): most negative (sells) first, ties by column.
            order.iter_mut().enumerate().for_each(|(k, o)| *o = k);
            order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));

            for &i in &order {
                if !tradeable[i] {
                    continue;
                }
                let exiting = tgt[i] <= 1e-9 && qty[i] > 0.0;
                if !p.rebalance && qty[i] > 0.0 && !exiting {
                    continue; // buy-and-hold: never resize
                }
                let band_i = p.band * tgt[i].max(0.01);
                if !exiting && (d[i].abs() < band_i || d[i].abs() < p.min_trade_frac) {
                    continue; // inside the band: leave it alone
                }
                let mut notional = d[i] * equity;
                let (c, buy);
                if notional > 0.0 {
                    notional = notional.min(cash);
                    if notional <= 1e-8 {
                        continue;
                    }
                    c = notional * p.fee;
                    qty[i] += (notional - c) / price[i]; // fee comes out of the ticket
                    cash -= notional;
                    buy = true;
                } else {
                    let sell_qty = if exiting { qty[i] } else { qty[i].min(-notional / price[i]) };
                    if sell_qty <= 1e-12 {
                        continue;
                    }
                    let proceeds = sell_qty * price[i];
                    c = proceeds * p.fee;
                    qty[i] -= sell_qty;
                    cash += proceeds - c;
                    notional = -proceeds;
                    buy = false;
                }
                out.fees += c;
                out.traded += notional.abs();
                out.fills.push(Fill { t, token: i, buy, notional: notional.abs(), fee: c });
            }
            equity = book_value(cash, &qty, &last);
            let held: f64 = qty.iter().zip(price).map(|(q, p)| q * nan_to_num(*p)).sum();
            out.log.push(LogRow {
                t,
                n_held: qty.iter().filter(|&&q| q > 0.0).count(),
                gross: held / equity.max(1e-9),
                cash_pct: cash / equity.max(1e-9),
            });
        }
        out.equity.push(equity);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(data: &[f64], n_tokens: usize) -> Grid<'_> {
        Grid { data, n_bars: data.len() / n_tokens, n_tokens }
    }

    fn params() -> Params {
        Params { fee: 0.001, initial: 1000.0, warmup: 1, check_bars: 1, band: 0.25,
                 min_trade_frac: 0.002, rebalance: true }
    }

    #[test]
    fn flat_target_never_trades() {
        let px = [1.0, 2.0, 1.1, 2.1, 1.2, 2.2];
        let w = [0.0; 6];
        let out = simulate(grid(&px, 2), grid(&w, 2), &params());
        assert!(out.fills.is_empty());
        assert_eq!(out.equity, vec![1000.0; 3]);
    }

    #[test]
    fn buy_then_exit_pays_fee_both_ways() {
        // One token: long on bar 1 (target from bar 0), flat on bar 2.
        let px = [10.0, 10.0, 10.0];
        let w = [1.0, 0.0, 0.0];
        let out = simulate(grid(&px, 1), grid(&w, 1), &params());
        assert_eq!(out.fills.len(), 2);
        assert!(out.fills[0].buy && !out.fills[1].buy);
        let expected = 1000.0 * (1.0 - 0.001) * (1.0 - 0.001);
        assert!((out.equity[2] - expected).abs() < 1e-9);
        assert_eq!(out.log.last().unwrap().n_held, 0);
    }

    #[test]
    fn no_rebalance_leaves_drifted_positions_alone() {
        // Token 0 doubles while both targets stay at 0.5: rebalancing would trim it.
        let px = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        let w = [0.5; 8];
        let hold = Params { rebalance: false, ..params() };
        let out = simulate(grid(&px, 2), grid(&w, 2), &hold);
        assert_eq!(out.fills.len(), 2); // the two entries, nothing after
        let out = simulate(grid(&px, 2), grid(&w, 2), &params());
        assert!(out.fills.len() > 2);
    }

    #[test]
    fn sells_fund_buys_on_the_same_bar() {
        // Rotate fully from token 1 into token 0 on bar 2: without sells-first,
        // the buy of token 0 would be capped at zero cash.
        let px = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let out = simulate(grid(&px, 2), grid(&w, 2), &params());
        let bar2: Vec<_> = out.fills.iter().filter(|f| f.t == 2).collect();
        assert_eq!(bar2.len(), 2);
        assert!(!bar2[0].buy && bar2[1].buy);
        assert!(bar2[1].notional > 990.0);
    }
}
