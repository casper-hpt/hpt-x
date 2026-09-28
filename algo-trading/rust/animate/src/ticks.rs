//! Tick placement and labels, following matplotlib's locators so the video
//! reads like the notebook charts.
//!
//! * `log_ticks`: decades plus 2..9 minors, labelled at mantissa 1, 2 and 5
//!   (the `plots._log_axis` formatter).
//! * `nice_ticks`: `MaxNLocator` with steps 1, 2, 2.5, 5, 10.
//! * `date_ticks`: `AutoDateLocator(minticks=4, maxticks=9)` with
//!   `ConciseDateFormatter` labels and no offset text.

/// A tick position and its label (empty when unlabelled).
pub type Tick = (f64, String);

/// Python's `f"{v:g}"` for the magnitudes an axis shows.
pub fn fmt_g(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if !(1e-4..1e6).contains(&a) {
        let s = format!("{v:e}"); // e.g. 1e6 -> "1e6"
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let e: i32 = e.parse().unwrap_or(0);
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let digits = (5 - a.log10().floor() as i32).max(0) as usize; // 6 significant figures
    let s = format!("{v:.digits$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// Log-scale ticks inside `[lo, hi]`: `(majors, minors)`, each labelled at 1/2/5 x 10^k.
pub fn log_ticks(lo: f64, hi: f64) -> (Vec<Tick>, Vec<Tick>) {
    let (mut majors, mut minors) = (vec![], vec![]);
    let (k0, k1) = (lo.log10().floor() as i32, hi.log10().ceil() as i32);
    for k in k0..=k1 {
        let base = 10f64.powi(k);
        for m in 1..=9 {
            let v = base * m as f64;
            if v < lo * (1.0 - 1e-9) || v > hi * (1.0 + 1e-9) {
                continue;
            }
            let label = if matches!(m, 1 | 2 | 5) { fmt_g(v) } else { String::new() };
            if m == 1 { majors.push((v, label)) } else { minors.push((v, label)) }
        }
    }
    (majors, minors)
}

/// `MaxNLocator(nbins)` ticks inside `[lo, hi]`.
pub fn nice_ticks(lo: f64, hi: f64, nbins: usize) -> Vec<f64> {
    let nbins = nbins.clamp(1, 9) as f64;
    let raw = (hi - lo) / nbins;
    if raw <= 0.0 || !raw.is_finite() {
        return vec![lo];
    }
    let scale = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0]
        .iter()
        .map(|s| s * scale)
        .find(|s| *s >= raw * (1.0 - 1e-9))
        .unwrap_or(10.0 * scale);
    let first = (lo / step).ceil() as i64;
    let last = (hi / step).floor() as i64;
    (first..=last).map(|i| i as f64 * step).map(|v| if v.abs() < step * 1e-9 { 0.0 } else { v })
        .collect()
}

// ── Dates ────────────────────────────────────────────────────────────────────

const MONTHS: [&str; 12] =
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/// Days since 1970-01-01 -> (year, month 1..=12, day 1..=31). Howard Hinnant's algorithm.
pub fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// (year, month, day) -> days since 1970-01-01.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// "Sep 2025" for a timestamp in days since the epoch.
pub fn month_year(t: f64) -> String {
    let (y, m, _) = civil(t.floor() as i64);
    format!("{} {y}", MONTHS[m as usize - 1])
}

fn pick(num: i64, intervals: &[i64]) -> i64 {
    const MAXTICKS: i64 = 9;
    *intervals.iter().find(|&&iv| num <= iv * (MAXTICKS - 1)).unwrap_or(intervals.last().unwrap())
}

/// Date ticks over `[t0, t1]` (days since the epoch, fractional for intraday).
pub fn date_ticks(t0: f64, t1: f64) -> Vec<Tick> {
    const MINTICKS: i64 = 4;
    let inside = |t: f64| t >= t0 - 1e-9 && t <= t1 + 1e-9;
    let (y0, m0, d0) = civil(t0.floor() as i64);
    let (y1, m1, d1) = civil(t1.floor() as i64);
    let months = (y1 - y0) * 12 + (m1 as i64 - m0 as i64) - i64::from(d1 < d0);
    let days = (t1 - t0).floor() as i64;
    let hours = ((t1 - t0) * 24.0).floor() as i64;
    let mut ticks = vec![];

    if months / 12 >= MINTICKS {
        let iv = pick(months / 12, &[1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500, 1000]);
        for y in y0..=y1 + 1 {
            let t = days_from_civil(y, 1, 1) as f64;
            if y.rem_euclid(iv) == 0 && inside(t) {
                ticks.push((t, y.to_string()));
            }
        }
    } else if months >= MINTICKS {
        let iv = pick(months, &[1, 2, 3, 4, 6]) as u32;
        for y in y0..=y1 {
            for m in (1..=12).step_by(iv as usize) {
                let t = days_from_civil(y, m, 1) as f64;
                if inside(t) {
                    let label = if m == 1 { y.to_string() } else { MONTHS[m as usize - 1].into() };
                    ticks.push((t, label));
                }
            }
        }
    } else if days >= MINTICKS {
        let iv = pick(days, &[1, 2, 3, 7, 14, 21]) as u32;
        // matplotlib's special cases: weekly and fortnightly ticks skip the 29th,
        // which would otherwise sit right beside the next month's 1st.
        let monthdays: Vec<u32> = match iv {
            7 => vec![1, 8, 15, 22],
            14 => vec![1, 15],
            _ => (1..=31).step_by(iv as usize).collect(),
        };
        for day in t0.floor() as i64..=t1.ceil() as i64 {
            let (_, m, d) = civil(day);
            if monthdays.contains(&d) && inside(day as f64) {
                let label = if d == 1 { MONTHS[m as usize - 1].to_string() } else { format!("{d:02}") };
                ticks.push((day as f64, label));
            }
        }
    } else {
        let iv = pick(hours, &[1, 2, 3, 4, 6, 12]);
        let (h0, h1) = ((t0 * 24.0).floor() as i64, (t1 * 24.0).ceil() as i64);
        for h in h0..=h1 {
            let t = h as f64 / 24.0;
            let hod = h.rem_euclid(24);
            if hod % iv == 0 && inside(t) {
                let label = if hod == 0 {
                    let (_, m, d) = civil(h.div_euclid(24));
                    format!("{}-{d:02}", MONTHS[m as usize - 1])
                } else {
                    format!("{hod:02}:00")
                };
                ticks.push((t, label));
            }
        }
    }
    ticks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_round_trips() {
        for d in [-1000, 0, 19_000, 20_604] {
            let (y, m, dd) = civil(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
        assert_eq!(civil(20_357), (2025, 9, 26));
    }

    #[test]
    fn formats_like_python_g() {
        assert_eq!(fmt_g(50.0), "50");
        assert_eq!(fmt_g(0.5), "0.5");
        assert_eq!(fmt_g(0.2), "0.2");
        assert_eq!(fmt_g(1e6), "1e+06");
    }

    #[test]
    fn log_ticks_label_1_2_5() {
        let (maj, min) = log_ticks(0.7, 70.0);
        let labelled: Vec<_> = maj.iter().chain(&min).filter(|t| !t.1.is_empty())
            .map(|t| t.1.as_str()).collect();
        assert!(labelled.contains(&"1") && labelled.contains(&"2") && labelled.contains(&"50"));
        assert!(!labelled.contains(&"3"));
    }

    #[test]
    fn drawdown_ticks_step_fifty() {
        assert_eq!(nice_ticks(-105.0, 2.0, 3), vec![-100.0, -50.0, 0.0]);
    }

    #[test]
    fn weekly_ticks_skip_the_29th() {
        let t0 = days_from_civil(2024, 7, 15) as f64;
        let t1 = days_from_civil(2024, 8, 30) as f64;
        let labels: Vec<_> = date_ticks(t0, t1).into_iter().map(|t| t.1).collect();
        assert_eq!(labels, ["15", "22", "Aug", "08", "15", "22"]);
    }

    #[test]
    fn two_years_tick_every_four_months() {
        let t0 = days_from_civil(2024, 7, 15) as f64;
        let t1 = days_from_civil(2026, 9, 26) as f64;
        let labels: Vec<_> = date_ticks(t0, t1).into_iter().map(|t| t.1).collect();
        assert_eq!(labels, ["Sep", "2025", "May", "Sep", "2026", "May", "Sep"]);
    }
}
