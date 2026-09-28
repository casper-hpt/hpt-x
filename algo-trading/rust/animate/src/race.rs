//! The equity-race scene: lay out once, then draw any frame on its own.
//!
//! Every frame is a pure function of its index, so frames render in parallel.
//! Layout and styling follow the matplotlib renderer in
//! `algo_trading/backtest/animate.py` (`_render_mpl`); keep the two in step.

use tiny_skia::{Pixmap, PixmapPaint, Transform};

use crate::canvas::{Canvas, Cap, Rgb};
use crate::text::{self, Face, HAlign, VAlign};
use crate::ticks::{self, Tick};

// Text sizes in points (animate._TEXT).
const TITLE: f32 = 17.0;
const SUBTITLE: f32 = 11.0;
const LABEL: f32 = 10.5;
const TICK: f32 = 10.0;
const DATE: f32 = 13.0;
const CAPTION: f32 = 13.0;
/// Points from a tip dot to its tag.
const TAG_OFFSET: f32 = 14.0;
/// matplotlib's tick-label pad and axis-label pad, in points.
const TICK_PAD: f32 = 3.5;
const LABEL_PAD: f32 = 4.0;
/// Points between the icon and the title block.
const ICON_GAP: f32 = 10.0;

/// Icon height in points: it spans the title block (title, plus subtitle if any).
pub fn icon_height(has_subtitle: bool) -> f32 {
    if has_subtitle { TITLE * 1.4 + SUBTITLE } else { TITLE * 1.2 }
}

/// Colour tokens for one surface.
pub struct Theme {
    pub ink: Rgb,
    pub ink_muted: Rgb,
    pub grid: Rgb,
    pub surface: Rgb,
}

/// Everything the Python side hands over.
pub struct Spec {
    /// Bar times in days since the epoch (fractional for intraday bars).
    pub x: Vec<f64>,
    /// Growth of 1, `n_bars x n_curves` row-major; NaN before a curve starts.
    pub growth: Vec<f64>,
    pub labels: Vec<String>,
    pub colors: Vec<Rgb>,
    /// Curve whose line is emphasised and whose drawdown is filled.
    pub primary: usize,
    pub title: String,
    pub subtitle: String,
    pub caption: String,
    pub log: bool,
    pub width: u32,
    pub height: u32,
    pub dpi: f32,
    pub theme: Theme,
    pub fill_alpha: f32,
    /// matplotlib's `axes.labelsize`, in points.
    pub label_size: f32,
    pub draw_frames: usize,
    pub hold_frames: usize,
    pub fps: u32,
    /// Pre-rasterised icon, drawn at the left of the title block.
    pub icon: Option<Pixmap>,
}

#[derive(Clone, Copy)]
struct Rect {
    l: f32,
    t: f32,
    w: f32,
    h: f32,
}

impl Rect {
    fn r(&self) -> f32 {
        self.l + self.w
    }
    fn b(&self) -> f32 {
        self.t + self.h
    }
}

/// A laid-out scene, ready to draw any frame.
pub struct Scene {
    spec: Spec,
    regular: Face,
    bold: Face,
    /// Pixels per point.
    pt: f32,
    n: usize,
    m: usize,
    dd: Vec<f64>,
    tip: Vec<f64>,
    growth_ax: Rect,
    dd_ax: Rect,
    g_lim: (f64, f64),
    dd_lim: (f64, f64),
    /// Growth y ticks: gridded majors and ungridded minors (log only).
    g_major: Vec<Tick>,
    g_minor: Vec<Tick>,
    dd_ticks: Vec<Tick>,
    g_ylabel_x: f32,
    dd_ylabel_x: f32,
    k0: usize,
}

fn nan_min_max(v: &[f64]) -> (f64, f64) {
    v.iter().filter(|x| x.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)))
}

/// Push positions apart by at least `gap` keeping their order, then re-centre.
fn spread(ys: &[f32], gap: f32) -> Vec<f32> {
    let mut order: Vec<usize> = (0..ys.len()).collect();
    order.sort_by(|&a, &b| ys[a].total_cmp(&ys[b]));
    let mut placed = ys.to_vec();
    for w in order.windows(2) {
        placed[w[1]] = placed[w[1]].max(placed[w[0]] + gap);
    }
    let shift = placed.iter().zip(ys).map(|(p, y)| p - y).sum::<f32>() / ys.len().max(1) as f32;
    placed.iter().map(|p| p - shift).collect()
}

impl Scene {
    pub fn new(spec: Spec, regular: Face, bold: Face) -> Result<Scene, String> {
        let (n, m) = (spec.x.len(), spec.labels.len());
        if n < 2 || m == 0 || spec.growth.len() != n * m || spec.colors.len() != m {
            return Err("need >= 2 bars, >= 1 curve, and one colour per curve".into());
        }
        if spec.draw_frames == 0 {
            return Err("seconds * fps must be at least one frame".into());
        }
        let pt = spec.dpi / 72.0;
        let (w, h) = (spec.width as f32, spec.height as f32);

        // Drawdown (%) and last-known value per curve.
        let (mut dd, mut tip) = (vec![f64::NAN; n * m], vec![f64::NAN; n * m]);
        for c in 0..m {
            let (mut peak, mut last) = (f64::NEG_INFINITY, f64::NAN);
            for t in 0..n {
                let v = spec.growth[t * m + c];
                if v.is_finite() {
                    peak = peak.max(v);
                    dd[t * m + c] = (v / peak - 1.0) * 100.0;
                    last = v;
                }
                tip[t * m + c] = last;
            }
        }
        let (lo, hi) = nan_min_max(&spec.growth);
        if !lo.is_finite() || (spec.log && lo <= 0.0) {
            return Err("growth has no finite values (or non-positive values on a log axis)".into());
        }

        // Vertical layout in points, top down (y grows downward here).
        let mut y = 16.0 * pt + TITLE * 1.4 * pt;
        if !spec.subtitle.is_empty() {
            y += SUBTITLE * 1.5 * pt;
        }
        let top = y + DATE * 2.0 * pt;
        let bottom = h - (10.0 + CAPTION * 1.6 + TICK * 2.2) * pt;
        let dd_h = 0.26 * (bottom - top);

        // Horizontal layout: the right margin holds the widest tag.
        let tag_w = spec.labels.iter()
            .map(|l| regular.width(l, LABEL * pt))
            .fold(0.0, f32::max);
        let left = 0.03 * w + spec.label_size * 1.6 * pt + regular.width("-100%", TICK * pt)
            + 8.0 * pt;
        let right = 0.97 * w - tag_w - TAG_OFFSET * pt;
        if right - left < 40.0 {
            return Err("frame too narrow for the labels; use a larger size".into());
        }
        let dd_ax = Rect { l: left, t: bottom - dd_h, w: right - left, h: dd_h };
        let growth_ax = Rect { l: left, t: top, w: right - left, h: bottom - dd_h - 14.0 * pt - top };

        // Y limits and ticks.
        let g_lim = if spec.log {
            (lo / 1.15, hi * 1.15)
        } else {
            let pad = (hi - lo) * 0.06;
            (lo - pad, hi + pad)
        };
        let nbins = |ax: &Rect| (ax.h / pt / (TICK * 2.0)).floor().max(1.0) as usize;
        let (g_major, g_minor) = if spec.log {
            ticks::log_ticks(g_lim.0, g_lim.1)
        } else {
            let t = ticks::nice_ticks(g_lim.0, g_lim.1, nbins(&growth_ax));
            (t.into_iter().map(|v| (v, ticks::fmt_g(v))).collect(), vec![])
        };
        let dd_lim = (nan_min_max(&dd).0.min(0.0) * 1.08, 2.0);
        let dd_lim = (dd_lim.0.min(-1.0), dd_lim.1);
        let dd_ticks = ticks::nice_ticks(dd_lim.0, dd_lim.1, nbins(&dd_ax))
            .into_iter().map(|v| (v, format!("{v:.0}%"))).collect::<Vec<_>>();

        // Y-axis labels sit left of their widest tick label.
        let label_h = {
            let l = regular.layout("portfolio growth", spec.label_size * pt);
            l.ascent - l.descent
        };
        let ylabel_x = |ticks: &[&Tick]| {
            let tw = ticks.iter().map(|t| regular.width(&t.1, TICK * pt)).fold(0.0, f32::max);
            left - (TICK_PAD + LABEL_PAD) * pt - tw - label_h / 2.0
        };
        let g_labelled: Vec<&Tick> = g_major.iter().chain(&g_minor).filter(|t| !t.1.is_empty())
            .filter(|t| t.0 >= g_lim.0 && t.0 <= g_lim.1).collect();
        let g_ylabel_x = ylabel_x(&g_labelled);
        let dd_ylabel_x = ylabel_x(&dd_ticks.iter().collect::<Vec<_>>());

        Ok(Scene {
            k0: (n / 30).max(2),
            spec, regular, bold, pt, n, m, dd, tip, growth_ax, dd_ax, g_lim, dd_lim, g_major,
            g_minor, dd_ticks, g_ylabel_x, dd_ylabel_x,
        })
    }

    pub fn total_frames(&self) -> usize {
        self.spec.draw_frames + self.spec.hold_frames
    }

    pub fn width(&self) -> u32 {
        self.spec.width
    }

    pub fn height(&self) -> u32 {
        self.spec.height
    }

    pub fn fps(&self) -> u32 {
        self.spec.fps
    }

    fn gy(&self, v: f64) -> f32 {
        let ax = &self.growth_ax;
        let frac = if self.spec.log {
            if v <= 0.0 { return f32::NAN; }
            (v.ln() - self.g_lim.0.ln()) / (self.g_lim.1.ln() - self.g_lim.0.ln())
        } else {
            (v - self.g_lim.0) / (self.g_lim.1 - self.g_lim.0)
        };
        ax.b() - (frac as f32) * ax.h
    }

    fn ddy(&self, v: f64) -> f32 {
        let ax = &self.dd_ax;
        ax.b() - ((v - self.dd_lim.0) / (self.dd_lim.1 - self.dd_lim.0)) as f32 * ax.h
    }

    /// Draw frame `f` into `c`.
    pub fn draw(&self, f: usize, c: &mut Canvas) {
        let s = &self.spec;
        let th = &s.theme;
        let pt = self.pt;
        let (n, m) = (self.n, self.m);
        let k = (self.k0 + ((n - self.k0) as f64 * (f + 1) as f64 / s.draw_frames as f64).ceil()
            as usize).min(n);
        let (x0, x1) = (s.x[0], s.x[k - 1]);
        let span = (x1 - x0).max(1e-9);
        let (gax, dax) = (self.growth_ax, self.dd_ax);
        let px = |x: f64| gax.l + ((x - x0) / span) as f32 * gax.w;

        c.clear(th.surface);

        // Grid, under everything else.
        let xt = ticks::date_ticks(x0, x1);
        let grid = |c: &mut Canvas, x0: f32, y0: f32, x1: f32, y1: f32| {
            c.line(x0, y0, x1, y1, th.grid, 0.9, 0.6 * pt, None);
        };
        for ax in [&gax, &dax] {
            for (t, _) in &xt {
                grid(c, px(*t), ax.t, px(*t), ax.b());
            }
        }
        for (v, _) in &self.g_major {
            let y = self.gy(*v);
            grid(c, gax.l, y, gax.r(), y);
        }
        for (v, _) in &self.dd_ticks {
            let y = self.ddy(*v);
            grid(c, dax.l, y, dax.r(), y);
        }

        // Growth panel: break-even line, then curves (primary on top).
        if (self.g_lim.0..=self.g_lim.1).contains(&1.0) {
            let y = self.gy(1.0);
            c.line(gax.l, y, gax.r(), y, th.ink_muted, 1.0, 0.8 * pt,
                   Some([0.8 * pt, 0.8 * 1.65 * pt]));
        }
        let order: Vec<usize> = (0..m).filter(|&i| i != s.primary).chain([s.primary]).collect();
        for &i in &order {
            let w = if i == s.primary { 2.4 } else { 1.6 };
            c.polyline((0..k).map(|t| (px(s.x[t]), self.gy(s.growth[t * m + i]))), s.colors[i],
                       w * pt, Cap::Round);
        }

        // Drawdown panel: the primary's filled area, then every curve's line.
        let p = s.primary;
        let mut area: Vec<(f32, f32)> = Vec::with_capacity(k + 2);
        area.push((px(s.x[0]), self.ddy(0.0)));
        area.extend((0..k).map(|t| {
            let v = self.dd[t * m + p];
            (px(s.x[t]), self.ddy(if v.is_finite() { v } else { 0.0 }))
        }));
        area.push((px(s.x[k - 1]), self.ddy(0.0)));
        c.polygon(&area, s.colors[p], s.fill_alpha);
        for i in 0..m {
            let w = if i == p { 1.4 } else { 1.0 };
            c.polyline((0..k).map(|t| (px(s.x[t]), self.ddy(self.dd[t * m + i]))),
                       s.colors[i], w * pt, Cap::Round);
        }

        // Spines (left and bottom), ticks and labels.
        for ax in [&gax, &dax] {
            c.line(ax.l, ax.t, ax.l, ax.b(), th.grid, 1.0, 0.8 * pt, None);
            c.line(ax.l, ax.b(), ax.r(), ax.b(), th.grid, 1.0, 0.8 * pt, None);
        }
        let ytick = |c: &mut Canvas, y: f32, label: &str| {
            text::draw(c, &self.regular, label, TICK * pt, gax.l - TICK_PAD * pt, y, HAlign::Right,
                       VAlign::Center, th.ink_muted, 1.0);
        };
        for (v, label) in &self.g_minor {
            let y = self.gy(*v);
            c.line(gax.l - 2.0 * pt, y, gax.l, y, th.ink_muted, 1.0, 0.6 * pt, None);
            ytick(c, y, label);
        }
        for (v, label) in &self.g_major {
            ytick(c, self.gy(*v), label);
        }
        for (v, label) in &self.dd_ticks {
            ytick(c, self.ddy(*v), label);
        }
        for (t, label) in &xt {
            text::draw(c, &self.regular, label, TICK * pt, px(*t), dax.b() + TICK_PAD * pt,
                       HAlign::Center, VAlign::Top, th.ink_muted, 1.0);
        }
        let ls = s.label_size * pt;
        text::draw_vertical(c, &self.regular, "portfolio growth", ls, self.g_ylabel_x,
                            gax.t + gax.h / 2.0, th.ink_muted);
        text::draw_vertical(c, &self.regular, "drawdown", ls, self.dd_ylabel_x,
                            dax.t + dax.h / 2.0, th.ink_muted);

        // Tags beside each current value, nudged apart, with leader lines; dots on top.
        let tips: Vec<(usize, f64, f32)> = (0..m)
            .map(|i| (i, self.tip[(k - 1) * m + i]))
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (i, v, self.gy(v)))
            .collect();
        let up: Vec<f32> = tips.iter().map(|t| -t.2).collect();
        let placed = spread(&up, LABEL * 1.45 * pt);
        let dot_x = px(x1);
        let text_x = dot_x + TAG_OFFSET * pt;
        for (&(i, _, y), &u) in tips.iter().zip(&placed) {
            let ty = -u;
            let (ax_, ay_) = (text_x, ty);
            let (dx, dy) = (dot_x - ax_, y - ay_);
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 8.0 * pt {
                let (ux, uy) = (dx / dist, dy / dist);
                c.line(ax_ + ux * 3.0 * pt, ay_ + uy * 3.0 * pt, dot_x - ux * 5.0 * pt,
                       y - uy * 5.0 * pt, s.colors[i], 1.0, 0.9 * pt, None);
            }
            text::draw(c, &self.regular, &s.labels[i], LABEL * pt, text_x, ty, HAlign::Left,
                       VAlign::Center, th.ink, 1.0);
        }
        for &(i, _, y) in &tips {
            c.dot(dot_x, y, 3.25 * pt, s.colors[i], th.surface, 1.5 * pt);
        }

        // Icon then headline text, the running date, and the caption fading in on the hold.
        let fx = 0.04 * s.width as f32;
        let mut tx = fx;
        if let Some(icon) = &s.icon {
            c.pixmap.draw_pixmap(fx.round() as i32, (16.0 * pt).round() as i32, icon.as_ref(),
                                 &PixmapPaint::default(), Transform::identity(), None);
            tx += icon.width() as f32 + ICON_GAP * pt;
        }
        text::draw(c, &self.bold, &s.title, TITLE * pt, tx, 16.0 * pt, HAlign::Left, VAlign::Top,
                   th.ink, 1.0);
        text::draw(c, &self.regular, &s.subtitle, SUBTITLE * pt, tx, (16.0 + TITLE * 1.4) * pt,
                   HAlign::Left, VAlign::Top, th.ink_muted, 1.0);
        text::draw(c, &self.bold, &ticks::month_year(x1), DATE * pt, gax.l,
                   gax.t - DATE * 0.5 * pt, HAlign::Left, VAlign::Bottom, th.ink_muted, 1.0);
        if f >= s.draw_frames {
            let a = ((f - s.draw_frames + 1) as f32 / (0.6 * s.fps as f32)).min(1.0);
            text::draw(c, &self.regular, &s.caption, CAPTION * pt, fx,
                       s.height as f32 - 10.0 * pt, HAlign::Left, VAlign::Bottom, th.ink, a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_keeps_order_and_gap() {
        let out = spread(&[0.0, 1.0, 2.0], 10.0);
        assert!(out[1] - out[0] >= 10.0 - 1e-4 && out[2] - out[1] >= 10.0 - 1e-4);
        assert!((out.iter().sum::<f32>() - 3.0).abs() < 1e-4); // re-centred
    }
}
