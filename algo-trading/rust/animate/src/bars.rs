//! The risk-bars scene: capital weights beside risk contributions, morphing
//! from one portfolio to the next.
//!
//! Layout and styling follow `animate._render_bars` in
//! `algo_trading/backtest/animate.py`; keep the two in step.

use tiny_skia::Pixmap;

use crate::canvas::{Canvas, Rgb};
use crate::race::{self, Frames, Theme, CAPTION, LABEL, TICK, TITLE};
use crate::text::{self, Face, HAlign, VAlign};

/// Everything the Python side hands over.
pub struct Spec {
    /// Stage names, in playing order.
    pub names: Vec<String>,
    /// Capital weights, `n_stages x n_assets` row-major; each row sums to 1.
    pub weights: Vec<f64>,
    /// Annualised covariance, `n_assets x n_assets` row-major.
    pub cov: Vec<f64>,
    pub assets: Vec<String>,
    pub colors: Vec<Rgb>,
    pub title: String,
    pub subtitle: String,
    /// One per stage; empty strings draw nothing.
    pub captions: Vec<String>,
    pub move_frames: usize,
    pub hold_frames: usize,
    pub width: u32,
    pub height: u32,
    pub dpi: f32,
    pub fps: u32,
    pub theme: Theme,
    pub icon: Option<Pixmap>,
}

/// Axes in pixels, with the data ranges the bars are drawn in.
#[derive(Clone, Copy)]
struct Axes {
    l: f32,
    t: f32,
    w: f32,
    h: f32,
}

const Y_MAX: f64 = 108.0; // headroom for a tag on a near-100% bar

impl Axes {
    /// x of data position `v` on the `[-0.6, n - 0.4]` axis.
    fn x(&self, v: f64, n: usize) -> f32 {
        self.l + ((v + 0.6) / (n as f64)) as f32 * self.w
    }
    fn y(&self, v: f64) -> f32 {
        self.t + self.h - (v / Y_MAX) as f32 * self.h
    }
}

/// Each asset's share of portfolio variance: `w_i (cov w)_i / (w' cov w)`.
pub fn risk_contributions(w: &[f64], cov: &[f64]) -> Vec<f64> {
    let n = w.len();
    let m: Vec<f64> = (0..n).map(|i| (0..n).map(|j| cov[i * n + j] * w[j]).sum()).collect();
    let var: f64 = w.iter().zip(&m).map(|(a, b)| a * b).sum();
    if var <= 0.0 {
        return vec![0.0; n];
    }
    w.iter().zip(&m).map(|(a, b)| a * b / var).collect()
}

fn ease(u: f64) -> f64 {
    u * u * (3.0 - 2.0 * u) // smoothstep: no jolt at either end
}

pub struct Scene {
    spec: Spec,
    regular: Face,
    bold: Face,
    pt: f32,
    n: usize,
    stage_y: f32,
    axes: [Axes; 2],
}

impl Scene {
    pub fn new(spec: Spec, regular: Face, bold: Face) -> Result<Scene, String> {
        let n = spec.assets.len();
        let s = spec.names.len();
        if n == 0 || s == 0 || spec.weights.len() != s * n || spec.cov.len() != n * n
            || spec.colors.len() != n || spec.captions.len() != s {
            return Err("need one weight per asset per stage, an n x n covariance, one colour \
                        per asset and one caption per stage".into());
        }
        if spec.move_frames == 0 {
            return Err("move * fps must be at least one frame".into());
        }
        let pt = spec.dpi / 72.0;
        let (w, h) = (spec.width as f32, spec.height as f32);
        let stage_y = race::header_bottom(!spec.subtitle.is_empty(), pt) + 14.0 * pt;
        let top = stage_y + TITLE * 1.25 * 1.5 * pt + LABEL * 2.2 * pt;
        let bottom = h - (12.0 + CAPTION * 2.6 + TICK * 1.8) * pt;
        let gap = 64.0 * pt;                     // room for the equal-risk label
        let pw = (0.92 * w - gap) / 2.0;
        let axes = [0, 1].map(|k| Axes { l: 0.04 * w + k as f32 * (pw + gap), t: top, w: pw,
                                          h: bottom - top });
        Ok(Scene { spec, regular, bold, pt, n, stage_y, axes })
    }

    fn stage_frames(&self) -> usize {
        self.spec.move_frames + self.spec.hold_frames
    }
}

impl Frames for Scene {
    fn total_frames(&self) -> usize {
        self.spec.names.len() * self.stage_frames()
    }

    fn width(&self) -> u32 {
        self.spec.width
    }

    fn height(&self) -> u32 {
        self.spec.height
    }

    fn fps(&self) -> u32 {
        self.spec.fps
    }

    fn draw(&self, frame: usize, c: &mut Canvas) {
        let sp = &self.spec;
        let th = &sp.theme;
        let (pt, n) = (self.pt, self.n);
        let (st, f) = (frame / self.stage_frames(), frame % self.stage_frames());
        let e = ease(((f + 1) as f64 / sp.move_frames as f64).min(1.0));
        let target = &sp.weights[st * n..(st + 1) * n];
        // The first stage grows in from zero; later ones morph from the stage before.
        let (w, grow): (Vec<f64>, f64) = if st == 0 {
            (target.to_vec(), e)
        } else {
            let prev = &sp.weights[(st - 1) * n..st * n];
            (prev.iter().zip(target).map(|(a, b)| a + (b - a) * e).collect(), 1.0)
        };
        let rc = risk_contributions(&w, &sp.cov);
        let heights = [
            w.iter().map(|v| v * 100.0 * grow).collect::<Vec<_>>(),
            rc.iter().map(|v| v * 100.0 * grow).collect::<Vec<_>>(),
        ];
        let settled = st > 0 || e > 0.95;

        c.clear(th.surface);
        race::draw_header(c, &self.regular, &self.bold, sp.icon.as_ref(), &sp.title, &sp.subtitle,
                          sp.width, pt, th);

        // Stage name, and the portfolio's volatility on the right.
        let a = if st == 0 { e as f32 } else { 1.0 };
        text::draw(c, &self.bold, &sp.names[st], TITLE * 1.25 * pt, 0.04 * sp.width as f32,
                   self.stage_y, HAlign::Left, VAlign::Top, th.ink, a);
        if settled {
            let var: f64 = (0..n).map(|i| (0..n).map(|j| w[i] * sp.cov[i * n + j] * w[j])
                .sum::<f64>()).sum();
            text::draw(c, &self.regular,
                       &format!("portfolio volatility {:.0}%/yr", var.sqrt() * 100.0),
                       LABEL * pt, 0.96 * sp.width as f32, self.stage_y + TITLE * 0.25 * pt,
                       HAlign::Right, VAlign::Top, th.ink_muted, 1.0);
        }

        // The equal-risk line sits under the risk bars.
        let rax = self.axes[1];
        let eq = 100.0 / n as f64;
        c.line(rax.l, rax.y(eq), rax.l + rax.w, rax.y(eq), th.ink_muted, 1.0, 1.1 * pt,
               Some([4.0 * 1.1 * pt, 3.0 * 1.1 * pt]));
        let fs = TICK * 0.9 * pt;
        let lh = fs * 1.1 * 1.2 / 2.0;          // half a line at linespacing 1.1
        for (dy, word) in [(-lh, "equal"), (lh, "risk")] {
            text::draw(c, &self.regular, word, fs, rax.x(-0.7, n), rax.y(eq) + dy, HAlign::Right,
                       VAlign::Center, th.ink_muted, 1.0);
        }

        let heads = ["Capital: where the money is", "Risk: what moves the portfolio"];
        for (k, ax) in self.axes.iter().enumerate() {
            text::draw(c, &self.regular, heads[k], LABEL * pt, ax.l, ax.t - LABEL * 0.8 * pt,
                       HAlign::Left, VAlign::Bottom, th.ink_muted, 1.0);
            let half = (0.72 / 2.0 / n as f64) as f32 * ax.w;
            for (i, &v) in heights[k].iter().enumerate() {
                let xc = ax.x(i as f64, n);
                let (y0, y1) = (ax.y(0.0), ax.y(v));
                if y0 - y1 > 0.0 {
                    c.polygon(&[(xc - half, y1), (xc + half, y1), (xc + half, y0), (xc - half, y0)],
                              sp.colors[i], 1.0);
                }
                if settled {
                    text::draw(c, &self.bold, &format!("{v:.0}%"), LABEL * 1.3 * pt, xc,
                               ax.y(v + 1.5), HAlign::Center, VAlign::Bottom, th.ink, 1.0);
                }
                text::draw(c, &self.regular, &sp.assets[i], TICK * pt, xc, y0 + 6.0 * pt,
                           HAlign::Center, VAlign::Top, th.ink, 1.0);
            }
            c.line(ax.l, ax.y(0.0), ax.l + ax.w, ax.y(0.0), th.ink_muted, 1.0, 0.8 * pt, None);
        }

        // The caption fades in once the stage has settled.
        let cap = &sp.captions[st];
        let ca = ((f as f32 - sp.move_frames as f32 + 1.0) / (0.4 * sp.fps as f32)).clamp(0.0, 1.0);
        text::draw(c, &self.regular, cap, CAPTION * pt, 0.04 * sp.width as f32,
                   sp.height as f32 - 12.0 * pt, HAlign::Left, VAlign::Bottom, th.ink, ca);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_contributions_sum_to_one_and_equalise_uncorrelated_inverse_vol() {
        let cov = [0.04, 0.0, 0.0, 0.16];            // vols 20% and 40%, uncorrelated
        let rc = risk_contributions(&[2.0 / 3.0, 1.0 / 3.0], &cov);
        assert!((rc[0] - 0.5).abs() < 1e-12 && (rc[1] - 0.5).abs() < 1e-12);
        let rc = risk_contributions(&[0.5, 0.5], &cov);
        assert!((rc.iter().sum::<f64>() - 1.0).abs() < 1e-12 && rc[1] > rc[0]);
    }
}
