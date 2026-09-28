//! A frame buffer with the few primitives the chart needs, on top of tiny-skia.

use tiny_skia::{
    Color, FillRule, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, StrokeDash, Transform,
};

/// An opaque sRGB colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    /// Parse `#rrggbb`.
    pub fn parse(hex: &str) -> Result<Rgb, String> {
        let h = hex.trim_start_matches('#');
        let byte = |i: usize| u8::from_str_radix(h.get(i..i + 2).unwrap_or("zz"), 16);
        match (h.len(), byte(0), byte(2), byte(4)) {
            (6, Ok(r), Ok(g), Ok(b)) => Ok(Rgb(r, g, b)),
            _ => Err(format!("expected a #rrggbb colour, got {hex:?}")),
        }
    }

    fn paint(self, alpha: f32) -> Paint<'static> {
        let mut p = Paint::default();
        p.set_color(Color::from_rgba8(self.0, self.1, self.2, (alpha.clamp(0.0, 1.0) * 255.0) as u8));
        p.anti_alias = true;
        p
    }
}

/// How a stroke's ends look.
#[derive(Clone, Copy)]
pub enum Cap {
    Butt,
    Round,
}

/// An RGBA frame; always opaque once `clear` has run.
pub struct Canvas {
    pub pixmap: Pixmap,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Canvas {
        Canvas { pixmap: Pixmap::new(width, height).expect("frame size must be non-zero") }
    }

    pub fn clear(&mut self, c: Rgb) {
        self.pixmap.fill(Color::from_rgba8(c.0, c.1, c.2, 255));
    }

    /// Source-over one pixel of `c` at `coverage` (0..1). The canvas is opaque,
    /// so premultiplied and straight alpha coincide.
    #[inline]
    pub fn blend(&mut self, x: i32, y: i32, c: Rgb, coverage: f32) {
        let (w, h) = (self.pixmap.width() as i32, self.pixmap.height() as i32);
        if x < 0 || y < 0 || x >= w || y >= h || coverage <= 0.0 {
            return;
        }
        let a = coverage.min(1.0);
        let i = ((y * w + x) * 4) as usize;
        let px = &mut self.pixmap.data_mut()[i..i + 3];
        for (d, s) in px.iter_mut().zip([c.0, c.1, c.2]) {
            *d = (s as f32 * a + *d as f32 * (1.0 - a) + 0.5) as u8;
        }
    }

    /// Stroke a polyline; `NaN` y values break it into separate runs.
    pub fn polyline(&mut self, pts: impl IntoIterator<Item = (f32, f32)>, c: Rgb, width: f32,
                    cap: Cap) {
        let mut pb = PathBuilder::new();
        let mut pen_down = false;
        for (x, y) in pts {
            if !y.is_finite() {
                pen_down = false;
            } else if pen_down {
                pb.line_to(x, y);
            } else {
                pb.move_to(x, y);
                pen_down = true;
            }
        }
        if let Some(path) = pb.finish() {
            let stroke = Stroke {
                width,
                line_cap: match cap { Cap::Butt => LineCap::Butt, Cap::Round => LineCap::Round },
                line_join: LineJoin::Round,
                ..Stroke::default()
            };
            self.pixmap.stroke_path(&path, &c.paint(1.0), &stroke, Transform::identity(), None);
        }
    }

    /// A straight segment, optionally dashed (`dash` = on/off lengths in px).
    #[allow(clippy::too_many_arguments)]
    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, c: Rgb, alpha: f32, width: f32,
                dash: Option<[f32; 2]>) {
        let mut pb = PathBuilder::new();
        pb.move_to(x0, y0);
        pb.line_to(x1, y1);
        if let Some(path) = pb.finish() {
            let stroke = Stroke {
                width,
                line_cap: LineCap::Butt,
                dash: dash.and_then(|d| StrokeDash::new(d.to_vec(), 0.0)),
                ..Stroke::default()
            };
            self.pixmap.stroke_path(&path, &c.paint(alpha), &stroke, Transform::identity(), None);
        }
    }

    /// Fill a closed polygon.
    pub fn polygon(&mut self, pts: &[(f32, f32)], c: Rgb, alpha: f32) {
        let mut pb = PathBuilder::new();
        for (i, &(x, y)) in pts.iter().enumerate() {
            if i == 0 { pb.move_to(x, y) } else { pb.line_to(x, y) }
        }
        pb.close();
        if let Some(path) = pb.finish() {
            self.pixmap.fill_path(&path, &c.paint(alpha), FillRule::Winding,
                                  Transform::identity(), None);
        }
    }

    /// A filled dot with a ring in `ring` colour (matplotlib's marker edge).
    pub fn dot(&mut self, x: f32, y: f32, r: f32, fill: Rgb, ring: Rgb, ring_w: f32) {
        if let Some(path) = PathBuilder::from_circle(x, y, r) {
            self.pixmap.fill_path(&path, &fill.paint(1.0), FillRule::Winding,
                                  Transform::identity(), None);
            let stroke = Stroke { width: ring_w, ..Stroke::default() };
            self.pixmap.stroke_path(&path, &ring.paint(1.0), &stroke, Transform::identity(), None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex() {
        assert_eq!(Rgb::parse("#3987e5").unwrap(), Rgb(0x39, 0x87, 0xe5));
        assert!(Rgb::parse("blue").is_err());
    }
}
