//! Font loading, text measurement and anti-aliased text drawing.
//!
//! Sizes are given as an em size in pixels, like matplotlib's `fontsize` once
//! converted from points, so the same numbers produce the same text size.

use ab_glyph::{point, Font, FontVec, GlyphId, PxScale, ScaleFont};

use crate::canvas::{Canvas, Rgb};

/// A loaded font face.
pub struct Face {
    font: FontVec,
    /// `(ascent - descent) / units_per_em`: converts an em size to ab_glyph's scale.
    height_per_em: f32,
}

/// Horizontal alignment of a text anchor.
#[derive(Clone, Copy)]
pub enum HAlign {
    Left,
    Center,
    Right,
}

/// Vertical alignment of a text anchor.
#[derive(Clone, Copy)]
pub enum VAlign {
    Top,
    Center,
    Bottom,
}

/// Glyphs placed along a baseline starting at x = 0.
pub struct Layout {
    glyphs: Vec<(GlyphId, f32)>,
    scale: PxScale,
    pub width: f32,
    pub ascent: f32,
    /// Negative: distance below the baseline.
    pub descent: f32,
}

impl Face {
    /// Load face `index` of a font file (`.ttf`, `.otf`, or a `.ttc` collection).
    pub fn load(path: &str, index: u32) -> Result<Face, String> {
        let data = std::fs::read(path).map_err(|e| format!("cannot read font {path}: {e}"))?;
        let font = FontVec::try_from_vec_and_index(data, index)
            .map_err(|e| format!("cannot parse font {path} (face {index}): {e}"))?;
        let upem = font.units_per_em().unwrap_or(1000.0);
        let height_per_em = font.height_unscaled() / upem;
        Ok(Face { font, height_per_em })
    }

    /// Lay out one line of `text` at `em_px` pixels per em, with kerning.
    pub fn layout(&self, text: &str, em_px: f32) -> Layout {
        let scale = PxScale::from(em_px * self.height_per_em);
        let sf = self.font.as_scaled(scale);
        let mut x = 0.0;
        let mut prev: Option<GlyphId> = None;
        let mut glyphs = Vec::with_capacity(text.len());
        for ch in text.chars() {
            let id = sf.glyph_id(ch);
            if let Some(p) = prev {
                x += sf.kern(p, id);
            }
            glyphs.push((id, x));
            x += sf.h_advance(id);
            prev = Some(id);
        }
        Layout { glyphs, scale, width: x, ascent: sf.ascent(), descent: sf.descent() }
    }

    /// Width of `text` in pixels.
    pub fn width(&self, text: &str, em_px: f32) -> f32 {
        self.layout(text, em_px).width
    }

    /// Visit each covered pixel of `layout` placed with its baseline origin at
    /// `(x, baseline)`: `f(px, py, coverage)`.
    fn raster(&self, layout: &Layout, x: f32, baseline: f32, mut f: impl FnMut(i32, i32, f32)) {
        for &(id, gx) in &layout.glyphs {
            let glyph = id.with_scale_and_position(layout.scale, point(x + gx, baseline));
            if let Some(outlined) = self.font.outline_glyph(glyph) {
                let b = outlined.px_bounds();
                outlined.draw(|px, py, c| f(b.min.x as i32 + px as i32, b.min.y as i32 + py as i32, c));
            }
        }
    }
}

/// Where the baseline origin goes so the text box is anchored at `(x, y)`.
fn origin(l: &Layout, x: f32, y: f32, ha: HAlign, va: VAlign) -> (f32, f32) {
    let x0 = match ha {
        HAlign::Left => x,
        HAlign::Center => x - l.width / 2.0,
        HAlign::Right => x - l.width,
    };
    let base = match va {
        VAlign::Top => y + l.ascent,
        VAlign::Center => y + (l.ascent + l.descent) / 2.0,
        VAlign::Bottom => y + l.descent,
    };
    (x0, base)
}

/// Draw `text` anchored at `(x, y)` with the given alignment.
#[allow(clippy::too_many_arguments)]
pub fn draw(canvas: &mut Canvas, face: &Face, text: &str, em_px: f32, x: f32, y: f32, ha: HAlign,
            va: VAlign, color: Rgb, alpha: f32) {
    if text.is_empty() || alpha <= 0.0 {
        return;
    }
    let l = face.layout(text, em_px);
    let (x0, base) = origin(&l, x, y, ha, va);
    face.raster(&l, x0, base, |px, py, c| canvas.blend(px, py, color, c * alpha));
}

/// Draw `text` rotated 90 degrees counter-clockwise (reading bottom to top),
/// centred on `(cx, cy)`. Used for y-axis labels; pixels map exactly, no resampling.
pub fn draw_vertical(canvas: &mut Canvas, face: &Face, text: &str, em_px: f32, cx: f32, cy: f32,
                     color: Rgb) {
    let l = face.layout(text, em_px);
    let (w, h) = (l.width.ceil() as i32 + 2, (l.ascent - l.descent).ceil() as i32 + 2);
    let mut buf = vec![0.0f32; (w * h) as usize];
    face.raster(&l, 1.0, 1.0 + l.ascent, |px, py, c| {
        if (0..w).contains(&px) && (0..h).contains(&py) {
            let v = &mut buf[(py * w + px) as usize];
            *v = (*v + c).min(1.0);
        }
    });
    // Rotated box is h wide and w tall; buffer (u, v) lands at (v, w - 1 - u).
    let (left, top) = ((cx - h as f32 / 2.0).round() as i32, (cy - w as f32 / 2.0).round() as i32);
    for v in 0..h {
        for u in 0..w {
            let c = buf[(v * w + u) as usize];
            if c > 0.0 {
                canvas.blend(left + v, top + (w - 1 - u), color, c);
            }
        }
    }
}
