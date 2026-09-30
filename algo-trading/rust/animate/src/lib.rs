//! Python bindings for the video renderer.
//!
//! `algo_trading.backtest.animate` calls into this module (`equity_race`,
//! `risk_bars`, and `hold_frame` for stills) when it is installed and falls back
//! to matplotlib otherwise. Frames are drawn in parallel with tiny-skia and
//! piped to ffmpeg in order.

use std::io::Write as _;
use std::path::PathBuf;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

pub mod bars;
pub mod canvas;
pub mod race;
pub mod text;
pub mod ticks;
pub mod video;

use canvas::{Canvas, Rgb};
use race::{Frames, Scene, Spec, Theme};
use tiny_skia::{Pixmap, Transform};

/// Draws a source image into a pixmap at a given scale.
type Blit = Box<dyn Fn(&mut Pixmap, f32)>;

/// Load an SVG or PNG and rasterise it `height` pixels tall, keeping its aspect.
fn load_icon(path: &std::path::Path, height: f32) -> Result<Pixmap, String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read icon {}: {e}", path.display()))?;
    let is_svg = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("svg"));
    let (src_w, src_h, draw): (f32, f32, Blit) = if is_svg {
        let tree = resvg::usvg::Tree::from_data(&data, &resvg::usvg::Options::default())
            .map_err(|e| format!("cannot parse SVG {}: {e}", path.display()))?;
        let size = tree.size();
        (size.width(), size.height(),
         Box::new(move |pm: &mut Pixmap, k: f32| {
             resvg::render(&tree, Transform::from_scale(k, k), &mut pm.as_mut())
         }))
    } else {
        let src = Pixmap::decode_png(&data)
            .map_err(|e| format!("cannot decode PNG {}: {e}", path.display()))?;
        (src.width() as f32, src.height() as f32,
         Box::new(move |pm: &mut Pixmap, k: f32| {
             let paint = tiny_skia::PixmapPaint {
                 quality: tiny_skia::FilterQuality::Bicubic, ..Default::default()
             };
             pm.draw_pixmap(0, 0, src.as_ref(), &paint, Transform::from_scale(k, k), None)
         }))
    };
    let k = height / src_h;
    let mut pm = Pixmap::new((src_w * k).ceil().max(1.0) as u32, height.ceil().max(1.0) as u32)
        .ok_or("icon has zero size")?;
    draw(&mut pm, k);
    Ok(pm)
}

fn rgb(hex: &str) -> PyResult<Rgb> {
    Rgb::parse(hex).map_err(PyValueError::new_err)
}

/// Render the scene to `path`, drawing a batch of frames at a time in parallel.
fn render(scene: &impl Frames, path: &std::path::Path, bitrate_kbps: u32, progress: bool)
          -> Result<(), String> {
    let s = scene;
    let (w, h, fps) = (s.width(), s.height(), s.fps());
    let total = s.total_frames();
    let mut enc = video::Encoder::spawn(path, w, h, fps, bitrate_kbps)?;
    let batch = (rayon::current_num_threads() * 2).max(4);
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    for start in (0..total).step_by(batch) {
        let end = (start + batch).min(total);
        let frames: Vec<Vec<u8>> = (start..end)
            .into_par_iter()
            .map_init(|| Canvas::new(w, h), |canvas, f| {
                s.draw(f, canvas);
                canvas.pixmap.data().to_vec()
            })
            .collect();
        for frame in &frames {
            enc.write(frame)?;
        }
        if progress {
            eprint!("\r  rendering {name}: frame {end}/{total}");
            let _ = std::io::stderr().flush();
        }
    }
    if progress {
        eprintln!();
    }
    enc.finish()
}

/// Render an equity race to ``path`` (``.mp4`` or ``.gif``; needs ffmpeg).
///
/// Called by ``animate.equity_race``, which prepares every argument; see its
/// docstring for what each one means. ``x`` is bar time in days since the
/// epoch and ``growth`` is ``(bars, curves)`` rebased to 1. Fonts are
/// ``(path, face_index)`` pairs. ``icon`` is an optional SVG or PNG drawn at
/// the left of the title block.
#[pyfunction]
#[pyo3(signature = (path, x, growth, labels, colors, primary, title, subtitle, caption, log,
                    seconds, hold, fps, width, height, dpi, ink, ink_muted, grid, surface,
                    font_regular, font_bold, fill_alpha, label_size, icon=None,
                    bitrate_kbps=4500, progress=true, fees=None, drawdown=true, widths=None))]
#[allow(clippy::too_many_arguments)]
fn equity_race(py: Python<'_>, path: PathBuf, x: PyReadonlyArray1<'_, f64>,
               growth: PyReadonlyArray2<'_, f64>, labels: Vec<String>, colors: Vec<String>,
               primary: usize, title: String, subtitle: String, caption: String, log: bool,
               seconds: f64, hold: f64, fps: u32, width: u32, height: u32, dpi: f32,
               ink: &str, ink_muted: &str, grid: &str, surface: &str,
               font_regular: (String, u32), font_bold: (String, u32), fill_alpha: f32,
               label_size: f32, icon: Option<PathBuf>, bitrate_kbps: u32, progress: bool,
               fees: Option<PyReadonlyArray2<'_, f64>>, drawdown: bool,
               widths: Option<Vec<(f32, f32)>>) -> PyResult<PathBuf> {
    let g = growth.as_array();
    if fees.as_ref().is_some_and(|f| f.as_array().dim() != g.dim()) {
        return Err(PyValueError::new_err("fees needs the same shape as growth"));
    }
    if g.dim().1 != labels.len() {
        return Err(PyValueError::new_err("growth needs one column per label"));
    }
    if primary >= labels.len() {
        return Err(PyValueError::new_err("primary is out of range"));
    }
    let widths = widths.unwrap_or_else(|| (0..labels.len())
        .map(|i| if i == primary { (2.4, 1.4) } else { (1.6, 1.0) }).collect());
    if widths.len() != labels.len() {
        return Err(PyValueError::new_err("widths needs one pair per label"));
    }
    let icon_px = race::icon_height(!subtitle.is_empty()) * dpi / 72.0;
    let icon = icon.map(|p| load_icon(&p, icon_px)).transpose().map_err(PyValueError::new_err)?;
    let spec = Spec {
        x: x.as_array().to_vec(),
        growth: g.iter().copied().collect(),
        colors: colors.iter().map(|c| rgb(c)).collect::<PyResult<_>>()?,
        fees: fees.map(|f| f.as_array().iter().copied().collect()),
        labels, primary, widths, drawdown, title, subtitle, caption, log, width, height, dpi,
        theme: Theme { ink: rgb(ink)?, ink_muted: rgb(ink_muted)?, grid: rgb(grid)?,
                       surface: rgb(surface)? },
        fill_alpha, label_size,
        draw_frames: (seconds * fps as f64) as usize,
        hold_frames: (hold * fps as f64) as usize,
        fps,
        icon,
    };
    let load = |(p, i): &(String, u32)| text::Face::load(p, *i).map_err(PyValueError::new_err);
    let scene = Scene::new(spec, load(&font_regular)?, load(&font_bold)?)
        .map_err(PyValueError::new_err)?;
    py.detach(|| render(&scene, &path, bitrate_kbps, progress))
        .map_err(PyRuntimeError::new_err)?;
    Ok(path)
}

/// Render capital-vs-risk bars morphing through ``stages`` to ``path``.
///
/// Called by ``animate.risk_bars``, which prepares every argument; see its
/// docstring. ``weights`` is ``(stages, assets)`` and ``cov`` ``(assets, assets)``.
#[pyfunction]
#[pyo3(signature = (path, names, weights, cov, assets, colors, title, subtitle, captions,
                    move_s, hold, fps, width, height, dpi, ink, ink_muted, grid, surface,
                    font_regular, font_bold, icon=None, bitrate_kbps=4500, progress=true))]
#[allow(clippy::too_many_arguments)]
fn risk_bars(py: Python<'_>, path: PathBuf, names: Vec<String>,
             weights: PyReadonlyArray2<'_, f64>, cov: PyReadonlyArray2<'_, f64>,
             assets: Vec<String>, colors: Vec<String>, title: String, subtitle: String,
             captions: Vec<String>, move_s: f64, hold: f64, fps: u32, width: u32, height: u32,
             dpi: f32, ink: &str, ink_muted: &str, grid: &str, surface: &str,
             font_regular: (String, u32), font_bold: (String, u32), icon: Option<PathBuf>,
             bitrate_kbps: u32, progress: bool) -> PyResult<PathBuf> {
    let icon_px = race::icon_height(!subtitle.is_empty()) * dpi / 72.0;
    let icon = icon.map(|p| load_icon(&p, icon_px)).transpose().map_err(PyValueError::new_err)?;
    let spec = bars::Spec {
        weights: weights.as_array().iter().copied().collect(),
        cov: cov.as_array().iter().copied().collect(),
        colors: colors.iter().map(|c| rgb(c)).collect::<PyResult<_>>()?,
        names, assets, title, subtitle, captions, width, height, dpi, fps, icon,
        move_frames: (move_s * fps as f64) as usize,
        hold_frames: (hold * fps as f64) as usize,
        theme: Theme { ink: rgb(ink)?, ink_muted: rgb(ink_muted)?, grid: rgb(grid)?,
                       surface: rgb(surface)? },
    };
    let load = |(p, i): &(String, u32)| text::Face::load(p, *i).map_err(PyValueError::new_err);
    let scene = bars::Scene::new(spec, load(&font_regular)?, load(&font_bold)?)
        .map_err(PyValueError::new_err)?;
    py.detach(|| render(&scene, &path, bitrate_kbps, progress))
        .map_err(PyRuntimeError::new_err)?;
    Ok(path)
}

/// Encode one ``(height, width, 4)`` RGBA frame held for ``frames`` frames.
///
/// Used for stills drawn once (``animate.allocation_still``), so they come out
/// encoded exactly like the other renders and can be joined with ``concat``.
#[pyfunction]
#[pyo3(signature = (path, rgba, fps, frames, bitrate_kbps=4500))]
fn hold_frame(py: Python<'_>, path: PathBuf, rgba: PyReadonlyArray3<'_, u8>, fps: u32,
              frames: usize, bitrate_kbps: u32) -> PyResult<PathBuf> {
    let a = rgba.as_array();
    let (h, w, ch) = a.dim();
    if ch != 4 {
        return Err(PyValueError::new_err("rgba must be (height, width, 4)"));
    }
    let data: Vec<u8> = a.iter().copied().collect();
    py.detach(|| -> Result<(), String> {
        let mut enc = video::Encoder::spawn(&path, w as u32, h as u32, fps, bitrate_kbps)?;
        for _ in 0..frames.max(1) {
            enc.write(&data)?;
        }
        enc.finish()
    }).map_err(PyRuntimeError::new_err)?;
    Ok(path)
}

#[pymodule]
fn algo_animate_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(equity_race, m)?)?;
    m.add_function(wrap_pyfunction!(risk_bars, m)?)?;
    m.add_function(wrap_pyfunction!(hold_frame, m)?)?;
    Ok(())
}
