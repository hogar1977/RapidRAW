//! Quality and timing metrics for the panorama prototype.
//!
//! These are the metrics the Python v86 prototype recorded in
//! `out/v86/<tag>/params.json` and `run.log`, reimplemented in Rust so lean
//! numbers are directly comparable to the `boundary_step`, `blinds_flat`,
//! `woodline`, `ghost_band`, `shadow_median`, `pano_mean_lum` and
//! `sharpness_lapvar` values of the reference implementation.
//!
//! Two of them are seam-local and are therefore measured while the composite is
//! live (`ghost_band`, plus the seam boundary that the tone-step metrics need);
//! the rest are measured on the final cropped panorama, exactly as v86 did.

use rayon::prelude::*;
use std::collections::BTreeMap;

/// Per-seam seam-visibility and ghosting measurements.
#[derive(Default, Clone)]
pub struct SeamMetrics {
    /// normalized gradient-magnitude disagreement in the band around the cut
    pub ghost_band: f64,
    /// fraction of the co-valid overlap the incoming frame keeps; ~1.0 means
    /// the frame was swallowed and contributed no unique strip
    pub keep_a: f64,
    pub cut_px: usize,
    pub both_px: usize,
    pub window: (usize, usize),
    /// canvas-space pixels on the cut, kept for the tone-step metrics
    pub boundary: Vec<(u32, u32)>,
}

/// Tone-step metrics, measured on the final cropped panorama.
#[derive(Default, Clone)]
pub struct StepMetrics {
    pub n: usize,
    pub median: f64,
    pub p90: f64,
}

#[derive(Default, Clone)]
pub struct PanoMetrics {
    pub seams: Vec<SeamMetrics>,
    pub boundary_step: StepMetrics,
    pub blinds_flat: StepMetrics,
    pub woodline: StepMetrics,
    pub ghost_band: f64,
    pub lapvar: f64,
    pub mean_lum: f64,
    pub median_lum: f64,
    pub swallowed: usize,
    pub width: u32,
    pub height: u32,
    pub levels: usize,
    pub window_px: usize,
    pub timing: BTreeMap<String, f64>,
    pub extra: BTreeMap<String, f64>,
}

fn median_of(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn percentile_of(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let i = (((v.len() - 1) as f64) * q).round() as usize;
    v[i.min(v.len() - 1)]
}

fn summarize(mut v: Vec<f64>) -> StepMetrics {
    let n = v.len();
    let median = median_of(&mut v);
    let p90 = percentile_of(&mut v, 0.90);
    StepMetrics { n, median, p90 }
}

/// 3x3 Sobel gradient magnitude, matching cv2.Sobel(ksize=3) squared.
fn sobel_sq(lum: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y + 1 >= h {
            return;
        }
        for x in 1..w - 1 {
            let at = |dx: isize, dy: isize| lum[((y as isize + dy) * w as isize + (x as isize + dx)) as usize];
            let gx = -at(-1, -1) - 2.0 * at(-1, 0) - at(-1, 1) + at(1, -1) + 2.0 * at(1, 0) + at(1, 1);
            let gy = -at(-1, -1) - 2.0 * at(0, -1) - at(1, -1) + at(-1, 1) + 2.0 * at(0, 1) + at(1, 1);
            row[x] = gx * gx + gy * gy;
        }
    });
    out
}

/// Sobel evaluated on a stride-subsampled set of *positions*, packed to
/// stride x stride. The operator itself stays at one pixel: sampling the
/// operator's spacing instead would turn this into a coarser-scale measurement
/// and make the number incomparable with the reference values.
fn sobel_strided(lum: &[f32], w: usize, h: usize, stride: usize) -> Vec<f32> {
    let sw = (w + stride - 1) / stride;
    let sh = (h + stride - 1) / stride;
    let mut out = vec![0f32; sw * sh];
    out.par_chunks_mut(sw).enumerate().for_each(|(oy, row)| {
        let y = oy * stride;
        if y == 0 || y + 1 >= h {
            return;
        }
        for ox in 0..sw {
            let x = ox * stride;
            if x == 0 || x + 1 >= w {
                continue;
            }
            let at = |dx: isize, dy: isize| lum[((y as isize + dy) * w as isize + (x as isize + dx)) as usize];
            let gx = -at(-1, -1) - 2.0 * at(-1, 0) - at(-1, 1) + at(1, -1) + 2.0 * at(1, 0) + at(1, 1);
            let gy = -at(-1, -1) - 2.0 * at(0, -1) - at(1, -1) + at(-1, 1) + 2.0 * at(0, 1) + at(1, 1);
            row[ox] = gx * gx + gy * gy;
        }
    });
    out
}

/// Boundary of the A/B cut inside the co-valid overlap, matching v86's
/// `mask_boundary`: threshold the mask, mark every pixel whose 4-neighbourhood
/// differs, and intersect with the co-valid region.
#[allow(clippy::too_many_arguments)]
fn mask_boundary(
    hard: &[f32],
    both: &[bool],
    w: usize,
    h: usize,
    ox: usize,
    oy: usize,
    canvas_w: usize,
) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let m: Vec<bool> = hard.iter().map(|v| *v >= 0.5).collect();
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            if !both[row + x] {
                continue;
            }
            let here = m[row + x];
            let dx = x > 0 && here != m[row + x - 1];
            let dy = y > 0 && here != m[row - w + x];
            if dx || dy {
                out.push(((ox + x) as u32, (oy + y) as u32));
            }
        }
    }
    let _ = canvas_w;
    out
}

/// Grow a boolean mask by `radius` in x and then in y - two separable passes,
/// so a 32 px dilation stays linear instead of quadratic.
fn dilate_separable(mark: &[bool], band: &mut [bool], w: usize, h: usize, radius: usize) {
    let mut horiz = vec![false; w * h];
    horiz.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &mark[y * w..(y + 1) * w];
        for x in 0..w {
            if !src[x] {
                continue;
            }
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(w);
            row[lo..hi].fill(true);
        }
    });
    band.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut any = false;
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(h);
            for yy in lo..hi {
                if horiz[yy * w + x] {
                    any = true;
                    break;
                }
            }
            row[x] = any;
        }
    });
}

/// Seam metrics for one overlap: `ghost_band` and the cut boundary.
///
/// Definitions follow v86 exactly: the boundary is dilated by 32 px with a 3x3
/// structuring element, intersected with the co-valid region, and the metric is
/// mean|Sobel(A) - Sobel(B)| / mean((Sobel(A) + Sobel(B)) / 2).
#[allow(clippy::too_many_arguments)]
pub fn seam_metrics(
    lum_a: &[f32],
    lum_b: &[f32],
    av: &[bool],
    bv: &[bool],
    hard: &[f32],
    w: usize,
    h: usize,
    ox: usize,
    oy: usize,
) -> SeamMetrics {
    let mut m = SeamMetrics { window: (w, h), ..Default::default() };
    let both: Vec<bool> = av.iter().zip(bv.iter()).map(|(a, b)| *a && *b).collect();
    m.both_px = both.iter().filter(|v| **v).count();
    if m.both_px == 0 {
        return m;
    }
    m.keep_a = both
        .iter()
        .zip(hard.iter())
        .filter(|(b, _)| **b)
        .map(|(_, k)| *k as f64)
        .sum::<f64>()
        / m.both_px as f64;

    m.boundary = mask_boundary(hard, &both, w, h, ox, oy, 0);
    m.cut_px = m.boundary.len();

    let mut mark = vec![false; w * h];
    for &(x, y) in &m.boundary {
        let xi = x as usize - ox;
        let yi = y as usize - oy;
        mark[yi * w + xi] = true;
    }
    let mut band = vec![false; w * h];
    dilate_separable(&mark, &mut band, w, h, 32);
    for (b, ok) in band.iter_mut().zip(both.iter()) {
        *b &= *ok;
    }
    // Sobel over the whole window costs ~2 x 9 taps per pixel and dominated the
    // measurement budget. The metric is a ratio of means, so an evenly strided
    // subsample of the band is an unbiased estimate of it.
    let band_px = band.iter().filter(|v| **v).count();
    let stride = (band_px / 1_500_000).max(1);
    let ga = sobel_strided(lum_a, w, h, stride);
    let gb = sobel_strided(lum_b, w, h, stride);
    let (sw, sh) = ((w + stride - 1) / stride, (h + stride - 1) / stride);
    let mut num = 0.0f64;
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for y in 0..sh {
        for x in 0..sw {
            let si = y * w + x;
            let di = y * sw + x;
            if !band[si] || ga[di] <= 0.0 || gb[di] <= 0.0 {
                continue;
            }
            let a = (ga[di] as f64).sqrt();
            let b = (gb[di] as f64).sqrt();
            num += (a - b).abs();
            sum += (a + b) * 0.5;
            n += 1;
        }
    }
    m.ghost_band = if n == 0 {
        f64::NAN
    } else {
        (num / n as f64) / (sum / n as f64).max(1e-6)
    };
    m
}

/// Tone-step metrics on the final cropped panorama, matching v86's
/// `boundary_step` / `blinds_flat` / `woodline`.
pub fn step_metrics(
    rgb: &[f32],
    w: u32,
    h: u32,
    crop_x: i64,
    crop_y: i64,
    boundaries: &[Vec<(u32, u32)>],
) -> (StepMetrics, StepMetrics, StepMetrics) {
    let (cw, ch) = (w as usize, h as usize);
    let mut lum = vec![0f32; cw * ch];
    lum.par_chunks_mut(cw).enumerate().for_each(|(y, row)| {
        for x in 0..cw {
            let p = &rgb[(y * cw + x) * 3..(y * cw + x) * 3 + 3];
            row[x] = (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).max(0.0);
        }
    });
    // v86's row flatness probe: mean |L(x+6) - L(x)| across the whole row.
    const BAND: usize = 6;
    let mut flatness = vec![0f32; ch];
    flatness.par_iter_mut().enumerate().for_each(|(y, f)| {
        let row = &lum[y * cw..(y + 1) * cw];
        if row.len() <= BAND {
            *f = 1.0;
            return;
        }
        let acc: f32 = row[BAND..].iter().zip(row.iter()).map(|(a, b)| (a - b).abs()).sum();
        *f = acc / (row.len() - BAND) as f32;
    });
    let y3 = ch * 2 / 3;

    let step_at = |y: usize, x: usize| -> Option<f64> {
        if x <= 12 || x + 12 >= cw {
            return None;
        }
        let row = &lum[y * cw..(y + 1) * cw];
        let l = median_of_slice(&row[x - 10..x - 2]);
        let r = median_of_slice(&row[x + 2..x + 10]);
        Some((l - r).abs() / ((l + r) * 0.5).max(1e-3))
    };

    let mut all: Vec<f64> = Vec::new();
    let mut flat: Vec<f64> = Vec::new();
    let mut wood: Vec<f64> = Vec::new();
    for bnd in boundaries {
        for (i, &(x, y)) in bnd.iter().enumerate() {
            let xx = x as i64 - crop_x;
            let yy = y as i64 - crop_y;
            if xx < 0 || yy < 0 || xx >= cw as i64 || yy >= ch as i64 {
                continue;
            }
            let (xx, yy) = (xx as usize, yy as usize);
            // v86 samples every 7th boundary pixel for the overall step and
            // every 3rd for the flat and woodline subsets.
            if i % 7 == 0 {
                if let Some(v) = step_at(yy, xx) {
                    all.push(v);
                }
            }
            if i % 3 == 0 {
                if flatness[yy] <= 0.004 {
                    if let Some(v) = step_at(yy, xx) {
                        flat.push(v);
                    }
                }
                if yy >= y3 {
                    if let Some(v) = step_at(yy, xx) {
                        wood.push(v);
                    }
                }
            }
        }
    }
    (summarize(all), summarize(flat), summarize(wood))
}

fn median_of_slice(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut b = v.to_vec();
    b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b[b.len() / 2] as f64
}

fn to_srgb8(x: f32) -> u8 {
    let v = x.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Whole-panorama metrics on the final cropped linear composite.
///
/// `lapvar` reproduces v86's `sharpness_lapvar`: the variance of a 4-neighbour
/// Laplacian over the OpenCV BGR->GRAY of the 8-bit sRGB encoding, so the
/// numbers are on the same scale as `params.json`.
pub fn pano_metrics(rgb: &[f32], w: u32, h: u32) -> (f64, f64, f64) {
    let n = (w as usize) * (h as usize);
    let mut lum: Vec<f64> = Vec::with_capacity(n);
    for p in rgb.chunks_exact(3).take(n) {
        lum.push((0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).max(0.0) as f64);
    }
    let mean = lum.iter().sum::<f64>() / n.max(1) as f64;
    let med = median_of(&mut lum.clone());
    if n < 9 {
        return (0.0, mean, med);
    }
    // sRGB-encode on a stride: the result is a variance, so a regular stride is
    // an unbiased sample of it and this is 1/stride^2 of the work.
    let stride = (n / 8_000_000).max(1);
    let wu = w as usize;
    let hu = h as usize;
    let sw = wu / stride;
    let sh = hu / stride;
    if sw < 3 || sh < 3 {
        return (0.0, mean, med);
    }
    // Positions are strided but every Laplacian still uses its true one-pixel
    // neighbours, so the value stays on the same scale as the reference.
    let gray = |x: usize, y: usize| -> f64 {
        let i = (y * wu + x) * 3;
        let r = to_srgb8(rgb[i]) as f64;
        let gg = to_srgb8(rgb[i + 1]) as f64;
        let b = to_srgb8(rgb[i + 2]) as f64;
        // OpenCV RGB2GRAY coefficients.
        0.299 * r + 0.587 * gg + 0.114 * b
    };
    let mut acc = 0.0f64;
    let mut cnt = 0usize;
    for oy in 0..sh {
        let y = oy * stride;
        if y == 0 || y + 1 >= hu {
            continue;
        }
        for ox in 0..sw {
            let x = ox * stride;
            if x == 0 || x + 1 >= wu {
                continue;
            }
            let c = gray(x, y);
            let l = gray(x - 1, y) + gray(x + 1, y) + gray(x, y - 1) + gray(x, y + 1) - 4.0 * c;
            acc += l * l;
            cnt += 1;
        }
    }
    (if cnt == 0 { 0.0 } else { acc / cnt as f64 }, mean, med)
}

/// Print a metrics block in the prototype's log style.
pub fn report(m: &PanoMetrics, label: &str) {
    use rapidraw_lib::panorama_utils::v86::trace;
    let pct = |s: &StepMetrics| format!("n={} median_step={:.2}% p90={:.2}%", s.n, s.median * 100.0, s.p90 * 100.0);
    trace::line(&format!("boundary_step {label}: {}", pct(&m.boundary_step)));
    trace::line(&format!("blinds_flat  {label}: {}", pct(&m.blinds_flat)));
    trace::line(&format!("woodline    {label}: {}", pct(&m.woodline)));
    trace::line(&format!(
        "ghost_band   {label}: mean={:.3} keepA={:.2} cut_px={} swallowed={}",
        m.ghost_band,
        if m.seams.is_empty() { 0.0 } else { m.seams.iter().map(|s| s.keep_a).sum::<f64>() / m.seams.len() as f64 },
        m.seams.iter().map(|s| s.cut_px).sum::<usize>(),
        m.swallowed
    ));
    trace::line(&format!(
        "sharpness_lapvar {label}: {}",
        m.lapvar
    ));
    trace::line(&format!("pano_mean_lum {} shadow_median {} canvas {}x{} levels={} window_px={}", m.mean_lum, m.median_lum, m.width, m.height, m.levels, m.window_px));
    for (k, v) in &m.extra {
        trace::line(&format!("{k} {label}: {v:.5}"));
    }
    let mut stages: Vec<String> = Vec::new();
    for (k, v) in &m.timing {
        stages.push(format!("{k}={:.2}s", v));
    }
    if !stages.is_empty() {
        trace::line(&format!("time  {label} {}", stages.join(" ")));
    }
}
