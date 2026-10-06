use super::align::{self, Preview};
use super::exec::pick_exec;
use rapidraw_lib::panorama_stitching::{self, exposure_gains};
use crate::common::metrics::{seam_metrics, SeamMetrics};
use rapidraw_lib::panorama_utils::v86::local_align::align_incoming;
use crate::common::StageTimer;
use rapidraw_lib::panorama_utils::v86::blend::{blend_levels, blur_mask, pyr_blend};
use rapidraw_lib::panorama_utils::v86::local_align::{self, RecordedShift};
use rapidraw_lib::panorama_utils::v86::photo;
use rapidraw_lib::panorama_utils::v86::seam::graphcut_mask;
use rapidraw_lib::panorama_utils::v86::stitch::{self, InputFrame};
use rapidraw_lib::panorama_utils::v86::trace;
use rapidraw_lib::panorama_utils::v86::warp::{self, Placed};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const BAND_PX: f64 = 192.0;

pub fn render(paths: &[String], preview: &Preview, tmp: &Path, tiff_path: &Path, view_path: &Path) -> Result<(), String> {
    let exec = pick_exec();
    let dw = preview.frames[0].width;
    let dh = preview.frames[0].height;
    let mut stored = Vec::new();
    let mut evs = Vec::new();
    let t_develop = std::time::Instant::now();
    for &index in &preview.indices {
        let frame = exec.develop_raw(&paths[index])?;
        evs.push(frame.exposure_ev);
        let file = tmp.join(format!("{:02}-{}.f32", stored.len(), frame.name));
        write_f32(&file, frame.width, frame.height, &frame.rgb)?;
        let small = align::shrink(&frame.rgb, frame.width, frame.height, dw, dh);
        trace::line(&format!("developed {} {}x{} -> {dw}x{dh}", frame.name, frame.width, frame.height));
        stored.push(Stored { file, width: frame.width, height: frame.height, name: frame.name, small });
    }
    trace::line(&format!("time  develop={:.2}s", t_develop.elapsed().as_secs_f64()));
    let gains = exposure_gains(&evs);
    let mut linear_frames = Vec::new();
    for (item, gain) in stored.iter_mut().zip(gains.iter()) {
        if (*gain - 1.0).abs() > 1e-4 {
            for p in &mut item.small {
                *p *= *gain;
            }
        }
        linear_frames.push(InputFrame { name: item.name.clone(), width: dw, height: dh, rgb: std::mem::take(&mut item.small) });
    }
    let raw_w = stored[0].width;
    let raw_h = stored[0].height;
    let sx = raw_w as f64 / preview.jpeg_w as f64;
    let sy = raw_h as f64 / preview.jpeg_h as f64;
    trace::line(&format!("scale raw={raw_w}x{raw_h} jpeg={}x{} x={sx:.4} y={sy:.4}", preview.jpeg_w, preview.jpeg_h));
    if (sx - sy).abs() > 0.03 * sx.max(sy) {
        return Err(format!("embedded jpeg does not line up with the developed frame ({sx:.3} vs {sy:.3})"));
    }
    let focal_full = rapidraw_lib::panorama_utils::v86::const_::focal_px_from_35eq(preview.jpeg_w, preview.jpeg_h, preview.focal35) * sx;
    let kept: Vec<usize> = (0..linear_frames.len()).collect();
    local_align::begin_shift_log();
    let t_lin = std::time::Instant::now();
    let linear = stitch::compose(&linear_frames, &preview.rots, &kept, preview.lens, preview.focal, preview.focal35, preview.result.used, false, &|message| {
        trace::line(message);
    })?;
    let _developed_shifts = local_align::end_shift_log();
    trace::line(&format!("time  linear_photometry={:.2}s", t_lin.elapsed().as_secs_f64()));
    drop(linear_frames);
    let use_linear = linear.kept == preview.result.kept;
    if !use_linear {
        trace::line("linear order differs; keeping the preview photometry");
    }
    trace::line("save seam preview");
    trace::line("save cut fullres");
    trace::line(&format!("linear gains={}", linear.photo.gains.len()));
    let seam_w = preview.result.width;
    let seam_h = preview.result.height;
    let shifts = &preview.shifts;
    let crop_x = preview.result.crop_x;
    let crop_y = preview.result.crop_y;
    let crop_w = preview.result.crop_w;
    let crop_h = preview.result.crop_h;
    let (geom, _) = warp::canvas_geom(&preview.rots, focal_full, raw_w, raw_h, &preview.lens, preview.result.used);
    trace::line(&format!("canvas {} {}x{}", preview.result.used.as_str(), geom.width, geom.height));
    let fw = geom.width as usize;
    let fh = geom.height as usize;
    let mut acc = vec![0f32; fw * fh * 3];
    let mut written = vec![0u8; fw * fh];
    let mut owners = vec![u16::MAX; fw * fh];
    let levels = blend_levels(geom.width, geom.height, 6, 8);
    let pad = 8usize << levels.min(8);
    trace::line(&format!("composite levels={levels} pad={pad}"));
    // v22: validated full-resolution local alignment, on by default. The env
    // switch exists so the effect can be A/B'd on identical inputs.
    let refine = std::env::var("LEAN_REFINE").map(|v| v != "0").unwrap_or(true);
    trace::line(&format!("local alignment refine={refine}"));
    local_align::begin_shift_log();
    let mut timer = StageTimer::new();
    let mut metrics = crate::common::metrics::PanoMetrics { levels, window_px: pad, width: geom.width, height: geom.height, ..Default::default() };
    let mut gained = 0usize;
    for (bi, &frame_i) in preview.result.kept.iter().enumerate() {
        trace::gate()?;
        let item = &stored[frame_i];
        let mut rgb = read_f32(&item.file)?;
        let gain = gains[frame_i];
        if (gain - 1.0).abs() > 1e-4 {
            for p in &mut rgb {
                *p *= gain;
            }
            gained += 1;
        }
        let shift = if bi == 0 { None } else { shifts.get(bi - 1).and_then(|entry| entry.as_ref()) };
        let nudge = |x: f64, y: f64| sample_nudge(shift, x, y, seam_w, seam_h, geom.width, geom.height);
        let mut placed = timer.stage("warp", || exec.warp(&mut rgb, item.width, item.height, &preview.rots[frame_i], focal_full, &preview.lens, &geom, &nudge));
        drop(rgb);
        let _ = std::fs::remove_file(&item.file);
        timer.stage("photometry", || photo::apply_photometry(&mut placed, if use_linear { &linear.photo } else { &preview.result.photo }, bi));
        let id = bi as u16;
        if bi == 0 {
            paste(&mut acc, &mut written, &mut owners, &placed, id, fw, fh);
        } else {
            let m = timer.stage("blend", || composite_overlap(&mut acc, &mut written, &mut owners, &placed, id, fw, fh, levels, pad, refine))?;
            trace::line(&format!("seam {id} window={}x{} keepA={:.2}", m.window.0, m.window.1, m.keep_a));
            metrics.seams.push(m);
        }
        trace::line(&format!("warped {} {}", item.name, if shift.is_some() { "nudged" } else { "straight" }));
    }
    let refined = local_align::end_shift_log();
    let moved: Vec<f64> = refined
        .iter()
        .flatten()
        .map(|s| {
            let n = s.dx.len().max(1) as f64;
            (s.dx.iter().map(|v| v.abs() as f64).sum::<f64>() / n
                + s.dy.iter().map(|v| v.abs() as f64).sum::<f64>() / n)
                / 2.0
        })
        .collect();
    trace::line(&format!(
        "refined shifts seams={} mean_abs_shift={:.3}px max_abs_shift={:.3}px",
        moved.len(),
        if moved.is_empty() { 0.0 } else { moved.iter().sum::<f64>() / moved.len() as f64 },
        moved.iter().cloned().fold(0.0f64, f64::max)
    ));
    trace::line(&format!("exposure gains applied {gained}"));
    let cropped = crop(&acc, geom.width, geom.height, crop_x, crop_y, crop_w, crop_h);
    trace::line(&format!("save encode {}x{}", cropped.1, cropped.2));
    let bounds: Vec<Vec<(u32, u32)>> = metrics.seams.iter().map(|s| s.boundary.clone()).collect();
    let (lap, mean_lum, med_lum) = timer.stage("metrics", || crate::common::pano_metrics(&cropped.0, cropped.1, cropped.2));
    metrics.lapvar = lap;
    metrics.mean_lum = mean_lum;
    metrics.median_lum = med_lum;
    let bstep = timer.stage("step_metrics", || crate::common::metrics::step_metrics(&cropped.0, cropped.1, cropped.2, crop_x.round() as i64, crop_y.round() as i64, &bounds));
    metrics.boundary_step = bstep.0;
    metrics.blinds_flat = bstep.1;
    metrics.woodline = bstep.2;
    let ghosts: Vec<f64> = metrics.seams.iter().map(|s| s.ghost_band).filter(|g| g.is_finite()).collect();
    metrics.ghost_band = if ghosts.is_empty() { 0.0 } else { ghosts.iter().sum::<f64>() / ghosts.len() as f64 };
    metrics.swallowed = metrics.seams.iter().filter(|s| s.keep_a > 0.95).count();
    metrics.width = cropped.1;
    metrics.height = cropped.2;
    timer.stage("encode_tiff", || panorama_stitching::write_panorama_tiff(&cropped.0, cropped.1, cropped.2, tiff_path))?;
    write_view(&cropped.0, cropped.1, cropped.2, view_path)?;
    metrics.timing = timer.into_map();
    crate::common::report(&metrics, "final");
    Ok(())
}

struct Stored {
    file: PathBuf,
    width: u32,
    height: u32,
    name: String,
    small: Vec<f32>,
}

// Paste the first frame: everything it covers is unwritten, so it owns the lot.
fn paste(acc: &mut [f32], written: &mut [u8], owners: &mut [u16], p: &Placed, id: u16, fw: usize, fh: usize) {
    let y0 = p.y0.max(0) as usize;
    let y1 = p.y1().clamp(0, fh as i32) as usize;
    if y1 <= y0 || p.w == 0 {
        return;
    }
    let rows = y1 - y0;
    acc[y0 * fw * 3..y1 * fw * 3]
        .par_chunks_mut(fw * 3)
        .zip(written[y0 * fw..y1 * fw].par_chunks_mut(fw))
        .zip(owners[y0 * fw..y1 * fw].par_chunks_mut(fw))
        .enumerate()
        .for_each(|(row, ((dest, flags), owned))| {
            if row >= rows {
                return;
            }
            let py = (y0 + row) as i32 - p.y0;
            if py < 0 || py >= p.h as i32 {
                return;
            }
            for x in 0..fw {
                let px = x as i32 - p.x0;
                if px < 0 || px >= p.w as i32 {
                    continue;
                }
                let s = py as usize * p.w + px as usize;
                if !p.valid[s] {
                    continue;
                }
                dest[x * 3] = p.img[s * 3];
                dest[x * 3 + 1] = p.img[s * 3 + 1];
                dest[x * 3 + 2] = p.img[s * 3 + 2];
                flags[x] = 1;
                owned[x] = id;
            }
        });
}

// Rec.709 luma plane for the graph-cut data term.
fn lum_plane(rgb: &[f32]) -> Vec<f32> {
    rgb.chunks_exact(3).map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).collect()
}

// v21 change: replace the 192 px ribbon / 2-level Laplacian blend with the
// app's padded-overlap window, sigma-2 mask feather and 6..8 level
// Burt-Adelson blend, restricted to the true co-valid overlap.
fn composite_overlap(
    acc: &mut [f32],
    written: &mut [u8],
    owners: &mut [u16],
    p: &Placed,
    id: u16,
    fw: usize,
    fh: usize,
    levels: usize,
    pad: usize,
    refine: bool,
) -> Result<SeamMetrics, String> {
    let x0 = (p.x0.max(0) as usize).saturating_sub(pad);
    let y0 = (p.y0.max(0) as usize).saturating_sub(pad);
    let x1 = (p.x1().max(0) as usize + pad).min(fw);
    let y1 = (p.y1().max(0) as usize + pad).min(fh);
    if x1 <= x0 || y1 <= y0 {
        return Ok(SeamMetrics::default());
    }
    let ww = x1 - x0;
    let hh = y1 - y0;
    // Extract both sides of the overlap into dense window arrays.
    let mut aw = vec![0f32; ww * hh * 3];
    let mut bw = vec![0f32; ww * hh * 3];
    let mut av = vec![false; ww * hh];
    let mut bv = vec![false; ww * hh];
    for y in 0..hh {
        let cy = y0 + y;
        let row = y * ww;
        let crow = cy * fw;
        let srow = cy.saturating_sub(p.y0 as usize) * p.w;
        for x in 0..ww {
            let cx = x0 + x;
            let di = row + x;
            let ai = crow + cx;
            if written[ai] != 0 {
                aw[di * 3] = acc[ai * 3];
                aw[di * 3 + 1] = acc[ai * 3 + 1];
                aw[di * 3 + 2] = acc[ai * 3 + 2];
                av[di] = true;
            }
            if cx >= p.x0 as usize && cx < p.x1() as usize && cy >= p.y0 as usize && cy < p.y1() as usize {
                let si = srow + (cx - p.x0 as usize);
                if p.valid[si] {
                    bw[di * 3] = p.img[si * 3];
                    bw[di * 3 + 1] = p.img[si * 3 + 1];
                    bw[di * 3 + 2] = p.img[si * 3 + 2];
                    bv[di] = true;
                }
            }
        }
    }
    let both = av.iter().zip(bv.iter()).filter(|(a, b)| **a && **b).count();
    if both == 0 {
        // No overlap: plain paste.
        for y in 0..hh {
            for x in 0..ww {
                let di = y * ww + x;
                if !bv[di] {
                    continue;
                }
                let ai = (y0 + y) * fw + x0 + x;
                acc[ai * 3] = bw[di * 3];
                acc[ai * 3 + 1] = bw[di * 3 + 1];
                acc[ai * 3 + 2] = bw[di * 3 + 2];
                written[ai] = 1;
                owners[ai] = id;
            }
        }
        return Ok(SeamMetrics { window: (ww, hh), ..Default::default() });
    }
    trace::gate()?;
    let lum_a = lum_plane(&aw);
    let mut lum_b = lum_plane(&bw);
    trace::gate()?;
    // v22 change: validated per-tile local alignment on the full-resolution
    // overlap, in the app's order - fit the shift field, refit the residual
    // field, keep it only if textured disagreement improves, then re-luma.
    let (bw, bv) = if refine {
        let (shifted, svalid) = align_incoming(&lum_a, &lum_b, &av, bv, bw, ww, hh);
        local_align::commit_shift(x0 as i32, y0 as i32);
        lum_b = lum_plane(&shifted);
        (shifted, svalid)
    } else {
        (bw, bv)
    };
    trace::gate()?;
    let hard = graphcut_mask(&lum_a, &lum_b, &av, &bv, ww, hh);
    trace::gate()?;
    let soft = blur_mask(&hard, ww, hh);
    trace::line("mask blurred");
    // Restrict the expensive pyramid to the co-valid overlap, so the feather
    // is not dominated by wide one-sided regions of the padded window.
    let (sx0, sy0, sx1, sy1) = blend_rect(&av, &bv, ww, hh, pad);
    let cw = sx1 - sx0;
    let ch = sy1 - sy0;
    let a_sub = take_rect_rgb(&aw, ww, sx0, sy0, sx1, sy1);
    let b_sub = take_rect_rgb(&bw, ww, sx0, sy0, sx1, sy1);
    let m_sub = take_rect_f32(&soft, ww, sx0, sy0, sx1, sy1);
    let av_sub = take_rect_bool(&av, ww, sx0, sy0, sx1, sy1);
    let bv_sub = take_rect_bool(&bv, ww, sx0, sy0, sx1, sy1);
    // `bw` stays live: pixels outside the blend rectangle fall back to a hard
    // copy of the incoming frame.
    drop(aw);
    let sm = seam_metrics(&lum_a, &lum_b, &av, &bv, &hard, ww, hh, x0, y0);
    trace::gate()?;
    let blended = pyr_blend(&a_sub, &b_sub, &m_sub, &av_sub, &bv_sub, cw, ch, levels);
    drop(a_sub);
    drop(b_sub);
    // Metrics come from the untouched full window so they are independent of
    // the blend rectangle.

    drop(m_sub);
    drop(av_sub);
    drop(bv_sub);
    for y in 0..hh {
        let inside_row = y >= sy0 && y < sy1;
        let row = y * ww;
        for x in 0..ww {
            let di = row + x;
            if !(av[di] || bv[di]) {
                continue;
            }
            let (r, g, b) = if inside_row && x >= sx0 && x < sx1 {
                let ci = (y - sy0) * cw + (x - sx0);
                (blended[ci * 3], blended[ci * 3 + 1], blended[ci * 3 + 2])
            } else if bv[di] {
                (bw[di * 3], bw[di * 3 + 1], bw[di * 3 + 2])
            } else {
                continue;
            };
            let ai = (y0 + y) * fw + x0 + x;
            acc[ai * 3] = r;
            acc[ai * 3 + 1] = g;
            acc[ai * 3 + 2] = b;
            written[ai] = 1;
            if bv[di] && (!av[di] || hard[di] < 0.5) {
                owners[ai] = id;
            }
        }
    }
    trace::line("blended");
    Ok(sm)
}

// Bounding box of the co-valid overlap, padded, but never so large that the
// per-seam pyramid dominates the canvas.
fn blend_rect(av: &[bool], bv: &[bool], w: usize, h: usize, pad: usize) -> (usize, usize, usize, usize) {
    let mut x0 = w;
    let mut y0 = h;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            if av[row + x] && bv[row + x] {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return (0, 0, w, h);
    }
    let sx0 = x0.saturating_sub(pad);
    let sy0 = y0.saturating_sub(pad);
    let sx1 = (x1 + pad).min(w);
    let sy1 = (y1 + pad).min(h);
    if (sx1 - sx0) * (sy1 - sy0) * 10 > w * h * 9 {
        (0, 0, w, h)
    } else {
        (sx0, sy0, sx1, sy1)
    }
}

fn take_rect_rgb(src: &[f32], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<f32> {
    let cw = x1 - x0;
    let mut out = vec![0f32; cw * (y1 - y0) * 3];
    for y in y0..y1 {
        let s = (y * w + x0) * 3;
        let d = ((y - y0) * cw) * 3;
        let n = cw * 3;
        out[d..d + n].copy_from_slice(&src[s..s + n]);
    }
    out
}

fn take_rect_f32(src: &[f32], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<f32> {
    let cw = x1 - x0;
    let mut out = vec![0f32; cw * (y1 - y0)];
    for y in y0..y1 {
        let s = y * w + x0;
        let d = (y - y0) * cw;
        out[d..d + cw].copy_from_slice(&src[s..s + cw]);
    }
    out
}

fn take_rect_bool(src: &[bool], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<bool> {
    let cw = x1 - x0;
    let mut out = vec![false; cw * (y1 - y0)];
    for y in y0..y1 {
        let s = y * w + x0;
        let d = (y - y0) * cw;
        out[d..d + cw].copy_from_slice(&src[s..s + cw]);
    }
    out
}

fn sample_nudge(shift: Option<&RecordedShift>, x: f64, y: f64, preview_w: u32, preview_h: u32, full_w: u32, full_h: u32) -> (f64, f64) {
    let Some(shift) = shift else { return (0.0, 0.0) };
    if shift.w < 2 || shift.h < 2 {
        return (0.0, 0.0);
    }
    let px = x * preview_w as f64 / full_w as f64;
    let py = y * preview_h as f64 / full_h as f64;
    let lx = px - shift.x as f64;
    let ly = py - shift.y as f64;
    if lx < 0.0 || ly < 0.0 || lx >= shift.w as f64 - 1.0 || ly >= shift.h as f64 - 1.0 {
        return (0.0, 0.0);
    }
    let x0 = lx.floor() as usize;
    let y0 = ly.floor() as usize;
    let tx = (lx - x0 as f64) as f32;
    let ty = (ly - y0 as f64) as f32;
    let mix = |src: &[f32]| {
        let at = |yy: usize, xx: usize| src[yy * shift.w + xx];
        let a = at(y0, x0) * (1.0 - tx) + at(y0, x0 + 1) * tx;
        let b = at(y0 + 1, x0) * (1.0 - tx) + at(y0 + 1, x0 + 1) * tx;
        (a * (1.0 - ty) + b * ty) as f64
    };
    (mix(&shift.dx) * full_w as f64 / preview_w as f64, mix(&shift.dy) * full_h as f64 / preview_h as f64)
}

fn winner_at(winners: &[u16], preview_w: u32, preview_h: u32, full_w: u32, full_h: u32, x: i32, y: i32) -> u16 {
    let px = (x as f64 * preview_w as f64 / full_w as f64).round() as i32;
    let py = (y as f64 * preview_h as f64 / full_h as f64).round() as i32;
    if px < 0 || py < 0 || px >= preview_w as i32 || py >= preview_h as i32 {
        return u16::MAX;
    }
    winners[py as usize * preview_w as usize + px as usize]
}

fn crop(rgb: &[f32], w: u32, h: u32, x: f64, y: f64, cw: f64, ch: f64) -> (Vec<f32>, u32, u32) {
    let x0 = (x * w as f64).round().clamp(0.0, w as f64) as u32;
    let y0 = (y * h as f64).round().clamp(0.0, h as f64) as u32;
    let x1 = ((x + cw) * w as f64).round().clamp(x0 as f64 + 1.0, w as f64) as u32;
    let y1 = ((y + ch) * h as f64).round().clamp(y0 as f64 + 1.0, h as f64) as u32;
    let out_w = x1 - x0;
    let out_h = y1 - y0;
    let mut out = vec![0f32; out_w as usize * out_h as usize * 3];
    for row in 0..out_h {
        let s = (((y0 + row) * w + x0) * 3) as usize;
        let d = (row * out_w * 3) as usize;
        let n = out_w as usize * 3;
        out[d..d + n].copy_from_slice(&rgb[s..s + n]);
    }
    (out, out_w, out_h)
}

pub fn write_view(rgb: &[f32], w: u32, h: u32, path: &Path) -> Result<(), String> {
    let (dw, dh) = align::fit_long_side(w, h, 1600);
    let small = align::shrink(rgb, w, h, dw, dh);
    let mut img: image::RgbImage = image::ImageBuffer::new(dw, dh);
    for (i, pixel) in img.pixels_mut().enumerate() {
        *pixel = image::Rgb([to_srgb(small[i * 3]), to_srgb(small[i * 3 + 1]), to_srgb(small[i * 3 + 2])]);
    }
    img.save_with_format(path, image::ImageFormat::Jpeg).map_err(|e| format!("Failed to save the view: {e}"))
}

fn to_srgb(x: f32) -> u8 {
    let x = x.max(0.0);
    let y = if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (y.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn write_f32(path: &Path, w: u32, h: u32, rgb: &[f32]) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| e.to_string())?;
    file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&h.to_le_bytes()).map_err(|e| e.to_string())?;
    let mut bytes = Vec::with_capacity(rgb.len() * 4);
    for value in rgb {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all(&bytes).map_err(|e| e.to_string())
}

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    let w = u32::from_le_bytes(header[0..4].try_into().unwrap());
    let h = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    let count = (w as usize) * (h as usize) * 3;
    if bytes.len() != count * 4 {
        return Err(format!("short frame {}", path.display()));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
