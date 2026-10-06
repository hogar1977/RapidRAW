use super::align::{self, Preview};
use super::exec::pick_exec;
use rapidraw_lib::panorama_stitching::{self, exposure_gains};
use crate::common::metrics::{seam_metrics, SeamMetrics};
use rapidraw_lib::panorama_utils::v86::local_align::align_incoming;
use crate::common::StageTimer;
use rapidraw_lib::panorama_utils::v86::blend::{blend_levels, blur_mask, pyr_blend};
use rapidraw_lib::panorama_utils::v86::local_align::{self, RecordedShift};
use rapidraw_lib::panorama_utils::v86::photo::{self, luma};
use rapidraw_lib::panorama_utils::v86::seam::graphcut_mask;
use rapidraw_lib::panorama_utils::v86::stitch::{self, InputFrame};
use rapidraw_lib::panorama_utils::v86::trace;
use rapidraw_lib::panorama_utils::v86::warp::{self, Placed};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const BAND_PX: f64 = 192.0;

thread_local! {
    /// Per-seam blend sub-stage totals, reported once per set. This is what
    /// showed that local alignment, the seam solve and the pyramid blend are
    /// the three things worth attacking.
    static BLEND_BREAKDOWN: std::cell::RefCell<std::collections::BTreeMap<String, f64>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

pub fn report_blend_breakdown() {
    BLEND_BREAKDOWN.with(|b| {
        for (k, v) in b.borrow().iter() {
            trace::line(&format!("blend_stage {k}={v:.2}s"));
        }
    });
}

pub fn render(paths: &[String], preview: &Preview, tmp: &Path, tiff_path: &Path, view_path: &Path) -> Result<(), String> {
    let exec = pick_exec();
    // Snapshot key, so LEAN_SNAP=DSCF0566 only fires for that set.
    let stem = tiff_path
        .file_name()
        .map(|n| n.to_string_lossy().split('_').next().unwrap_or("").to_string())
        .unwrap_or_default();
    let dw = preview.frames[0].width;
    let dh = preview.frames[0].height;
    // v28: develop the frames concurrently. Each develop is single-threaded
    // inside the app, so a 10-frame set spent 21 s doing them one after another
    // on a 24-thread box. Concurrency is capped because every in-flight frame
    // costs its own full-resolution buffer.
    let concurrency = std::env::var("LEAN_DEVELOP_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1);
    trace::line(&format!("develop jobs={concurrency}"));
    let t_develop = std::time::Instant::now();
    let developed: Result<Vec<Stored>, String> = paths
        .par_iter()
        .with_max_len(concurrency)
        .map(|path| {
            let frame = exec.develop_raw(path)?;
            let file = tmp.join(format!("{:02}-{}.f32", 0, frame.name));
            write_f32(&file, frame.width, frame.height, &frame.rgb)?;
            let small = align::shrink(&frame.rgb, frame.width, frame.height, dw, dh);
            Ok(Stored { file, width: frame.width, height: frame.height, name: frame.name, small, exposure_ev: frame.exposure_ev })
        })
        .collect();
    let mut stored: Vec<Stored> = developed?;
    // Name the temp files after their position now that ordering is known.
    for (i, item) in stored.iter_mut().enumerate() {
        let want = tmp.join(format!("{:02}-{}.f32", i, item.name));
        if item.file != want {
            let _ = std::fs::rename(&item.file, &want);
            item.file = want;
        }
        trace::line(&format!("developed {} {}x{} -> {dw}x{dh}", item.name, item.width, item.height));
    }
    let evs: Vec<Option<f64>> = stored.iter().map(|s| s.exposure_ev).collect();
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
    // v28: this was tried with the second compose switched off, on the theory
    // that re-solving photometry on a second shrink of frames the preview had
    // already solved was redundant. It is not: with the preview-only model the
    // shadow median fell from 0.0361 to 0.0308 and ghost_band rose from 0.659
    // to 0.675, so the 7.7 s is worth paying. LEAN_SECOND_COMPOSE=0 A/Bs it.
    let mut linear: Option<stitch::StitchResult> = None;
    let mut developed_shifts: Option<Vec<Option<local_align::RecordedShift>>> = None;
    let use_second = std::env::var("LEAN_SECOND_COMPOSE").map(|v| v != "0").unwrap_or(true);
    if use_second {
        let t_lin = std::time::Instant::now();
        let composed = stitch::compose(&linear_frames, &preview.rots, &kept, preview.lens, preview.focal, preview.focal35, preview.result.used, false, &|message| {
            trace::line(message);
        })?;
        developed_shifts = Some(local_align::end_shift_log());
        trace::line(&format!("time  linear_photometry={:.2}s", t_lin.elapsed().as_secs_f64()));
        drop(linear_frames);
        if composed.kept != preview.result.kept {
            trace::line("linear order differs; keeping the preview photometry");
        } else {
            linear = Some(composed);
        }
    };
    let photo_model: &photo::Photo = match &linear {
        Some(l) => &l.photo,
        None => &preview.result.photo,
    };
    let use_linear = linear.is_some();
    trace::line(&format!("photometry source={}", if use_linear { "second compose" } else { "preview" }));
    trace::line("save seam preview");
    trace::line("save cut fullres");
    trace::line(&format!("photometry gains={}", photo_model.gains.len()));
    {
        let ph = &photo_model;
        let gl: Vec<String> = ph.gains.iter().map(|g| format!("{g:.4}")).collect();
        trace::line(&format!("photo pedestal={:.6} field_norm={:.6} gains=[{}]", ph.pedestal, ph.coef.iter().map(|c| c.abs()).sum::<f64>(), gl.join(" ")));
    }
    // v31: take the seam geometry and the shift field from the developed
    // compose when it agrees with the preview, the way v10 did.
    //
    // The nudge is sampled in canvas pixels, so the field has to be expressed
    // on the same canvas it is applied to. `developed_shifts` is solved on
    // frames that already carry the exposure gains and the real developed
    // pixels; `preview.shifts` is solved on the 1920 px JPEG preview. Taking
    // the winners and crop from one and the shifts from the other mixes two
    // different geometries, and a mismatch there displaces a frame by a few
    // pixels exactly where the overlap is narrowest. v10 selected all five from
    // the same source and 0200's roof ridge came out clean.
    let same_canvas = linear.as_ref().is_some_and(|l| l.width == preview.result.width && l.height == preview.result.height);
    let use_developed = use_linear && same_canvas && developed_shifts.is_some();
    trace::line(if use_developed { "save seam developed" } else { "save seam preview" });
    let (seam_w, seam_h, shifts, crop_x, crop_y, crop_w, crop_h) = if use_developed {
        let l = linear.as_ref().unwrap();
        (
            l.width,
            l.height,
            developed_shifts.as_ref().unwrap(),
            l.crop_x,
            l.crop_y,
            l.crop_w,
            l.crop_h,
        )
    } else {
        (
            preview.result.width,
            preview.result.height,
            &preview.shifts,
            preview.result.crop_x,
            preview.result.crop_y,
            preview.result.crop_w,
            preview.result.crop_h,
        )
    };
    let (geom, _) = warp::canvas_geom(&preview.rots, focal_full, raw_w, raw_h, &preview.lens, preview.result.used);
    trace::line(&format!("canvas {} {}x{}", preview.result.used.as_str(), geom.width, geom.height));
    let fw = geom.width as usize;
    let fh = geom.height as usize;
    let mut acc = vec![0f32; fw * fh * 3];
    let mut written = vec![0u8; fw * fh];
    let mut owners = vec![u16::MAX; fw * fh];
    // v31: back to solving the shift field at full resolution.
    //
    // v30 halved the overlap to make this a quarter of the work, and that is
    // where the v30 fix's one regression came from. A tile is 128 px, so at
    // 1/2 scale each one covers 256 full-resolution pixels: a near-field
    // subject standing against a distant background - a tower against a city -
    // gets averaged together with what is behind it, and the field then ramps
    // across its silhouette instead of stepping at it. On DSCF0566 that put a
    // 37 px step in the left edge of the dominant tower, which the seam solve
    // cannot hide because the step is a geometry error, not a tone difference.
    //
    // It is not an acceptance failure either. The field is honestly better by
    // the disagreement measure (0.215 -> 0.140 on strong edges), because the
    // background really does dominate the average; the subject is simply
    // outvoted. Only a full-resolution solve can express a parallax
    // discontinuity, so the field is solved at 1/1.
    //
    // Measured on DSCF0566: 1/1 costs 34.4 s of local align against 26.1 s at
    // 1/2 and removes the step (tower edge 46.8 px, matching the no-refine
    // reference). The gate stays so 1/2 remains available for A/B.
    let halves = std::env::var("LEAN_ALIGN_HALVES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| [1usize, 2, 4].contains(v))
        .unwrap_or(1);
    trace::line(&format!("local align resolution 1/{halves}"));
    // Snapshot window origin in canvas coordinates. Every canvas-level snapshot
    // below is sliced at this origin so it lines up pixel-for-pixel with the
    // window-level snapshots taken inside composite_overlap.
    let snap_win_org = snap_win(fw, fh).map(|(a, b, _, _)| (a, b)).unwrap_or((0, 0));
    let snap_canvas_box = (0usize, 0usize, fw, fh);
    trace::line(&format!("snapbox canvas_origin=({},{})", snap_win_org.0, snap_win_org.1));
    let levels = blend_levels(geom.width, geom.height, 6, 8);
    let pad = 8usize << levels.min(8);
    trace::line(&format!("composite levels={levels} pad={pad}"));
    // v22: validated full-resolution local alignment, on by default. The env
    // switch exists so the effect can be A/B'd on identical inputs.
    // v31: off by default. The warp nudge already carries the whole
    // correction, solved on the preview at 1920 px and scaled to the canvas,
    // and re-solving it a second time on the full-resolution overlap applies
    // the same displacement twice. The nudge is a smooth low-order field fitted
    // across the overlap; once the canvas is warped with it, the residual is
    // small and mostly flat, and a second fit on that residual is dominated by
    // whatever fills the most area - usually distant background. A near-field
    // subject is a minority of the overlap, so the second pass racks it
    // sideways to satisfy the majority and the seam solver cannot hide the
    // result, because a displaced edge is geometry, not tone. DSCF0200's roof
    // ridge and DSCF0566's tower are that failure, and they alternate with the
    // resolution the second pass runs at, which is what made it look like two
    // unrelated bugs. v10 has no second pass and both sets are clean on it.
    // LEAN_REFINE=1 restores it for A/B.
    let refine = std::env::var("LEAN_REFINE").map(|v| v != "0").unwrap_or(false);
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
        // v38 diagnostic: LEAN_NUDGE=0 forces every frame to warp with a zero
        // displacement field, leaving rotations as the only geometric input.
        // Comparing the canvas-vs-incoming displacement ramp with the nudge on
        // and off separates "the nudge introduces the error" from "the ramp is
        // already present before the nudge" (i.e. it comes from the solver).
        // Default is 1, which reproduces v37 exactly.
        let nudge_on = std::env::var("LEAN_NUDGE").map(|v| v != "0").unwrap_or(true);
        let shift = if bi == 0 || !nudge_on {
            None
        } else {
            shifts.get(bi - 1).and_then(|entry| entry.as_ref())
        };
        if !nudge_on {
            trace::line(&format!("nudge DISABLED for {}", item.name));
        }
        let nudge = |x: f64, y: f64| sample_nudge(shift, x, y, seam_w, seam_h, geom.width, geom.height);
        let mut placed = timer.stage("warp", || exec.warp(&mut rgb, item.width, item.height, &preview.rots[frame_i], focal_full, &preview.lens, &geom, &nudge));
        flush_ntrace();
        drop(rgb);
        let _ = std::fs::remove_file(&item.file);
        timer.stage("photometry", || photo::apply_photometry(&mut placed, &photo_model, bi));
        let id = bi as u16;
        if bi == 0 {
            paste(&mut acc, &mut written, &mut owners, &placed, id, fw, fh);
        } else {
            snap_canvas(&stem, &format!("s{id:02}_C2_canvas_before"), &acc, fw, fh, snap_win_org, snap_canvas_box);
            let m = timer.stage("blend", || composite_overlap(&mut acc, &mut written, &mut owners, &placed, id, fw, fh, levels, pad, refine, halves, &stem))?;
            trace::line(&format!("seam {id} window={}x{} keepA={:.2}", m.window.0, m.window.1, m.keep_a));
            metrics.seams.push(m);
        }
        trace::line(&format!("warped {} {}", item.name, if shift.is_some() { "nudged" } else { "straight" }));
        // v39 diagnostic: write the recorded shift field for this frame as a
        // PNG. Blue-positive/red-negative diverging map of dx, plus a green
        // channel carrying dy, so the shape of the fitted field and the shape
        // of its support are both visible without any arithmetic.
        if let Some(sh) = shift {
            trace::line(&format!("field {} origin=({},{}) size={}x{} scale={:.4}", item.name, sh.x, sh.y, sh.w, sh.h, seam_w as f64 / sh.w as f64));
            snap_field(&format!("{stem}/field_{}_{}", stem, item.name.trim_start_matches("DSCF").trim_end_matches(".RAF")), sh);
        }
        // Same window as the in-function snapshots: the frame as warped, and
        // the canvas immediately after this frame is composited.
        snap_placed_window(&stem, &format!("s{id:02}_C_after_photometry"), &placed, fw, fh, snap_win_org);
        snap_canvas(&stem, &format!("s{id:02}_D_canvas_after"), &acc, fw, fh, snap_win_org, snap_canvas_box);
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
    // Dump the seam boundary so the artifacts can be located exactly instead of
    // guessed at from a downscaled overview.
    if let Some(dir) = tiff_path.parent() {
        let dump = dir.join("seams.txt");
        if let Ok(mut f) = std::fs::File::create(&dump) {
            for (bi, seam) in metrics.seams.iter().enumerate() {
                for &(x, y) in &seam.boundary {
                    let _ = writeln!(f, "{bi} {x} {y}");
                }
            }
            trace::line(&format!("seam dump {}", dump.display()));
        }
    }
    report_blend_breakdown();
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
    exposure_ev: Option<f64>,
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
    halves: usize,
    snap_stem: &str,
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
    let mut t = StageTimer::new();
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
    // v38 diagnostic: the incoming side IS the placed frame copied into the
    // window, so S3 already is the post-warp state. What is missing is the
    // same frame with the nudge disabled, which LEAN_NUDGE=0 provides for a
    // whole-run comparison. Nothing extra to dump here.
    trace::line(&format!("snapwin seam={id} canvas=({x0},{y0}) size={ww}x{hh}"));
    snap_valid(snap_stem, &format!("s{id:02}_S1_avbv"), &av, &bv, ww, hh, (x0, y0));
    snap_rgb(snap_stem, &format!("s{id:02}_S2_aw_canvas"), &aw, ww, hh, (x0, y0));
    snap_rgb(snap_stem, &format!("s{id:02}_S3_bw_incoming"), &bw, ww, hh, (x0, y0));
    trace::gate()?;
    let _ = t.stage("extract", || ());
    let lum_a = lum_plane(&aw);
    let mut lum_b = lum_plane(&bw);
    trace::gate()?;
    // v22 change: validated per-tile local alignment on the full-resolution
    // overlap, in the app's order - fit the shift field, refit the residual
    // field, keep it only if textured disagreement improves, then re-luma.
    // v30: the shift field is smooth (a low-order fit over 128 px tiles), so
    // solving it on a half-resolution overlap and applying the doubled field at
    // full resolution costs a quarter of the work. The seam solve and the
    // pyramid blend still run at full resolution, where the detail lives.
    let (mut bw, bv) = if refine && halves > 1 {
        match t.stage("local_align_half", || align_field_half(&aw, &lum_a, &lum_b, &bw, &av, &bv, ww, hh, halves)) {
            Some(field) => {
                apply_shift_field(&mut bw, &field, ww, hh);
                (bw, bv)
            }
            // "no staged field" means the coarse pass found nothing worth
            // correcting (`local copied sparse` / `overlap` / `no gain`).
            None => (bw, bv),
        }
    } else if refine {
        let original = bw.clone();
        let (shifted, svalid) = t.stage("local_align", || align_incoming(&lum_a, &lum_b, &av, bv, bw, ww, hh));
        local_align::commit_shift(x0 as i32, y0 as i32);
        let gated = t.stage("tile_gate", || mask_field_tiles(&lum_a, &original, &original, &shifted, &av, &svalid, ww, hh));
        (gated, svalid)
    } else {
        (bw, bv)
    };
    // v26: fit the gain field once, after local alignment, so the correction is
    // estimated on the pixels that will actually be blended - and never applied
    // twice. v25 fitted before and again after, which compounded to a 9% level
    // error on some seams.
    {
        let f1 = fit_gain_fields(&aw, &bw, &av, &bv, ww, hh);
        let peak1 = apply_gain_field(&mut bw, &f1, ww, hh);
        trace::line(&format!(
            "field n={} const=[{:.4} {:.4} {:.4}] peak={:.4} gain={:.4}{}",
            f1[0].n,
            f1[0].coef[0].exp(),
            f1[1].coef[0].exp(),
            f1[2].coef[0].exp(),
            peak1,
            last_gain(),
            if f1[0].n == 0 { " (rejected)" } else { "" }
        ));
    }
    snap_rgb(snap_stem, &format!("s{id:02}_S4_bw_after_gain"), &bw, ww, hh, (x0, y0));
    lum_b = lum_plane(&bw);
    trace::gate()?;
    let hard = t.stage("graphcut", || graphcut_mask(&lum_a, &lum_b, &av, &bv, ww, hh));
    snap_mask(snap_stem, &format!("s{id:02}_S5_mask_hard"), &hard, ww, hh, (x0, y0), None);
    trace::gate()?;
    let soft = t.stage("mask_blur", || blur_mask(&hard, ww, hh));
    snap_mask(snap_stem, &format!("s{id:02}_S6_mask_soft"), &soft, ww, hh, (x0, y0), None);
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
    trace::line(&format!("snapblendrect seam={id} canvas=({},{})..({},{})", x0 + sx0, y0 + sy0, x0 + sx1, y0 + sy1));
    snap_mask(snap_stem, &format!("s{id:02}_S7_mask_soft_rect"), &soft, ww, hh, (x0, y0), Some((sx0, sy0, sx1, sy1)));
    let blended = t.stage("pyr_blend", || pyr_blend(&a_sub, &b_sub, &m_sub, &av_sub, &bv_sub, cw, ch, levels));
    // The pyramid result only covers the rectangle, so it is placed into a
    // full-window-sized buffer to keep every snapshot on one comparable frame.
    {
        let mut full = vec![0f32; ww * hh * 3];
        for y in 0..ch {
            let sy = sy0 + y;
            if sy >= hh {
                break;
            }
            for x in 0..cw {
                let sx = sx0 + x;
                if sx >= ww {
                    break;
                }
                let si = (y * cw + x) * 3;
                let di = (sy * ww + sx) * 3;
                full[di] = blended[si];
                full[di + 1] = blended[si + 1];
                full[di + 2] = blended[si + 2];
            }
        }
        snap_rgb(snap_stem, &format!("s{id:02}_S8_pyrblend"), &full, ww, hh, (x0, y0));
    }
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
    for (k, v) in t.into_map() {
        BLEND_BREAKDOWN.with(|b| {
            let mut m = b.borrow_mut();
            *m.entry(k).or_insert(0.0) += v;
        });
    }
    Ok(sm)
}

// v25 change: per-seam, per-channel *gain field* instead of a scalar gain.
//
// v24 measured a single median ratio per channel and it barely moved the worst
// seams (0566 seam 5: +2.51% red before, +2.56% after). The residual is
// spatially varying - one scalar cannot remove a gradient - so this fits a
// low-order 2D polynomial in log space to log(B/A) over the quiet co-valid
// pixels, per channel, and applies it to the incoming frame.
//
// The fit is ridge-regularised towards the constant term, the field is clamped
// to +-3% so it can only ever make a small correction, and it is skipped
// entirely when there is too little evidence. Basis: 1, u, v, u^2, uv, v^2
// over the overlap in normalised coordinates.
const FIELD_TERMS: usize = 6;
const FIELD_CLAMP: f64 = 0.02;
const RIDGE_FRACTION: f64 = 0.02;
/// A correction is only kept if it reduces robust disagreement by this much.
const MIN_GAIN: f64 = 0.05;

#[derive(Clone, Copy)]
struct GainField {
    coef: [f64; FIELD_TERMS],
    n: usize,
}

impl GainField {
    fn identity() -> Self {
        Self { coef: [0.0; FIELD_TERMS], n: 0 }
    }

    fn at(&self, u: f64, v: f64) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let (uu, vv) = (u.clamp(-1.5, 1.5), v.clamp(-1.5, 1.5));
        let c0 = self.coef[0].clamp(-FIELD_CLAMP, FIELD_CLAMP);
        // Higher-order terms correct the *shape* of the difference; they are
        // never allowed to move the overall level on their own.
        let shape = self.coef[1] * uu
            + self.coef[2] * vv
            + self.coef[3] * uu * uu
            + self.coef[4] * uu * vv
            + self.coef[5] * vv * vv;
        let lim = (FIELD_CLAMP - c0.abs()).max(0.0);
        (c0 + shape.clamp(-lim, lim)).clamp(-FIELD_CLAMP, FIELD_CLAMP)
    }
}

fn basis(u: f64, v: f64) -> [f64; FIELD_TERMS] {
    [1.0, u, v, u * u, u * v, v * v]
}

/// Symmetric 6x6 solve with a ridge on every term, scaled to the sample count
/// so the regularisation does not depend on how many pixels voted.
fn solve_sym(mut a: [[f64; FIELD_TERMS]; FIELD_TERMS], mut b: [f64; FIELD_TERMS], n: usize) -> Option<[f64; FIELD_TERMS]> {
    let trace: f64 = (0..FIELD_TERMS).map(|i| a[i][i]).sum();
    let lambda = (trace / FIELD_TERMS as f64) * RIDGE_FRACTION;
    for i in 0..FIELD_TERMS {
        a[i][i] += lambda;
    }
    for k in 0..FIELD_TERMS {
        let mut pivot = k;
        for i in k + 1..FIELD_TERMS {
            if a[i][k].abs() > a[pivot][k].abs() {
                pivot = i;
            }
        }
        if a[pivot][k].abs() < 1e-9 {
            return None;
        }
        if pivot != k {
            a.swap(pivot, k);
            b.swap(pivot, k);
        }
        let d = a[k][k];
        for j in k..FIELD_TERMS {
            a[k][j] /= d;
        }
        b[k] /= d;
        for i in 0..FIELD_TERMS {
            if i == k {
                continue;
            }
            let f = a[i][k];
            if f == 0.0 {
                continue;
            }
            for j in k..FIELD_TERMS {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    Some(b)
}

fn fit_gain_fields(aw: &[f32], bw: &[f32], av: &[bool], bv: &[bool], w: usize, h: usize) -> [GainField; 3] {
    let mut out = [GainField::identity(); 3];
    // Enough samples for 6 terms with margin.
    const NEED: usize = 400;
    let step = 6usize;
    let mut acc = [[[0f64; FIELD_TERMS]; FIELD_TERMS]; 3];
    let mut rhs = [[0f64; FIELD_TERMS]; 3];
    let mut n = 0usize;
    let mut y = 6usize;
    while y + 6 < h {
        let vcoord = (y as f64 / h.max(1) as f64) * 2.0 - 1.0;
        let mut x = 6usize;
        while x + 6 < w {
            let i = y * w + x;
            if av[i] && bv[i] {
                let mut flat = true;
                for (dx, dy) in [(-4isize, 0isize), (4, 0), (0, -4), (0, 4)] {
                    let j = ((y as isize + dy) * w as isize + x as isize + dx) as usize;
                    for side in [aw, bw] {
                        if (side[j * 3 + 1] - side[i * 3 + 1]).abs() > 0.012 {
                            flat = false;
                        }
                    }
                }
                if flat {
                    let ucoord = (x as f64 / w.max(1) as f64) * 2.0 - 1.0;
                    let basis = basis(ucoord, vcoord);
                    let mut usable = true;
                    for ch in 0..3 {
                        let (la, lb) = (aw[i * 3 + ch] as f64, bw[i * 3 + ch] as f64);
                        if la <= 2e-3 || lb <= 2e-3 {
                            usable = false;
                            break;
                        }
                        let r = lb / la;
                        if !r.is_finite() || (r - 1.0).abs() > 0.35 {
                            usable = false;
                            break;
                        }
                    }
                    if usable {
                        n += 1;
                        for ch in 0..3 {
                            let r = bw[i * 3 + ch] as f64 / aw[i * 3 + ch] as f64;
                            let target = r.ln();
                            for p in 0..FIELD_TERMS {
                                rhs[ch][p] += basis[p] * target;
                                for q in 0..FIELD_TERMS {
                                    acc[ch][p][q] += basis[p] * basis[q];
                                }
                            }
                        }
                    }
                }
            }
            x += step;
        }
        y += step;
    }
    if n < NEED {
        return out;
    }
    for ch in 0..3 {
        if let Some(mut c) = solve_sym(acc[ch], rhs[ch], n) {
            c[0] = c[0].clamp(-FIELD_CLAMP, FIELD_CLAMP);
            if (c[0] - 0.0f64).abs() < 1e-4 && c[1..].iter().all(|v| v.abs() < 1e-4) {
                continue;
            }
            out[ch] = GainField { coef: c, n };
        }
    }
    // v26 guard: only keep the field if it actually reduces robust disagreement
    // on the very pixels it was fitted to. v25 applied an unvalidated fit and
    // made the worst seam worse, because "quiet" pixels are not always the same
    // physical surface.
    if out.iter().any(|f| f.n > 0) {
        let (before, after) = robust_disagreement(aw, bw, av, bv, &out, w, h);
        let gain = if before > 1e-9 { (before - after) / before } else { 0.0 };
        LAST_GAIN.with(|g| *g.borrow_mut() = gain);
        if gain < MIN_GAIN {
            out = [GainField::identity(); 3];
        }
    }
    out
}

thread_local! {
    /// Reported so the log can show whether the guard accepted the fit.
    static LAST_GAIN: std::cell::RefCell<f64> = const { std::cell::RefCell::new(0.0) };
}

fn last_gain() -> f64 {
    LAST_GAIN.with(|g| *g.borrow())
}

/// Median absolute relative difference between canvas and frame, before and
/// after applying `fields`.
fn robust_disagreement(
    aw: &[f32],
    bw: &[f32],
    av: &[bool],
    bv: &[bool],
    fields: &[GainField; 3],
    w: usize,
    h: usize,
) -> (f64, f64) {
    let mut before: Vec<f64> = Vec::new();
    let mut after: Vec<f64> = Vec::new();
    let step = 6usize;
    let mut y = 6usize;
    while y + 6 < h {
        let vv = (y as f64 / h.max(1) as f64) * 2.0 - 1.0;
        let mut x = 6usize;
        while x + 6 < w {
            let i = y * w + x;
            if av[i] && bv[i] {
                let mut flat = true;
                for (dx, dy) in [(-4isize, 0isize), (4, 0), (0, -4), (0, 4)] {
                    let j = ((y as isize + dy) * w as isize + x as isize + dx) as usize;
                    for side in [aw, bw] {
                        if (side[j * 3 + 1] - side[i * 3 + 1]).abs() > 0.012 {
                            flat = false;
                        }
                    }
                }
                if flat {
                    let uu = (x as f64 / w.max(1) as f64) * 2.0 - 1.0;
                    let mut b0 = 0.0;
                    let mut a1 = 0.0;
                    let mut ok = true;
                    for ch in 0..3 {
                        let la = aw[i * 3 + ch] as f64;
                        if la <= 2e-3 {
                            ok = false;
                            break;
                        }
                        b0 += ((bw[i * 3 + ch] as f64 - la) / la).abs();
                        a1 += ((bw[i * 3 + ch] as f64 * fields[ch].at(uu, vv).exp() - la) / la).abs();
                    }
                    if ok {
                        before.push(b0 / 3.0);
                        after.push(a1 / 3.0);
                    }
                }
            }
            x += step;
        }
        y += step;
    }
    if before.len() < 32 {
        return (0.0, 0.0);
    }
    before.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    after.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (before[before.len() / 2], after[after.len() / 2])
}

fn apply_gain_field(bw: &mut [f32], fields: &[GainField; 3], w: usize, h: usize) -> f64 {
    let active = fields.iter().filter(|f| f.n > 0).count();
    if active == 0 {
        return 0.0;
    }
    let peak: f64 = if active == 3 {
        let mut corners = [f64::MIN; 2];
        corners.par_iter_mut().enumerate().for_each(|(k, c)| {
            let u = if k == 0 { -1.0 } else { 1.0 };
            *c = 0.0;
            for v in [-1.0f64, 1.0] {
                *c = (*c).max(fields[0].at(u, v).abs()).max(fields[1].at(u, v).abs()).max(fields[2].at(u, v).abs());
            }
        });
        corners.iter().cloned().fold(0.0f64, f64::max)
    } else {
        0.0
    };
    bw.par_chunks_mut(3).enumerate().for_each(|(i, p)| {
        let (x, y) = (i % w, i / w);
        let u = (x as f64 / w.max(1) as f64) * 2.0 - 1.0;
        let v = (y as f64 / h.max(1) as f64) * 2.0 - 1.0;
        for ch in 0..3 {
            p[ch] *= fields[ch].at(u, v).exp() as f32;
        }
    });
    peak
}

/// Solve the shift field on a `1/halves` resolution overlap and return it in
/// full-resolution window coordinates.
fn align_field_half(
    aw: &[f32],
    lum_a: &[f32],
    lum_b: &[f32],
    bw: &[f32],
    av: &[bool],
    bv: &[bool],
    w: usize,
    h: usize,
    halves: usize,
) -> Option<Vec<(f32, f32)>> {
    let hw = w / halves;
    let hh = h / halves;
    if hw < 192 || hh < 192 {
        trace::line(&format!("align-half skip tiny {hw}x{hh}"));
        return None;
    }
    let half_rgb = shrink_rgb(bw, w, h, hw, hh);
    let half_a = shrink_luma(lum_a, w, h, hw, hh);
    let half_b = shrink_luma(lum_b, w, h, hw, hh);
    let half_av = shrink_bool(av, w, h, hw, hh);
    let half_bv = shrink_bool(bv, w, h, hw, hh);
    // align_incoming only *stages* the field; commit_shift is what moves it
    // into the log, so without this the log comes back empty.
    local_align::begin_shift_log();
    let _ = align_incoming(&half_a, &half_b, &half_av, half_bv, half_rgb, hw, hh);
    local_align::commit_shift(0, 0);
    let logged = local_align::end_shift_log();
    let Some(entry) = logged.into_iter().flatten().next() else {
        trace::line("align-half skip no staged field");
        return None;
    };
    if entry.w < 8 || entry.h < 8 {
        trace::line(&format!("align-half skip thin {}x{}", entry.w, entry.h));
        return None;
    }
    // The staged rectangle is the co-valid sub-rect, not the whole window, so
    // map through its own origin rather than assuming it starts at (0,0).
    //
    // v31: this upsample was nearest-neighbour with a truncating cast, which
    // turned one half-res sample into 2x2 blocks whose phase against the pixel
    // grid ratchets along the row (the first block covers 3 px, the rest 2).
    // A real shift gradient therefore became a staircase of full-pixel jumps,
    // and the jumps landed wherever the fitted field changed fastest - which
    // on a near-field subject against a distant background is its silhouette.
    // Bilinear keeps the field continuous, so any remaining step has to come
    // from the geometry rather than from the resampling.
    let scale = halves as f32;
    let (ex, ey, ew, eh) = (entry.x.max(0) as usize, entry.y.max(0) as usize, entry.w, entry.h);
    let mut field = vec![(0f32, 0f32); w * h];
    field.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            // Full-res window pixel -> fractional half-res window pixel,
            // relative to the staged rectangle's own origin.
            let fx = (x as f32 + 0.5) / scale - 0.5 - ex as f32;
            let fy = (y as f32 + 0.5) / scale - 0.5 - ey as f32;
            if fx < 0.0 || fy < 0.0 || fx >= ew as f32 - 1.0 || fy >= eh as f32 - 1.0 {
                continue;
            }
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            let i00 = y0 * ew + x0;
            let i10 = i00 + 1;
            let i01 = i00 + ew;
            let i11 = i01 + 1;
            let w00 = (1.0 - tx) * (1.0 - ty);
            let w10 = tx * (1.0 - ty);
            let w01 = (1.0 - tx) * ty;
            let w11 = tx * ty;
            let bx = entry.dx[i00] * w00 + entry.dx[i10] * w10 + entry.dx[i01] * w01 + entry.dx[i11] * w11;
            let by = entry.dy[i00] * w00 + entry.dy[i10] * w10 + entry.dy[i01] * w01 + entry.dy[i11] * w11;
            row[x] = (bx * scale, by * scale);
        }
    });
    Some(field)
}

/// Keeps, per tile, whichever of the two frames agrees better with what is
/// already on the canvas.
///
/// `align_incoming` accepts or rejects its whole shift field with one averaged
/// disagreement number, and that average is dominated by whatever fills most of
/// the overlap - usually distant background. A near-field subject is a small
/// minority of the area, so a field that drags the background into place while
/// racking the subject sideways still clears the global test, and the subject
/// keeps the error. DSCF0200's roof ridge and DSCF0566's tower are both this
/// failure, and they are complementary: whichever resolution the field is
/// solved at, it helps one and wrecks the other.
///
/// Deciding per tile instead of per seam makes the pass strictly conservative
/// - a tile is only moved when moving it demonstrably reduces disagreement on
/// its own edges - so the correction accumulates where it helps and is refused
/// where it would hurt. Nothing about the scene is assumed.
fn luma_at(rgb: &[f32], i: usize) -> f32 {
    0.2126 * rgb[i * 3] + 0.7152 * rgb[i * 3 + 1] + 0.0722 * rgb[i * 3 + 2]
}

fn mask_field_tiles(
    lum_a: &[f32],
    lum_b: &[f32],
    before: &[f32],
    after: &[f32],
    av: &[bool],
    bv: &[bool],
    w: usize,
    h: usize,
) -> Vec<f32> {
    const TILE: usize = 128;
    if w < TILE * 2 || h < TILE * 2 {
        return after.to_vec();
    }
    let lum_after = lum_plane(after);
    let nbx = (w + TILE - 1) / TILE;
    let nby = (h + TILE - 1) / TILE;
    let mut d0 = vec![0f64; nbx * nby];
    let mut d1 = vec![0f64; nbx * nby];
    let mut cnt = vec![0u32; nbx * nby];
    // Only judge tiles that carry edges. Without this the flat majority of a
    // wide overlap outnumbers the few tiles that hold the subject.
    let mut mag = vec![0f32; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let up = (y - 1) * w;
            let mid = y * w;
            let dn = (y + 1) * w;
            let gx = -lum_a[up + x - 1] + lum_a[up + x + 1] - 2.0 * lum_a[mid + x - 1] + 2.0 * lum_a[mid + x + 1] - lum_a[dn + x - 1]
                + lum_a[dn + x + 1];
            let gy = -lum_a[up + x - 1] - 2.0 * lum_a[up + x] - lum_a[up + x + 1] + lum_a[dn + x - 1] + 2.0 * lum_a[dn + x] + lum_a[dn + x + 1];
            mag[mid + x] = gx.abs() + gy.abs();
        }
    }
    let mut edges: Vec<f32> = mag
        .par_iter()
        .enumerate()
        .filter(|(i, _)| av[*i] && bv[*i])
        .map(|(_, g)| *g)
        .collect();
    if edges.len() < 1024 {
        return after.to_vec();
    }
    let p70 = edges.len() / 2 + edges.len() / 5;
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let thr = edges[p70.min(edges.len() - 1)];
    for y in 0..h {
        let by = (y / TILE) * nbx;
        let row = y * w;
        for x in 0..w {
            let i = row + x;
            if !(av[i] && bv[i] && mag[i] > thr) {
                continue;
            }
            let b = by + x / TILE;
            let lb0 = luma_at(before, i);
            let lb1 = lum_after[i];
            let ref_l = ((lum_a[i] + lb0) * 0.5).max(0.02);
            d0[b] += (lum_a[i] - lb0).abs() as f64 / ref_l as f64;
            d1[b] += (lum_a[i] - lb1).abs() as f64 / ref_l as f64;
            cnt[b] += 1;
        }
    }
    // keep[b] = 1 keeps the corrected frame, 0 keeps the original.
    let mut keep = vec![0f32; nbx * nby];
    let mut kept_tiles = 0usize;
    let mut judged_tiles = 0usize;
    for b in 0..nbx * nby {
        if cnt[b] < 256 {
            // Too little texture to judge. Leave the frame untouched rather
            // than guess; there is nothing here to align.
            keep[b] = 0.0;
            continue;
        }
        judged_tiles += 1;
        let (a0, a1) = (d0[b] / cnt[b] as f64, d1[b] / cnt[b] as f64);
        if a1 <= a0 * 0.99 || (a0 - a1) >= 0.004 {
            keep[b] = 1.0;
            kept_tiles += 1;
        }
    }
    if kept_tiles == 0 {
        trace::line(&format!("tile gate kept 0/{judged_tiles} tiles"));
        return before.to_vec();
    }
    trace::line(&format!("tile gate kept {kept_tiles}/{judged_tiles} tiles"));
    // Feather the tile decisions so a refused tile next to an accepted one does
    // not reintroduce a step along their border.
    let mut soft = vec![0f32; nbx * nby];
    for by in 0..nby {
        for bx in 0..nbx {
            let mut acc = 0f32;
            let mut n = 0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = by as i32 + dy;
                    let nx = bx as i32 + dx;
                    if ny < 0 || nx < 0 || ny >= nby as i32 || nx >= nbx as i32 {
                        continue;
                    }
                    let j = ny as usize * nbx + nx as usize;
                    let wgt = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                    acc += keep[j] * wgt;
                    n += wgt;
                }
            }
            soft[by * nbx + bx] = acc / n;
        }
    }
    let mut out = vec![0f32; w * h * 3];
    out.par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let by = (y / TILE) * nbx;
            let ty = ((y % TILE) as f32 + 0.5) / TILE as f32;
            for x in 0..w {
                let bx = x / TILE;
                let tx = ((x % TILE) as f32 + 0.5) / TILE as f32;
                let gx = soft[by + bx] * (1.0 - tx) + soft[by + (bx + 1).min(nbx - 1)] * tx;
                let gy = if by + nbx < soft.len() { soft[by + nbx + bx] } else { gx };
                let t = gx * (1.0 - ty) + gy * ty;
                let i = (y * w + x) * 3;
                row[x * 3] = before[i] * (1.0 - t) + after[i] * t;
                row[x * 3 + 1] = before[i + 1] * (1.0 - t) + after[i + 1] * t;
                row[x * 3 + 2] = before[i + 2] * (1.0 - t) + after[i + 2] * t;
            }
        });
    out
}

/// Bilinear resample of `bw` by a per-pixel shift field, in place.
fn apply_shift_field(bw: &mut [f32], field: &[(f32, f32)], w: usize, h: usize) {
    let src = bw.to_vec();
    let out = bw;
    out.par_chunks_mut(3).enumerate().for_each(|(i, p)| {
        let (x, y) = (i % w, i / w);
        let (dx, dy) = field[i];
        let fx = x as f32 + dx;
        let fy = y as f32 + dy;
        let (x0, y0) = (fx.floor().max(0.0) as usize, fy.floor().max(0.0) as usize);
        if x0 + 1 >= w || y0 + 1 >= h {
            p.copy_from_slice(&src[i * 3..i * 3 + 3]);
            return;
        }
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let at = |xx: usize, yy: usize, c: usize| src[(yy * w + xx) * 3 + c];
        for c in 0..3 {
            let a = at(x0, y0, c) * (1.0 - tx) + at(x0 + 1, y0, c) * tx;
            let b = at(x0, y0 + 1, c) * (1.0 - tx) + at(x0 + 1, y0 + 1, c) * tx;
            p[c] = a * (1.0 - ty) + b * ty;
        }
    });
}

/// Source span `[start, end)` that one destination pixel covers.
fn box_span(i: usize, dn: usize, sn: usize) -> (usize, usize) {
    let start = (i * sn / dn).min(sn.saturating_sub(1));
    let end = (((i + 1) * sn + dn - 1) / dn).min(sn).max(start + 1);
    (start, end)
}

/// Half-resolution copy of an RGB window, box-averaged over the source block.
/// The matcher runs a phase correlation per 128 px tile, so point-sampling the
/// block instead of averaging it throws away the high-frequency roof and ridge
/// texture the correlation needs to find a shift at all.
fn shrink_rgb(src: &[f32], w: usize, h: usize, hw: usize, hh: usize) -> Vec<f32> {
    let mut out = vec![0f32; hw * hh * 3];
    out.par_chunks_mut(3).enumerate().for_each(|(i, p)| {
        let (y, x) = (i / hw, i % hw);
        let (x0, x1) = box_span(x, hw, w);
        let (y0, y1) = box_span(y, hh, h);
        let mut acc = [0f32; 3];
        let mut n = 0f32;
        for sy in y0..y1 {
            let row = sy * w;
            for sx in x0..x1 {
                let s = (row + sx) * 3;
                acc[0] += src[s];
                acc[1] += src[s + 1];
                acc[2] += src[s + 2];
                n += 1.0;
            }
        }
        if n > 0.0 {
            p[0] = acc[0] / n;
            p[1] = acc[1] / n;
            p[2] = acc[2] / n;
        }
    });
    out
}

fn luma_plane(rgb: &[f32]) -> Vec<f32> {
    rgb.chunks_exact(3).map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).collect()
}

/// Half-resolution luma plane, box-averaged over the source block.
fn shrink_luma(src: &[f32], w: usize, h: usize, hw: usize, hh: usize) -> Vec<f32> {
    let mut out = vec![0f32; hw * hh];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let (y, x) = (i / hw, i % hw);
        let (x0, x1) = box_span(x, hw, w);
        let (y0, y1) = box_span(y, hh, h);
        let mut acc = 0f32;
        let mut n = 0f32;
        for sy in y0..y1 {
            let row = sy * w;
            for sx in x0..x1 {
                acc += src[row + sx];
                n += 1.0;
            }
        }
        *o = if n > 0.0 { acc / n } else { 0.0 };
    });
    out
}

fn shrink_plane(src: &[f32], w: usize, h: usize, hw: usize, hh: usize) -> Vec<f32> {
    let mut out = vec![0f32; hw * hh];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let y = i / hw;
        let x = i % hw;
        let (mut sy, mut sx) = (0usize, 0usize);
        for k in 0..4 {
            sy += (y * 4 + k) * w / hh;
            sx += (x * 4 + k) * w / hw;
        }
        let (sy, sx) = (sy / 4, sx / 4);
        *o = if sy < h && sx < w { src[sy * w + sx] } else { 0.0 };
    });
    out
}

fn shrink_bool(src: &[bool], w: usize, h: usize, hw: usize, hh: usize) -> Vec<bool> {
    let mut out = vec![false; hw * hh];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let (y, x) = (i / hw, i % hw);
        let sy = (y * h / hh).min(h.saturating_sub(1));
        let sx = (x * w / hw).min(w.saturating_sub(1));
        *o = src[sy * w + sx];
    });
    out
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

/// Warp displacement for one frame, in full-resolution canvas pixels.
///
/// `shift` is the field solved on the preview canvas. It is defined only over
/// the seam's co-valid overlap box (plus a margin), and this used to return
/// exactly zero outside that rectangle, which makes the displacement a
/// hard-edged patch: the frame moves by the full local amount inside the box
/// and not at all one pixel beyond it.
///
/// That is invisible wherever the box edge falls on textureless ground, and a
/// visible step of exactly the local shift wherever it crosses a strong edge.
/// DSCF0566's tower is the clean case - an owner-map dump shows the whole
/// tower supplied by one frame, so no seam is involved, yet the top and bottom
/// of it sit 8 px apart because the box edge for that frame crosses the
/// silhouette at y~1745.
///
/// So the field is faded out over a band at the edge of its own support, which
/// makes the displacement continuous everywhere. Inside the overlap nothing
/// changes - that is where the field was measured and where it is worth
/// trusting. Only the unsupported rim is ramped, and it ramps to nothing rather
/// than to a guess.
fn sample_nudge(shift: Option<&RecordedShift>, x: f64, y: f64, preview_w: u32, preview_h: u32, full_w: u32, full_h: u32) -> (f64, f64) {
    // v38 diagnostic: report the sampled field so a run can be checked without
    // relying on the pixel snapshots alone. Reports the value at the seam-9
    // window corners and midspan, which is where the measured ramp lives.
    if std::env::var("LEAN_NUDGE_TRACE").is_ok() {
        let (nx, ny) = sample_nudge_inner(shift, x, y, preview_w, preview_h, full_w, full_h);
        if (x as usize) % 977 == 0 && (y as usize) % 977 == 0 {
            trace::line(&format!("nudge_trace x={x:.1} y={y:.1} dx={nx:.3} dy={ny:.3}"));
        }
        return (nx, ny);
    }
    sample_nudge_inner(shift, x, y, preview_w, preview_h, full_w, full_h)
}

/// v39 diagnostic buffer. Must be shared, not thread-local: the warp pixel loop
/// runs on Rayon workers and every worker contributes samples.
static NTRACE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Emit the collected samples from the calling (main) thread, where the
/// thread-local stitch log is actually installed.
pub fn flush_ntrace() {
    if let Ok(mut guard) = NTRACE.lock() {
        for line in guard.drain(..) {
            trace::line(&line);
        }
    }
}

fn sample_nudge_inner(shift: Option<&RecordedShift>, x: f64, y: f64, preview_w: u32, preview_h: u32, full_w: u32, full_h: u32) -> (f64, f64) {
    let Some(shift) = shift else { return (0.0, 0.0) };
    if shift.w < 2 || shift.h < 2 {
        return (0.0, 0.0);
    }
    // v39 diagnostic: the warp pixel loop runs on Rayon worker threads, where
    // trace::line has no log installed (it is a thread-local). Collect the
    // samples here and let the caller flush them on the main thread.
    if std::env::var("LEAN_NUDGE_TRACE").is_ok() {
        let mut sx = x as usize;
        let mut sy = y as usize;
        if sx >= 5000 && sx < 8500 && sy >= 2000 && sy < 6600 && sx % 500 < 60 && sy % 2200 < 40 {
            let px = x * preview_w as f64 / full_w as f64;
            let py = y * preview_h as f64 / full_h as f64;
            let lx = px - shift.x as f64;
            let ly = py - shift.y as f64;
            let (raw_x, taper, ok) = if lx >= 0.0 && ly >= 0.0 && lx < shift.w as f64 - 1.0 && ly < shift.h as f64 - 1.0 {
                let x0 = lx.floor() as usize;
                let y0 = ly.floor() as usize;
                let tx = (lx - x0 as f64) as f32;
                let ty = (ly - y0 as f64) as f32;
                let mix = |src: &[f32]| {
                    let at = |yy: usize, xx: usize| src[yy * shift.w + xx];
                    let a = at(y0, x0) * (1.0 - tx) + at(y0, x0 + 1) * tx;
                    let b = at(y0 + 1, x0) * (1.0 - tx) + at(y0 + 1, x0 + 1) * tx;
                    a * (1.0 - ty) + b * ty
                };
                (mix(&shift.dx), edge_taper(lx, ly, shift.w as f64, shift.h as f64), true)
            } else {
                (0.0, 0.0, false)
            };
            let scale = full_w as f64 / preview_w as f64;
            if let Ok(mut guard) = NTRACE.lock() {
                guard.push(format!(
                    "ntrace x={x:.0} y={y:.0} lx={lx:.1} in_field={ok} raw_dx={raw_x:.3} taper={taper:.3} scaled={:.2} final={:.2}",
                    raw_x as f64 * scale, raw_x as f64 * scale * taper
                ));
            }
        }
    }
    // v39 diagnostic: report the RAW field value before the edge taper and
    // before the preview->full scaling. Comparing this against the taper
    // output tells us whether the ramp comes from the field itself or from
    // the taper/scale applied on top of it.
    if std::env::var("LEAN_NUDGE_TRACE").is_ok() {
        let px = x * preview_w as f64 / full_w as f64;
        let py = y * preview_h as f64 / full_h as f64;
        let lx = px - shift.x as f64;
        let ly = py - shift.y as f64;
        if lx >= 0.0 && ly >= 0.0 && lx < shift.w as f64 - 1.0 && ly < shift.h as f64 - 1.0 {
            let x0 = lx.floor() as usize;
            let y0 = ly.floor() as usize;
            let tx = (lx - x0 as f64) as f32;
            let ty = (ly - y0 as f64) as f32;
            let mix = |src: &[f32]| {
                let at = |yy: usize, xx: usize| src[yy * shift.w + xx];
                let a = at(y0, x0) * (1.0 - tx) + at(y0, x0 + 1) * tx;
                let b = at(y0 + 1, x0) * (1.0 - tx) + at(y0 + 1, x0 + 1) * tx;
                a * (1.0 - ty) + b * ty
            };
            let raw_x = mix(&shift.dx);
            let raw_y = mix(&shift.dy);
            let taper = edge_taper(lx, ly, shift.w as f64, shift.h as f64);
            let scale = full_w as f64 / preview_w as f64;
            let xu = x as usize;
            let yu = y as usize;
            if xu >= 5000 && xu < 8500 && yu >= 2000 && yu < 6600 && xu % 500 < 60 && yu % 2200 < 40 {
                trace::line(&format!(
                    "ntrace x={x:.0} y={y:.0} lx={lx:.1} raw_dx={raw_x:.3} taper={taper:.3} scaled={:.2} final={:.2} edge_d={:.0}",
                    raw_x as f64 * scale,
                    raw_x as f64 * scale * taper,
                    lx.min(shift.w as f64 - lx)
                ));
            }
        } else {
            let xu = x as usize;
            let yu = y as usize;
            if xu >= 5000 && xu < 8500 && yu >= 2000 && yu < 6600 && xu % 500 < 60 && yu % 2200 < 40 {
                trace::line(&format!("ntrace x={x:.0} y={y:.0} OUTSIDE field lx={lx:.1} w={} -> 0", shift.w));
            }
        }
    }
    let px = x * preview_w as f64 / full_w as f64;
    let py = y * preview_h as f64 / full_h as f64;
    let lx = px - shift.x as f64;
    let ly = py - shift.y as f64;
    if lx < 0.0 || ly < 0.0 || lx >= shift.w as f64 - 1.0 || ly >= shift.h as f64 - 1.0 {
        return (0.0, 0.0);
    }
    let taper = edge_taper(lx, ly, shift.w as f64, shift.h as f64);
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
    (
        mix(&shift.dx) * full_w as f64 / preview_w as f64 * taper,
        mix(&shift.dy) * full_h as f64 / preview_h as f64 * taper,
    )
}

/// Smooth ramp to zero across the rim of the field's own support.
///
/// The band is a fraction of the smaller field dimension with a floor of a few
/// preview pixels, so it stays smooth after the field is scaled up to the
/// full-resolution canvas. `smoothstep` is used rather than a linear ramp
/// because its derivative is zero at both ends, so the warp stays continuous in
/// value *and* in slope; a linear ramp would leave a crease of its own where it
/// meets zero, trading one step for another.
fn edge_taper(lx: f64, ly: f64, w: f64, h: f64) -> f64 {
    let band = (w.min(h) * 0.06).max(4.0);
    let dx = ((w - 1.0 - lx) / band).clamp(0.0, 1.0);
    let dy = ((h - 1.0 - ly) / band).clamp(0.0, 1.0);
    let d = dx.min(dy);
    d * d * (3.0 - 2.0 * d)
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

// ---------------------------------------------------------------------------
// v37 snapshots. Diagnostics only - nothing here alters what the pipeline
// computes, it just writes PNGs so the DSCF0566 defect can be attributed to a
// step rather than guessed at.
//
// Every snapshot uses ONE window, defined in window-local coordinates, and
// every filename carries the window origin and size. Stages inside
// `composite_overlap` address their arrays in window coordinates and the
// stages in `render` address the full canvas, so the canvas-local stages are
// shifted into the same window before writing. That way all the files are
// directly comparable: same pixels, same size, one origin.
//
// Gated on LEAN_SNAP=<set stem>. LEAN_SNAP_BOX=x0,y0,x1,y1 is in WINDOW
// coordinates. Output directory is LEAN_SNAP_DIR, defaulting to a `snapshots`
// folder beside the set's own output.
// ---------------------------------------------------------------------------

fn snap_stem() -> Option<String> {
    std::env::var("LEAN_SNAP").ok().filter(|s| !s.is_empty())
}

fn snap_is(stem: &str) -> bool {
    snap_stem().as_deref() == Some(stem)
}

fn snap_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("LEAN_SNAP_DIR").unwrap_or_else(|_| "/tmp/opencode/snaps".into()));
    std::fs::create_dir_all(&d).ok()?;
    Some(d)
}

/// Window-local box, clamped to the buffer it will index.
fn snap_win(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let v = std::env::var("LEAN_SNAP_BOX").ok()?;
    let p: Vec<i64> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    if p.len() != 4 {
        return None;
    }
    let x0 = p[0].max(0) as usize;
    let y0 = p[1].max(0) as usize;
    let x1 = ((p[2].max(0) as usize).min(w)).max(x0 + 1);
    let y1 = ((p[3].max(0) as usize).min(h)).max(y0 + 1);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0, y0, x1, y1))
}

fn snap_save(stem: &str, tag: &str, w: usize, h: usize, origin: (usize, usize), build: impl Fn(usize, usize) -> Vec<u8>) {
    if !snap_is(stem) {
        return;
    }
    let Some(dir) = snap_dir() else { return };
    let Some((bx0, by0, bx1, by1)) = snap_win(w, h) else { return };
    let cw = bx1 - bx0;
    let ch = by1 - by0;
    let buf = build(cw, ch);
    let path = dir.join(format!("{stem}_{tag}_w{}_h{}_ox{}_oy{}.png", cw, ch, origin.0, origin.1));
    match image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(cw as u32, ch as u32, buf) {
        Some(i) => match i.save_with_format(&path, image::ImageFormat::Png) {
            Ok(()) => trace::line(&format!("SNAP {tag} {}", path.display())),
            Err(e) => trace::line(&format!("SNAP {tag} FAILED {e}")),
        },
        None => trace::line(&format!("SNAP {tag} FAILED buffer")),
    }
}

/// v39 diagnostic: render a RecordedShift field as a PNG so the fitted
/// displacement and its support can be inspected directly.
///
/// Red/blue encode dx (blue = positive, red = negative, grey = zero) at a
/// fixed gain so a ramp across the field is directly visible. Green encodes
/// dy on the same scale. A fully black region is zero displacement, which is
/// what `align_incoming` produces outside the overlap it measured.
fn snap_field(tag: &str, sh: &RecordedShift) {
    let Some(dir) = snap_dir() else { return };
    // Fixed gain in field (preview) pixels, so field values compare across
    // frames and runs. Anything beyond +-4 px saturates.
    const GAIN: f32 = 64.0;
    let w = sh.w;
    let h = sh.h;
    let mut buf = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let dx = sh.dx[i];
            let dy = sh.dy[i];
            let mag = dx.hypot(dy);
            // Saturated magnitude is pushed into the blue channel's excess so a
            // strong field is still distinguishable from a merely signed one.
            let r = ((-dx * GAIN).clamp(0.0, 255.0)) as u8;
            let g = ((dy.abs() * GAIN).clamp(0.0, 255.0)) as u8;
            let b = ((dx * GAIN).clamp(0.0, 255.0)) as u8;
            let boost = if mag > 4.0 { 40u8 } else { 0 };
            buf[i * 3] = r.saturating_add(boost);
            buf[i * 3 + 1] = g;
            buf[i * 3 + 2] = b.saturating_add(boost);
        }
    }
    let path = dir.join(format!("{tag}.png"));
    match image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, buf) {
        Some(i) => match i.save_with_format(&path, image::ImageFormat::Png) {
            Ok(()) => trace::line(&format!("SNAP {tag} {} ({w}x{h})", path.display())),
            Err(e) => trace::line(&format!("SNAP {tag} FAILED {e}")),
        },
        None => trace::line(&format!("SNAP {tag} FAILED buffer")),
    }
}

/// Window-local RGB snapshot of a packed f32 window.
fn snap_rgb(stem: &str, tag: &str, buf: &[f32], w: usize, h: usize, origin: (usize, usize)) {
    snap_save(stem, tag, w, h, origin, |cw, _ch| {
        let Some((bx0, by0, bx1, by1)) = snap_win(w, h) else { return Vec::new() };
        let mut out = Vec::with_capacity(cw * (by1 - by0) * 3);
        for y in by0..by1 {
            let row = y * w;
            for x in bx0..bx1 {
                let i = (row + x) * 3;
                out.push(to_srgb(buf[i]));
                out.push(to_srgb(buf[i + 1]));
                out.push(to_srgb(buf[i + 2]));
            }
        }
        out
    });
}

/// Window-local snapshot of a mask, written into the red channel, with the
/// blend rectangle outlined in green so the two can be seen together.
fn snap_mask(stem: &str, tag: &str, mask: &[f32], w: usize, h: usize, origin: (usize, usize), rect: Option<(usize, usize, usize, usize)>) {
    snap_save(stem, tag, w, h, origin, move |cw, ch| {
        let Some((bx0, by0, bx1, by1)) = snap_win(w, h) else { return Vec::new() };
        let mut out = vec![0u8; cw * ch * 3];
        for y in by0..by1 {
            for x in bx0..bx1 {
                let di = y * w + x;
                let o = ((y - by0) * cw + (x - bx0)) * 3;
                out[o] = (mask[di].clamp(0.0, 1.0) * 255.0) as u8;
                if let Some((rx0, ry0, rx1, ry1)) = rect {
                    if x == rx0 || x + 1 == rx1 || y == ry0 || y + 1 == ry1 {
                        out[o + 1] = 255;
                    }
                }
            }
        }
        out
    });
}

/// Window-local validity map: red = canvas side valid, green = incoming side
/// valid, blue = both. Shows at a glance which side can even contribute.
fn snap_valid(stem: &str, tag: &str, av: &[bool], bv: &[bool], w: usize, h: usize, origin: (usize, usize)) {
    snap_save(stem, tag, w, h, origin, move |cw, ch| {
        let Some((bx0, by0, bx1, by1)) = snap_win(w, h) else { return Vec::new() };
        let mut out = vec![0u8; cw * ch * 3];
        for y in by0..by1 {
            for x in bx0..bx1 {
                let di = y * w + x;
                let o = ((y - by0) * cw + (x - bx0)) * 3;
                out[o] = if av[di] { 255 } else { 0 };
                out[o + 1] = if bv[di] { 255 } else { 0 };
                out[o + 2] = if av[di] && bv[di] { 255 } else { 0 };
            }
        }
        out
    });
}

/// Placed-frame snapshot on the same window as everything else. The frame's
/// own image is addressed in its own coordinates, so it is written through the
/// same canvas offset to land in the identical window.
fn snap_placed_window(stem: &str, tag: &str, p: &Placed, canvas_w: usize, canvas_h: usize, win_org: (usize, usize)) {
    snap_save(stem, tag, canvas_w, canvas_h, win_org, move |cw, ch| {
        let mut out = vec![0u8; cw * ch * 3];
        let px0 = p.x0.max(0) as usize;
        let py0 = p.y0.max(0) as usize;
        let px1 = (p.x0 + p.w as i32).clamp(0, canvas_w as i32) as usize;
        let py1 = (p.y0 + p.h as i32).clamp(0, canvas_h as i32) as usize;
        for y in win_org.1..(win_org.1 + ch).min(canvas_h) {
            for x in win_org.0..(win_org.0 + cw).min(canvas_w) {
                if x < px0 || x >= px1 || y < py0 || y >= py1 {
                    continue;
                }
                let si = ((y - py0) * p.w + (x - px0)) * 3;
                if si + 2 >= p.img.len() {
                    continue;
                }
                let o = ((y - win_org.1) * cw + (x - win_org.0)) * 3;
                out[o] = to_srgb(p.img[si]);
                out[o + 1] = to_srgb(p.img[si + 1]);
                out[o + 2] = to_srgb(p.img[si + 2]);
            }
        }
        out
    });
}

/// Canvas-local RGB snapshot, written into the same window as the in-function
/// shots by slicing at `win_origin + box`.
fn snap_canvas(stem: &str, tag: &str, canvas: &[f32], cw: usize, ch: usize, win_origin: (usize, usize), canvas_box: (usize, usize, usize, usize)) {
    snap_save(stem, tag, cw, ch, win_origin, |bw, bh| {
        let mut out = Vec::with_capacity(bw * bh * 3);
        let (ox, oy) = win_origin;
        for y in 0..bh {
            let cy = oy + y;
            if cy >= ch {
                out.resize(out.len() + bw * 3, 0);
                continue;
            }
            for x in 0..bw {
                let cx = ox + x;
                let inside = cx >= canvas_box.0 && cx < canvas_box.2 && cy >= canvas_box.1 && cy < canvas_box.3;
                if inside {
                    let i = (cy * cw + cx) * 3;
                    out.push(to_srgb(canvas[i]));
                    out.push(to_srgb(canvas[i + 1]));
                    out.push(to_srgb(canvas[i + 2]));
                } else {
                    out.push(0);
                    out.push(0);
                    out.push(0);
                }
            }
        }
        out
    });
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
