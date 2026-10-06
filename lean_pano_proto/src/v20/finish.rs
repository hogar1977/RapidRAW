use super::align::{self, Preview};
use super::exec::pick_exec;
use rapidraw_lib::panorama_stitching::{self, exposure_gains};
use rapidraw_lib::panorama_utils::v86::blend::pyr_blend;
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

pub fn render(paths: &[String], preview: &Preview, tmp: &Path, tiff_path: &Path, view_path: &Path) -> Result<(), String> {
    let exec = pick_exec();
    let dw = preview.frames[0].width;
    let dh = preview.frames[0].height;
    let mut stored = Vec::new();
    let mut evs = Vec::new();
    for &index in &preview.indices {
        let frame = exec.develop_raw(&paths[index])?;
        evs.push(frame.exposure_ev);
        let file = tmp.join(format!("{:02}-{}.f32", stored.len(), frame.name));
        write_f32(&file, frame.width, frame.height, &frame.rgb)?;
        let small = align::shrink(&frame.rgb, frame.width, frame.height, dw, dh);
        trace::line(&format!("developed {} {}x{} -> {dw}x{dh}", frame.name, frame.width, frame.height));
        stored.push(Stored { file, width: frame.width, height: frame.height, name: frame.name, small });
    }
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
    let linear = stitch::compose(&linear_frames, &preview.rots, &kept, preview.lens, preview.focal, preview.focal35, preview.result.used, false, &|message| {
        trace::line(message);
    })?;
    let _developed_shifts = local_align::end_shift_log();
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
    let mut dist = vec![u16::MAX; fw * fh];
    let blur_repeats = 1usize;
    for (bi, &frame_i) in preview.result.kept.iter().enumerate() {
        trace::gate()?;
        let item = &stored[frame_i];
        let mut rgb = read_f32(&item.file)?;
        let gain = gains[frame_i];
        if (gain - 1.0).abs() > 1e-4 {
            for p in &mut rgb {
                *p *= gain;
            }
        }
        let shift = if bi == 0 { None } else { shifts.get(bi - 1).and_then(|entry| entry.as_ref()) };
        let nudge = |x: f64, y: f64| sample_nudge(shift, x, y, seam_w, seam_h, geom.width, geom.height);
        let mut placed = exec.warp(&mut rgb, item.width, item.height, &preview.rots[frame_i], focal_full, &preview.lens, &geom, &nudge);
        drop(rgb);
        let _ = std::fs::remove_file(&item.file);
        photo::apply_photometry(&mut placed, if use_linear { &linear.photo } else { &preview.result.photo }, bi);
        let id = bi as u16;
        if bi > 0 {
            residual_match(&acc, &written, &mut placed, fw, fh, id);
        }
        let cut = if bi > 0 { full_cut(&acc, &written, &placed, fw, fh, id) } else { None };
        if let Some(cut) = &cut {
            trace::line(&format!("full seam {id} {}x{}", cut.w, cut.h));
        } else if bi > 0 {
            trace::line(&format!("full seam {id} none"));
        }
        paint_owned(&mut acc, &mut written, &mut owners, &placed, id, cut.as_ref(), fw, fh);
        if bi > 0 {
            let pad = 192isize;
            let x0 = (placed.x0 as isize - pad).max(0) as usize;
            let y0 = (placed.y0 as isize - pad).max(0) as usize;
            let x1 = (placed.x0 + placed.w as i32 + pad as i32).clamp(0, fw as i32) as usize;
            let y1 = (placed.y0 + placed.h as i32 + pad as i32).clamp(0, fh as i32) as usize;
            fill_distance(&mut dist, &owners, fw, fh, x0, y0, x1, y1, 192);
            exec.blend_ribbon(&mut Ribbon {
                acc: &mut acc,
                written: &mut written,
                placed: &placed,
                id,
                winners: &owners,
                dist: &dist,
                pw: fw as u32,
                ph: fh as u32,
                fw,
                fh,
                scan_x0: x0,
                scan_y0: y0,
                scan_x1: x1,
                scan_y1: y1,
                band_cap: 192,
                repeats: blur_repeats,
            });
        }
        trace::line(&format!("warped {} {}", item.name, if shift.is_some() { "nudged" } else { "straight" }));
    }
    let cropped = crop(&acc, geom.width, geom.height, crop_x, crop_y, crop_w, crop_h);
    trace::line(&format!("save encode {}x{}", cropped.1, cropped.2));
    panorama_stitching::write_panorama_tiff(&cropped.0, cropped.1, cropped.2, tiff_path)?;
    write_view(&cropped.0, cropped.1, cropped.2, view_path)?;
    Ok(())
}

struct Stored {
    file: PathBuf,
    width: u32,
    height: u32,
    name: String,
    small: Vec<f32>,
}

fn paint_owned(acc: &mut [f32], written: &mut [u8], owners: &mut [u16], placed: &Placed, id: u16, cut: Option<&Cut>, fw: usize, fh: usize) {
    let y0 = placed.y0.max(0) as usize;
    let y1 = (placed.y0 + placed.h as i32).clamp(0, fh as i32) as usize;
    if y1 <= y0 || placed.w == 0 {
        return;
    }
    let rows = y1 - y0;
    let (acc_rows, rest) = acc.split_at_mut(y1 * fw * 3);
    let _ = rest;
    acc_rows[y0 * fw * 3..y1 * fw * 3]
        .par_chunks_mut(fw * 3)
        .zip(written[y0 * fw..y1 * fw].par_chunks_mut(fw))
        .zip(owners[y0 * fw..y1 * fw].par_chunks_mut(fw))
        .enumerate()
        .for_each(|(row, ((dest, flags), owned))| {
            if row >= rows {
                return;
            }
            let cy = y0 + row;
            let py = cy as i32 - placed.y0;
            if py < 0 || py >= placed.h as i32 {
                return;
            }
            for x in 0..fw {
                let px = x as i32 - placed.x0;
                if px < 0 || px >= placed.w as i32 {
                    continue;
                }
                let s = py as usize * placed.w + px as usize;
                if !placed.valid[s] || !frame_owns(cut, flags[x], x, cy) {
                    continue;
                }
                dest[x * 3] = placed.img[s * 3];
                dest[x * 3 + 1] = placed.img[s * 3 + 1];
                dest[x * 3 + 2] = placed.img[s * 3 + 2];
                flags[x] = 1;
                owned[x] = id;
            }
        });
}

fn frame_owns(cut: Option<&Cut>, already: u8, x: usize, y: usize) -> bool {
    let Some(cut) = cut else { return true };
    if x >= cut.x0 && y >= cut.y0 && x < cut.x0 + cut.w && y < cut.y0 + cut.h {
        return cut.mask[(y - cut.y0) * cut.w + (x - cut.x0)] < 0.5;
    }
    already == 0
}

pub struct Ribbon<'a> {
    pub acc: &'a mut [f32],
    pub written: &'a mut [u8],
    pub placed: &'a Placed,
    pub id: u16,
    pub winners: &'a [u16],
    pub dist: &'a [u16],
    pub pw: u32,
    pub ph: u32,
    pub fw: usize,
    pub fh: usize,
    pub scan_x0: usize,
    pub scan_y0: usize,
    pub scan_x1: usize,
    pub scan_y1: usize,
    pub band_cap: u16,
    pub repeats: usize,
}

pub(super) fn run_ribbon(job: &mut Ribbon<'_>) {
    blend_strip(
        job.acc,
        job.written,
        job.placed,
        job.id,
        job.winners,
        job.dist,
        job.pw,
        job.ph,
        job.fw,
        job.fh,
        job.scan_x0,
        job.scan_y0,
        job.scan_x1,
        job.scan_y1,
        job.band_cap,
        job.repeats,
    );
}

fn blend_strip(
    acc: &mut [f32],
    written: &mut [u8],
    placed: &Placed,
    id: u16,
    winners: &[u16],
    dist: &[u16],
    pw: u32,
    ph: u32,
    fw: usize,
    fh: usize,
    scan_x0: usize,
    scan_y0: usize,
    scan_x1: usize,
    scan_y1: usize,
    band_cap: u16,
    repeats: usize,
) {
    let sx = fw as f64 / pw.max(1) as f64;
    let sy = fh as f64 / ph.max(1) as f64;
    let band_preview = ((BAND_PX / sx).ceil() as u16).clamp(2, band_cap);
    let slab_h = 256usize;
    let nslabs = fh.div_ceil(slab_h);
    let pw_us = pw as usize;
    let ph_us = ph as usize;
    let mut cols: Vec<Vec<u32>> = vec![Vec::new(); nslabs];
    let scan_y1 = scan_y1.min(ph_us);
    let scan_x1 = scan_x1.min(pw_us);
    for py in scan_y0..scan_y1 {
        let row = py * pw_us;
        let fy0 = ((py as f64) * sy).floor() as usize;
        let fy1 = (((py + 1) as f64) * sy).ceil() as usize;
        let s_lo = (fy0 / slab_h).min(nslabs - 1);
        let s_hi = (fy1.saturating_sub(1) / slab_h).min(nslabs - 1);
        for px in scan_x0..scan_x1 {
            if dist[row + px] > band_preview {
                continue;
            }
            let here = winners[row + px];
            if here != id && !neighbor_is(winners, pw_us, ph_us, px, py, id) {
                continue;
            }
            for s in s_lo..=s_hi {
                cols[s].push(px as u32);
            }
        }
    }
    let mut pieces: Vec<Vec<(usize, usize)>> = Vec::with_capacity(nslabs);
    for s in 0..nslabs {
        let col = &mut cols[s];
        let mut segs = Vec::new();
        if !col.is_empty() {
            col.sort_unstable();
            col.dedup();
            let mut i = 0usize;
            while i < col.len() {
                let mut j = i;
                while j + 1 < col.len() && col[j + 1] <= col[j] + 1 {
                    j += 1;
                }
                let core_x0 = ((col[i] as f64) * sx).floor() as usize;
                let core_x1 = (((col[j] as f64) + 1.0) * sx).ceil() as usize;
                let core_x1 = core_x1.min(fw);
                if core_x1 > core_x0 {
                    segs.push((core_x0, core_x1));
                }
                i = j + 1;
            }
        }
        pieces.push(segs);
    }
    let pad = 96usize;
    let budget = 2_500_000usize;
    let mut open: Vec<(usize, usize, usize)> = Vec::new();
    let mut slabs_used = 0usize;
    let mut pixels = 0usize;
    for s in 0..nslabs {
        let mut used = vec![false; pieces[s].len()];
        let mut next: Vec<(usize, usize, usize)> = Vec::new();
        for (s0, x0, x1) in open {
            let mut matched = None;
            for (i, &(nx0, nx1)) in pieces[s].iter().enumerate() {
                if used[i] || nx1 <= x0 || nx0 >= x1 {
                    continue;
                }
                let ux0 = x0.min(nx0);
                let ux1 = x1.max(nx1);
                let h = ((s + 1) * slab_h).min(fh) - s0 * slab_h;
                if (ux1 - ux0 + 2 * pad) * (h + 2 * pad) <= budget {
                    matched = Some((i, ux0, ux1));
                    break;
                }
            }
            if let Some((i, ux0, ux1)) = matched {
                used[i] = true;
                next.push((s0, ux0, ux1));
            } else {
                blend_piece(acc, written, placed, id, winners, dist, pw, ph, fw, fh, s0, s, x0, x1, slab_h, pad, sx, repeats, &mut slabs_used, &mut pixels);
            }
        }
        for (i, &(x0, x1)) in pieces[s].iter().enumerate() {
            if !used[i] {
                next.push((s, x0, x1));
            }
        }
        open = next;
    }
    for (s0, x0, x1) in open {
        blend_piece(acc, written, placed, id, winners, dist, pw, ph, fw, fh, s0, nslabs, x0, x1, slab_h, pad, sx, repeats, &mut slabs_used, &mut pixels);
    }
    if slabs_used > 0 {
        trace::line(&format!("seam strip {id} slabs={slabs_used} px={pixels} levels=2 hard"));
    }
}

fn blend_piece(
    acc: &mut [f32],
    written: &mut [u8],
    placed: &Placed,
    id: u16,
    winners: &[u16],
    dist: &[u16],
    pw: u32,
    ph: u32,
    fw: usize,
    fh: usize,
    s0: usize,
    s1: usize,
    core_x0: usize,
    core_x1: usize,
    slab_h: usize,
    pad: usize,
    sx: f64,
    repeats: usize,
    slabs_used: &mut usize,
    pixels: &mut usize,
) {
    let core_y0 = s0 * slab_h;
    let core_y1 = (s1 * slab_h).min(fh);
    if core_x1 <= core_x0 || core_y1 <= core_y0 {
        return;
    }
    let x0 = core_x0.saturating_sub(pad);
    let x1 = (core_x1 + pad).min(fw);
    let y0 = core_y0.saturating_sub(pad);
    let y1 = (core_y1 + pad).min(fh);
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return;
    }
    blend_window(acc, written, placed, id, winners, dist, pw, ph, fw, fh, x0, y0, x1, y1, core_y0, core_y1, core_x0, core_x1, sx, repeats);
    *slabs_used += 1;
    *pixels += (x1 - x0) * (y1 - y0);
}

fn blend_window(
    acc: &mut [f32],
    written: &mut [u8],
    placed: &Placed,
    id: u16,
    winners: &[u16],
    dist: &[u16],
    pw: u32,
    ph: u32,
    fw: usize,
    fh: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    core_y0: usize,
    core_y1: usize,
    core_x0: usize,
    core_x1: usize,
    sx: f64,
    _repeats: usize,
) {
    let w = x1 - x0;
    let h = y1 - y0;
    let mut a = vec![0f32; w * h * 3];
    let mut b = vec![0f32; w * h * 3];
    let mut va = vec![false; w * h];
    let mut vb = vec![false; w * h];
    let mut mask = vec![1f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let cx = x0 + x;
            let cy = y0 + y;
            let i = y * w + x;
            if written[cy * fw + cx] != 0 {
                let s = (cy * fw + cx) * 3;
                a[i * 3] = acc[s];
                a[i * 3 + 1] = acc[s + 1];
                a[i * 3 + 2] = acc[s + 2];
                va[i] = true;
            }
            if let Some(pix) = placed_at(placed, cx as i32, cy as i32) {
                b[i * 3] = pix[0];
                b[i * 3 + 1] = pix[1];
                b[i * 3 + 2] = pix[2];
                vb[i] = true;
            }
            if winner_at(winners, pw, ph, fw as u32, fh as u32, cx as i32, cy as i32) == id {
                mask[i] = 0.0;
            }
        }
    }
    let blended = pyr_blend(&a, &b, &mask, &va, &vb, w, h, 2);
    for y in 0..h {
        let cy = y0 + y;
        if cy < core_y0 || cy >= core_y1 {
            continue;
        }
        for x in 0..w {
            let cx = x0 + x;
            if cx < core_x0 || cx >= core_x1 {
                continue;
            }
            let d = dist_at(dist, pw, ph, fw as u32, fh as u32, cx as i32, cy as i32);
            if d == u16::MAX || (d as f64) * sx >= BAND_PX {
                continue;
            }
            let i = y * w + x;
            if !va[i] && !vb[i] {
                continue;
            }
            let s = (cy * fw + cx) * 3;
            acc[s] = blended[i * 3];
            acc[s + 1] = blended[i * 3 + 1];
            acc[s + 2] = blended[i * 3 + 2];
            written[cy * fw + cx] = 1;
        }
    }
}

fn neighbor_is(winners: &[u16], w: usize, h: usize, x: usize, y: usize, id: u16) -> bool {
    [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)].into_iter().any(|(dy, dx)| {
        let yy = y as isize + dy;
        let xx = x as isize + dx;
        yy >= 0 && xx >= 0 && (yy as usize) < h && (xx as usize) < w && winners[yy as usize * w + xx as usize] == id
    })
}

fn placed_at(placed: &Placed, x: i32, y: i32) -> Option<[f32; 3]> {
    let px = x - placed.x0;
    let py = y - placed.y0;
    if px < 0 || py < 0 || px >= placed.w as i32 || py >= placed.h as i32 {
        return None;
    }
    let s = py as usize * placed.w + px as usize;
    if !placed.valid[s] {
        return None;
    }
    Some([placed.img[s * 3], placed.img[s * 3 + 1], placed.img[s * 3 + 2]])
}

fn levels_for(w: usize, h: usize) -> usize {
    let mut levels = 6usize;
    let min = w.min(h).max(1);
    while levels > 1 && min >> levels < 2 {
        levels -= 1;
    }
    levels
}

struct Cut {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    mask: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Residual {
    dx: f32,
    dy: f32,
    gain: f32,
}

fn residual_match(acc: &[f32], written: &[u8], placed: &mut Placed, fw: usize, fh: usize, id: u16) {
    const CELL: usize = 128;
    const PATCH: i32 = 64;
    const RADIUS: i32 = 8;
    let quiet = rapidraw_lib::panorama_utils::v86::const_::QUIET_TEX;
    let x0 = placed.x0.max(PATCH) as usize;
    let y0 = placed.y0.max(PATCH) as usize;
    let x1 = ((placed.x0 + placed.w as i32).min(fw as i32) - PATCH).max(x0 as i32) as usize;
    let y1 = ((placed.y0 + placed.h as i32).min(fh as i32) - PATCH).max(y0 as i32) as usize;
    if x1 <= x0 || y1 <= y0 {
        trace::line(&format!("residual cells=0 moved=0"));
        return;
    }
    let cw = (x1 - x0).div_ceil(CELL).max(1);
    let ch = (y1 - y0).div_ceil(CELL).max(1);
    let mut cells = vec![Residual { dx: 0.0, dy: 0.0, gain: 1.0 }; cw * ch];
    cells.par_chunks_mut(cw).enumerate().for_each(|(iy, row)| {
        for (ix, cell) in row.iter_mut().enumerate() {
            let cx = (x0 + ix * CELL + CELL / 2).min(x1.saturating_sub(1));
            let cy = (y0 + iy * CELL + CELL / 2).min(y1.saturating_sub(1));
            *cell = measure_cell(acc, written, placed, fw, cx, cy, quiet, PATCH, RADIUS);
        }
    });
    let moved = cells.iter().filter(|c| c.dx != 0.0 || c.dy != 0.0 || (c.gain - 1.0).abs() > 1e-3).count();
    trace::line(&format!("residual cells={} moved={moved}", cw * ch));
    if moved == 0 {
        return;
    }
    let cells = blur_residual(&cells, cw, ch);
    let orig_img = placed.img.clone();
    let orig_valid = placed.valid.clone();
    let pw = placed.w;
    let ph = placed.h;
    let ox = placed.x0;
    let oy = placed.y0;
    for y in y0..y1 {
        for x in x0..x1 {
            if written[y * fw + x] == 0 {
                continue;
            }
            let (dx, dy, gain) = sample_residual(&cells, cw, ch, x0, y0, x, y);
            let px = x as i32 - ox;
            let py = y as i32 - oy;
            if px < 0 || py < 0 || px >= pw as i32 || py >= ph as i32 {
                continue;
            }
            let dest = py as usize * pw + px as usize;
            if !orig_valid[dest] {
                continue;
            }
            if let Some(pix) = sample_placed(&orig_img, &orig_valid, pw, ph, px as f32 + dx, py as f32 + dy) {
                let s = dest * 3;
                placed.img[s] = pix[0];
                placed.img[s + 1] = pix[1];
                placed.img[s + 2] = pix[2];
                placed.valid[dest] = true;
            }
            if (gain - 1.0).abs() < 1e-3 {
                continue;
            }
            let s = dest * 3;
            if dest < pw || dest % pw == 0 {
                continue;
            }
            let lb = luma(&placed.img[s..s + 3]);
            let left = luma(&placed.img[s - 3..s]);
            let up = luma(&placed.img[s - pw * 3..s - pw * 3 + 3]);
            if !flat(lb, left, up, quiet) {
                continue;
            }
            placed.img[s] *= gain;
            placed.img[s + 1] *= gain;
            placed.img[s + 2] *= gain;
        }
    }
}

fn measure_cell(acc: &[f32], written: &[u8], placed: &Placed, fw: usize, cx: usize, cy: usize, quiet: f32, patch: i32, radius: i32) -> Residual {
    let mut textured = 0u32;
    let mut ratios = Vec::new();
    let half = patch / 2;
    let mut y = cy as i32 - half;
    while y < cy as i32 + half {
        let mut x = cx as i32 - half;
        while x < cx as i32 + half {
            if let Some((la, lb, quiet_pair)) = pair_at(acc, written, placed, fw, x, y, quiet) {
                if quiet_pair {
                    if la > 1e-4 && lb > 1e-4 {
                        ratios.push(la / lb);
                    }
                } else {
                    textured += 1;
                }
            }
            x += 8;
        }
        y += 8;
    }
    if textured >= 4 {
        let Some(sad0) = patch_sad(acc, written, placed, fw, cx, cy, 0, 0, patch) else {
            return Residual { dx: 0.0, dy: 0.0, gain: 1.0 };
        };
        let mut best = (0i32, 0i32, sad0);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let Some(sad) = patch_sad(acc, written, placed, fw, cx, cy, dx, dy, patch) else {
                    continue;
                };
                if sad < best.2 {
                    best = (dx, dy, sad);
                }
            }
        }
        if best.2 < sad0 * 0.97 {
            return Residual { dx: best.0 as f32, dy: best.1 as f32, gain: 1.0 };
        }
        return Residual { dx: 0.0, dy: 0.0, gain: 1.0 };
    }
    if ratios.len() < 4 {
        return Residual { dx: 0.0, dy: 0.0, gain: 1.0 };
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Residual { dx: 0.0, dy: 0.0, gain: ratios[ratios.len() / 2].clamp(0.9, 1.1) }
}

fn pair_at(acc: &[f32], written: &[u8], placed: &Placed, fw: usize, x: i32, y: i32, quiet: f32) -> Option<(f32, f32, bool)> {
    if x < 1 || y < 1 {
        return None;
    }
    let x = x as usize;
    let y = y as usize;
    if y >= written.len() / fw || x >= fw || written[y * fw + x] == 0 || written[y * fw + x - 1] == 0 || written[(y - 1) * fw + x] == 0 {
        return None;
    }
    let px = x as i32 - placed.x0;
    let py = y as i32 - placed.y0;
    if px < 1 || py < 1 || px + 1 >= placed.w as i32 || py + 1 >= placed.h as i32 {
        return None;
    }
    let s = py as usize * placed.w + px as usize;
    if !placed.valid[s] || !placed.valid[s - 1] || !placed.valid[s - placed.w] {
        return None;
    }
    let la = luma(&acc[(y * fw + x) * 3..(y * fw + x) * 3 + 3]);
    let lb = luma(&placed.img[s * 3..s * 3 + 3]);
    let la_l = luma(&acc[(y * fw + x - 1) * 3..(y * fw + x - 1) * 3 + 3]);
    let la_u = luma(&acc[((y - 1) * fw + x) * 3..((y - 1) * fw + x) * 3 + 3]);
    let lb_l = luma(&placed.img[(s - 1) * 3..(s - 1) * 3 + 3]);
    let lb_u = luma(&placed.img[(s - placed.w) * 3..(s - placed.w) * 3 + 3]);
    let quiet_pair = flat(la, la_l, la_u, quiet) && flat(lb, lb_l, lb_u, quiet);
    Some((la, lb, quiet_pair))
}

fn patch_sad(acc: &[f32], written: &[u8], placed: &Placed, fw: usize, cx: usize, cy: usize, dx: i32, dy: i32, patch: i32) -> Option<f32> {
    let half = patch / 2;
    let mut sad = 0.0f32;
    let mut n = 0u32;
    let mut y = cy as i32 - half;
    while y < cy as i32 + half {
        let mut x = cx as i32 - half;
        while x < cx as i32 + half {
            let la = canvas_luma(acc, written, fw, x, y);
            let lb = placed_luma(placed, x + dx, y + dy);
            if let (Some(la), Some(lb)) = (la, lb) {
                sad += (la - lb).abs();
                n += 1;
            }
            x += 4;
        }
        y += 4;
    }
    if n < 32 { None } else { Some(sad / n as f32) }
}

fn canvas_luma(acc: &[f32], written: &[u8], fw: usize, x: i32, y: i32) -> Option<f32> {
    if x < 0 || y < 0 {
        return None;
    }
    let x = x as usize;
    let y = y as usize;
    if x >= fw || y >= written.len() / fw || written[y * fw + x] == 0 {
        return None;
    }
    Some(luma(&acc[(y * fw + x) * 3..(y * fw + x) * 3 + 3]))
}

fn placed_luma(placed: &Placed, x: i32, y: i32) -> Option<f32> {
    let px = x - placed.x0;
    let py = y - placed.y0;
    if px < 0 || py < 0 || px >= placed.w as i32 || py >= placed.h as i32 {
        return None;
    }
    let s = py as usize * placed.w + px as usize;
    if !placed.valid[s] {
        return None;
    }
    Some(luma(&placed.img[s * 3..s * 3 + 3]))
}

fn blur_residual(cells: &[Residual], cw: usize, ch: usize) -> Vec<Residual> {
    let mut out = cells.to_vec();
    for y in 0..ch {
        for x in 0..cw {
            let mut dx = 0.0f32;
            let mut dy = 0.0f32;
            let mut gain = 0.0f32;
            let mut n = 0.0f32;
            for oy in -1i32..=1 {
                for ox in -1i32..=1 {
                    let yy = y as i32 + oy;
                    let xx = x as i32 + ox;
                    if yy < 0 || xx < 0 || yy >= ch as i32 || xx >= cw as i32 {
                        continue;
                    }
                    let c = cells[yy as usize * cw + xx as usize];
                    dx += c.dx;
                    dy += c.dy;
                    gain += c.gain;
                    n += 1.0;
                }
            }
            out[y * cw + x] = Residual { dx: dx / n, dy: dy / n, gain: (gain / n).clamp(0.9, 1.1) };
        }
    }
    out
}

fn sample_residual(cells: &[Residual], cw: usize, ch: usize, x0: usize, y0: usize, x: usize, y: usize) -> (f32, f32, f32) {
    let fx = ((x as f32 - x0 as f32) / 128.0 - 0.5).clamp(0.0, (cw.saturating_sub(1)) as f32);
    let fy = ((y as f32 - y0 as f32) / 128.0 - 0.5).clamp(0.0, (ch.saturating_sub(1)) as f32);
    let x0i = fx.floor() as usize;
    let y0i = fy.floor() as usize;
    let x1i = (x0i + 1).min(cw - 1);
    let y1i = (y0i + 1).min(ch - 1);
    let tx = fx - x0i as f32;
    let ty = fy - y0i as f32;
    let mix = |a: Residual, b: Residual, t: f32| Residual {
        dx: a.dx + (b.dx - a.dx) * t,
        dy: a.dy + (b.dy - a.dy) * t,
        gain: a.gain + (b.gain - a.gain) * t,
    };
    let top = mix(cells[y0i * cw + x0i], cells[y0i * cw + x1i], tx);
    let bot = mix(cells[y1i * cw + x0i], cells[y1i * cw + x1i], tx);
    let c = mix(top, bot, ty);
    (c.dx, c.dy, c.gain)
}

fn sample_placed(img: &[f32], valid: &[bool], w: usize, h: usize, x: f32, y: f32) -> Option<[f32; 3]> {
    if x < 0.0 || y < 0.0 || x + 1.0 >= w as f32 || y + 1.0 >= h as f32 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let i00 = y0 * w + x0;
    let i10 = i00 + 1;
    let i01 = i00 + w;
    let i11 = i01 + 1;
    if !valid[i00] || !valid[i10] || !valid[i01] || !valid[i11] {
        return None;
    }
    let take = |i: usize| [img[i * 3], img[i * 3 + 1], img[i * 3 + 2]];
    let a = take(i00);
    let b = take(i10);
    let c = take(i01);
    let d = take(i11);
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        let top = a[k] + (b[k] - a[k]) * tx;
        let bot = c[k] + (d[k] - c[k]) * tx;
        out[k] = top + (bot - top) * ty;
    }
    Some(out)
}

fn flat(here: f32, left: f32, up: f32, quiet: f32) -> bool {
    let tex = ((here - left).abs() + (here - up).abs()).min(0.5) * 2.0;
    tex < quiet
}

fn full_cut(acc: &[f32], written: &[u8], placed: &Placed, fw: usize, fh: usize, id: u16) -> Option<Cut> {
    let y0s = placed.y0.max(0) as usize;
    let y1s = (placed.y0 + placed.h as i32).clamp(0, fh as i32) as usize;
    let x0s = placed.x0.max(0) as usize;
    let x1s = (placed.x0 + placed.w as i32).clamp(0, fw as i32) as usize;
    let mut x0 = usize::MAX;
    let mut y0 = usize::MAX;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    for y in y0s..y1s {
        let py = y as i32 - placed.y0;
        for x in x0s..x1s {
            let px = x as i32 - placed.x0;
            let s = py as usize * placed.w + px as usize;
            if written[y * fw + x] == 0 || !placed.valid[s] {
                continue;
            }
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let w = x1 - x0;
    let h = y1 - y0;
    let mut la = vec![0f32; w * h];
    let mut lb = vec![0f32; w * h];
    let mut va = vec![false; w * h];
    let mut vb = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let cx = x0 + x;
            let cy = y0 + y;
            let i = y * w + x;
            if written[cy * fw + cx] != 0 {
                let s = (cy * fw + cx) * 3;
                la[i] = luma(&acc[s..s + 3]);
                va[i] = true;
            }
            let px = cx as i32 - placed.x0;
            let py = cy as i32 - placed.y0;
            if px >= 0 && py >= 0 && px < placed.w as i32 && py < placed.h as i32 {
                let s = py as usize * placed.w + px as usize;
                if placed.valid[s] {
                    let p = s * 3;
                    lb[i] = luma(&placed.img[p..p + 3]);
                    vb[i] = true;
                }
            }
        }
    }
    log_span(&va, &vb, w, h, x0, id);
    Some(Cut { x0, y0, w, h, mask: graphcut_mask(&la, &lb, &va, &vb, w, h) })
}

fn log_span(va: &[bool], vb: &[bool], w: usize, h: usize, origin_x: usize, id: u16) {
    for frac in [0.15f64, 0.85] {
        let y = ((h as f64 - 1.0) * frac).round() as usize;
        let row = y * w;
        let mut x0 = None;
        let mut x1 = 0usize;
        for x in 0..w {
            if va[row + x] && vb[row + x] {
                if x0.is_none() {
                    x0 = Some(x);
                }
                x1 = x + 1;
            }
        }
        match x0 {
            Some(x0) => trace::line(&format!("seam span {id} y={:.0}% x0={} x1={} w={}", frac * 100.0, origin_x + x0, origin_x + x1, x1 - x0)),
            None => trace::line(&format!("seam span {id} y={:.0}% none", frac * 100.0)),
        }
    }
}

fn fill_distance(dist: &mut [u16], owners: &[u16], fw: usize, fh: usize, x0: usize, y0: usize, x1: usize, y1: usize, cap: u16) {
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    for y in y0..y1 {
        for x in x0..x1 {
            dist[y * fw + x] = u16::MAX;
        }
    }
    let mut queue = std::collections::VecDeque::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let id = owners[y * fw + x];
            if id == u16::MAX {
                continue;
            }
            let differ = [(1isize, 0isize), (0, 1)].into_iter().any(|(dy, dx)| {
                let yy = y as isize + dy;
                let xx = x as isize + dx;
                if yy < 0 || xx < 0 || yy >= fh as isize || xx >= fw as isize {
                    return false;
                }
                let other = owners[yy as usize * fw + xx as usize];
                other != u16::MAX && other != id
            });
            if differ {
                let i = y * fw + x;
                dist[i] = 0;
                queue.push_back(i);
            }
        }
    }
    while let Some(i) = queue.pop_front() {
        let d = dist[i];
        if d >= cap {
            continue;
        }
        let x = i % fw;
        let y = i / fw;
        for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let xx = x as isize + dx;
            let yy = y as isize + dy;
            if xx < x0 as isize || yy < y0 as isize || xx >= x1 as isize || yy >= y1 as isize {
                continue;
            }
            let j = yy as usize * fw + xx as usize;
            if dist[j] > d + 1 {
                dist[j] = d + 1;
                queue.push_back(j);
            }
        }
    }
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

fn dist_at(dist: &[u16], preview_w: u32, preview_h: u32, full_w: u32, full_h: u32, x: i32, y: i32) -> u16 {
    let px = (x as f64 * preview_w as f64 / full_w as f64).round() as i32;
    let py = (y as f64 * preview_h as f64 / full_h as f64).round() as i32;
    if px < 0 || py < 0 || px >= preview_w as i32 || py >= preview_h as i32 {
        return u16::MAX;
    }
    dist[py as usize * preview_w as usize + px as usize]
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
