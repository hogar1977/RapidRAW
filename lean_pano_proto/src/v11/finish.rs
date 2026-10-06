use super::align::{self, Preview};
use rapidraw_lib::panorama_stitching::{self, exposure_gains};
use rapidraw_lib::panorama_utils::v86::blend::{blur_mask, pyr_blend};
use rapidraw_lib::panorama_utils::v86::local_align::{self, RecordedShift};
use rapidraw_lib::panorama_utils::v86::photo;
use rapidraw_lib::panorama_utils::v86::stitch::{self, InputFrame};
use rapidraw_lib::panorama_utils::v86::trace;
use rapidraw_lib::panorama_utils::v86::warp::{self, warp_one_nudged, Placed};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const BAND_PX: f64 = 192.0;

pub fn render(paths: &[String], preview: &Preview, tmp: &Path, tiff_path: &Path, view_path: &Path) -> Result<(), String> {
    let dw = preview.frames[0].width;
    let dh = preview.frames[0].height;
    let mut stored = Vec::new();
    let mut evs = Vec::new();
    for &index in &preview.indices {
        let frame = panorama_stitching::develop_panorama_frame(&paths[index])?;
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
    let developed_shifts = local_align::end_shift_log();
    drop(linear_frames);
    let use_linear = linear.kept == preview.result.kept;
    let same_canvas = linear.width == preview.result.width && linear.height == preview.result.height;
    let use_developed = use_linear && same_canvas;
    if !use_linear {
        trace::line("linear order differs; keeping the preview photometry");
    }
    trace::line(if use_developed { "save seam developed" } else { "save seam preview" });
    trace::line(&format!("linear gains={}", linear.photo.gains.len()));
    let (winners, seam_w, seam_h, shifts, crop_x, crop_y, crop_w, crop_h) = if use_developed {
        (&linear.winners, linear.width, linear.height, &developed_shifts, linear.crop_x, linear.crop_y, linear.crop_w, linear.crop_h)
    } else {
        (
            &preview.result.winners,
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
    let dist = seam_distance(winners, seam_w as usize, seam_h as usize);
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
        warp::devignette(&mut rgb, item.width, item.height, &preview.lens);
        let shift = if bi == 0 { None } else { shifts.get(bi - 1).and_then(|entry| entry.as_ref()) };
        let nudge = |x: f64, y: f64| sample_nudge(shift, x, y, seam_w, seam_h, geom.width, geom.height);
        let mut placed = warp_one_nudged(&rgb, item.width, item.height, &preview.rots[frame_i], focal_full, &preview.lens, &geom, &nudge);
        drop(rgb);
        let _ = std::fs::remove_file(&item.file);
        photo::apply_photometry(&mut placed, if use_linear { &linear.photo } else { &preview.result.photo }, bi);
        let id = bi as u16;
        paint_owned(&mut acc, &mut written, &placed, id, winners, seam_w, seam_h, fw, fh);
        if bi > 0 {
            blend_strip(&mut acc, &mut written, &placed, id, winners, &dist, seam_w, seam_h, fw, fh);
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

fn paint_owned(acc: &mut [f32], written: &mut [u8], placed: &Placed, id: u16, winners: &[u16], pw: u32, ph: u32, fw: usize, fh: usize) {
    let y0 = placed.y0.max(0) as usize;
    let y1 = (placed.y0 + placed.h as i32).clamp(0, fh as i32) as usize;
    if y1 <= y0 || placed.w == 0 {
        return;
    }
    let rows = y1 - y0;
    let (acc_rows, flag_rows) = (acc, written);
    acc_rows[y0 * fw * 3..y1 * fw * 3]
        .par_chunks_mut(fw * 3)
        .zip(flag_rows[y0 * fw..y1 * fw].par_chunks_mut(fw))
        .enumerate()
        .for_each(|(row, (dest, flags))| {
            if row >= rows {
                return;
            }
            let cy = (y0 + row) as i32;
            let py = cy - placed.y0;
            if py < 0 || py >= placed.h as i32 {
                return;
            }
            for x in 0..fw {
                let cx = x as i32;
                let px = cx - placed.x0;
                if px < 0 || px >= placed.w as i32 {
                    continue;
                }
                let s = py as usize * placed.w + px as usize;
                if !placed.valid[s] {
                    continue;
                }
                if winner_at(winners, pw, ph, fw as u32, fh as u32, cx, cy) != id {
                    continue;
                }
                dest[x * 3] = placed.img[s * 3];
                dest[x * 3 + 1] = placed.img[s * 3 + 1];
                dest[x * 3 + 2] = placed.img[s * 3 + 2];
                flags[x] = 1;
            }
        });
}

fn blend_strip(acc: &mut [f32], written: &mut [u8], placed: &Placed, id: u16, winners: &[u16], dist: &[u16], pw: u32, ph: u32, fw: usize, fh: usize) {
    let sx = fw as f64 / pw.max(1) as f64;
    let sy = fh as f64 / ph.max(1) as f64;
    let band_preview = ((BAND_PX / sx).ceil() as u16).clamp(2, 80);
    let repeats = sx.max(sy).round().max(1.0) as usize;
    let slab_h = 256usize;
    let nslabs = fh.div_ceil(slab_h);
    let pw_us = pw as usize;
    let ph_us = ph as usize;
    let mut cols: Vec<Vec<u32>> = vec![Vec::new(); nslabs];
    for py in 0..ph_us {
        let row = py * pw_us;
        let fy0 = ((py as f64) * sy).floor() as usize;
        let fy1 = (((py + 1) as f64) * sy).ceil() as usize;
        let s_lo = (fy0 / slab_h).min(nslabs - 1);
        let s_hi = (fy1.saturating_sub(1) / slab_h).min(nslabs - 1);
        for px in 0..pw_us {
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
        trace::line(&format!("seam strip {id} slabs={slabs_used} px={pixels} blur={repeats}"));
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
    repeats: usize,
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
    let mut soft = mask;
    for _ in 0..repeats {
        soft = blur_mask(&soft, w, h);
    }
    let blended = pyr_blend(&a, &b, &soft, &va, &vb, w, h, levels_for(w, h));
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

fn seam_distance(winners: &[u16], w: usize, h: usize) -> Vec<u16> {
    let mut seam = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let id = winners[y * w + x];
            if id == u16::MAX {
                continue;
            }
            let differ = [(1isize, 0isize), (0, 1)].into_iter().any(|(dy, dx)| {
                let yy = y as isize + dy;
                let xx = x as isize + dx;
                if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                    return false;
                }
                let other = winners[yy as usize * w + xx as usize];
                other != u16::MAX && other != id
            });
            if differ {
                seam[y * w + x] = true;
            }
        }
    }
    let mut dist = vec![u16::MAX; w * h];
    let mut queue = std::collections::VecDeque::new();
    for (i, on) in seam.iter().enumerate() {
        if *on {
            dist[i] = 0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let d = dist[i];
        if d >= 80 {
            continue;
        }
        let x = i % w;
        let y = i / w;
        for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let xx = x as isize + dx;
            let yy = y as isize + dy;
            if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                continue;
            }
            let j = yy as usize * w + xx as usize;
            if dist[j] > d + 1 {
                dist[j] = d + 1;
                queue.push_back(j);
            }
        }
    }
    dist
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
