use super::align::{choose_projection, Alignment};
use image::imageops::FilterType;
use image::{ImageBuffer, ImageFormat, Rgb, RgbImage};
use rapidraw_lib::panorama_stitching::{self, exposure_gains};
use rapidraw_lib::panorama_utils::v86::local_align::{self, RecordedShift};
use rapidraw_lib::panorama_utils::v86::photo;
use rapidraw_lib::panorama_utils::v86::stitch::{self, InputFrame};
use rapidraw_lib::panorama_utils::v86::trace;
use rapidraw_lib::panorama_utils::v86::warp::{self, warp_one_nudged};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn render(paths: &[String], alignment: &Alignment, tmp: &Path, tiff_path: &Path, view_path: &Path) -> Result<(), String> {
    let mut order: Vec<usize> = alignment.kept.clone();
    order.sort_unstable();
    let rots: Vec<_> = order.iter().map(|&i| alignment.rots[i]).collect();
    let mut stored = Vec::new();
    let mut evs = Vec::new();
    for &index in &order {
        let frame = panorama_stitching::develop_panorama_frame(&paths[index])?;
        evs.push(frame.exposure_ev);
        let file = tmp.join(format!("{:02}-{}.f32", stored.len(), frame.name));
        write_f32(&file, frame.width, frame.height, &frame.rgb)?;
        let (dw, dh) = fit_long_side(frame.width, frame.height, 1920);
        let small = shrink(&frame.rgb, frame.width, frame.height, dw, dh);
        trace::line(&format!("developed {} {}x{} -> {dw}x{dh}", frame.name, frame.width, frame.height));
        stored.push(Stored { file, width: frame.width, height: frame.height, preview: InputFrame { name: frame.name, width: dw, height: dh, rgb: small } });
    }
    let gains = exposure_gains(&evs);
    for (item, gain) in stored.iter_mut().zip(gains.iter()) {
        if (*gain - 1.0).abs() > 1e-4 {
            for p in &mut item.preview.rgb {
                *p *= *gain;
            }
        }
    }
    let raw_w = stored[0].width;
    let raw_h = stored[0].height;
    let sx = raw_w as f64 / alignment.jpeg_w as f64;
    let sy = raw_h as f64 / alignment.jpeg_h as f64;
    trace::line(&format!("scale raw={raw_w}x{raw_h} jpeg={}x{} x={sx:.4} y={sy:.4}", alignment.jpeg_w, alignment.jpeg_h));
    if (sx - sy).abs() > 0.03 * sx.max(sy) {
        return Err(format!("embedded jpeg does not line up with the developed frame ({sx:.3} vs {sy:.3})"));
    }
    let focal_full = alignment.focal_jpeg * sx;
    let pw = stored[0].preview.width;
    let focal_preview = focal_full * (pw as f64 / raw_w as f64);
    let projection = choose_projection(&rots, focal_full, raw_w, raw_h, &alignment.lens);
    let frames: Vec<InputFrame> = stored.iter_mut().map(|item| InputFrame {
        name: item.preview.name.clone(),
        width: item.preview.width,
        height: item.preview.height,
        rgb: std::mem::take(&mut item.preview.rgb),
    }).collect();
    let kept: Vec<usize> = (0..frames.len()).collect();
    local_align::begin_shift_log();
    let planned = stitch::compose(&frames, &rots, &kept, alignment.lens, focal_preview, alignment.focal35, projection, false, &|message| {
        trace::line(message);
    })?;
    let shifts = local_align::end_shift_log();
    trace::line(&format!("plan {}x{} shifts={}", planned.width, planned.height, shifts.len()));
    let (geom, _) = warp::canvas_geom(&rots, focal_full, raw_w, raw_h, &alignment.lens, planned.used);
    trace::line(&format!("canvas {} {}x{}", planned.used.as_str(), geom.width, geom.height));
    let fw = geom.width as usize;
    let fh = geom.height as usize;
    let mut acc = vec![0f32; fw * fh * 3];
    for (bi, &frame_i) in planned.kept.iter().enumerate() {
        trace::gate()?;
        let item = &stored[frame_i];
        let mut rgb = read_f32(&item.file)?;
        let gain = gains[frame_i];
        if (gain - 1.0).abs() > 1e-4 {
            for p in &mut rgb {
                *p *= gain;
            }
        }
        warp::devignette(&mut rgb, item.width, item.height, &alignment.lens);
        let shift = if bi == 0 { None } else { shifts.get(bi - 1).and_then(|entry| entry.as_ref()) };
        let nudge = |x: f64, y: f64| sample_nudge(shift, x, y, planned.width, planned.height, geom.width, geom.height);
        let mut placed = warp_one_nudged(&rgb, item.width, item.height, &rots[frame_i], focal_full, &alignment.lens, &geom, &nudge);
        drop(rgb);
        let _ = std::fs::remove_file(&item.file);
        photo::apply_photometry(&mut placed, &planned.photo, bi);
        let id = bi as u16;
        for y in 0..placed.h {
            for x in 0..placed.w {
                let s = y * placed.w + x;
                if !placed.valid[s] {
                    continue;
                }
                let cx = placed.x0 + x as i32;
                let cy = placed.y0 + y as i32;
                if cx < 0 || cy < 0 || cx as usize >= fw || cy as usize >= fh {
                    continue;
                }
                let winner = winner_at(&planned.winners, planned.width, planned.height, geom.width, geom.height, cx, cy);
                if winner == id {
                    let d = (cy as usize * fw + cx as usize) * 3;
                    acc[d] = placed.img[s * 3];
                    acc[d + 1] = placed.img[s * 3 + 1];
                    acc[d + 2] = placed.img[s * 3 + 2];
                }
            }
        }
        trace::line(&format!("warped {} {}", item.preview.name, if shift.is_some() { "nudged" } else { "straight" }));
    }
    mix_seam(&mut acc, fw, fh, &planned.rgb, planned.width, planned.height, &planned.winners);
    let cropped = crop(&acc, geom.width, geom.height, planned.crop_x, planned.crop_y, planned.crop_w, planned.crop_h);
    trace::line(&format!("save encode {}x{}", cropped.1, cropped.2));
    panorama_stitching::write_panorama_tiff(&cropped.0, cropped.1, cropped.2, tiff_path)?;
    write_view(&cropped.0, cropped.1, cropped.2, view_path)?;
    Ok(())
}

struct Stored {
    file: PathBuf,
    width: u32,
    height: u32,
    preview: InputFrame,
}

fn sample_nudge(shift: Option<&RecordedShift>, x: f64, y: f64, preview_w: u32, preview_h: u32, full_w: u32, full_h: u32) -> (f64, f64) {
    let Some(shift) = shift else {
        return (0.0, 0.0);
    };
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
    let at = |yy: usize, xx: usize, src: &[f32]| src[yy * shift.w + xx];
    let mix = |src: &[f32]| {
        let a = at(y0, x0, src) * (1.0 - tx) + at(y0, x0 + 1, src) * tx;
        let b = at(y0 + 1, x0, src) * (1.0 - tx) + at(y0 + 1, x0 + 1, src) * tx;
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

fn mix_seam(acc: &mut [f32], fw: usize, fh: usize, preview: &[f32], pw: u32, ph: u32, winners: &[u16]) {
    let dist = seam_distance(winners, pw as usize, ph as usize);
    let up = upscale(preview, pw, ph, fw as u32, fh as u32);
    let soft = blur3(acc, fw, fh);
    let band = 96.0f64;
    let sx = fw as f64 / pw.max(1) as f64;
    for y in 0..fh {
        for x in 0..fw {
            let px = (x as f64 * pw as f64 / fw as f64).round() as usize;
            let py = (y as f64 * ph as f64 / fh as f64).round() as usize;
            if px >= pw as usize || py >= ph as usize {
                continue;
            }
            let d = dist[py * pw as usize + px];
            if d == u16::MAX {
                continue;
            }
            let t = (1.0 - (d as f64 * sx) / band).clamp(0.0, 1.0) as f32;
            if t <= 0.0 {
                continue;
            }
            let i = (y * fw + x) * 3;
            for c in 0..3 {
                let sharp = acc[i + c];
                let detail = sharp - soft[i + c];
                let blended = up[i + c] + detail * (1.0 - t);
                acc[i + c] = sharp * (1.0 - t) + blended * t;
            }
        }
    }
    trace::line("seam mixed");
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
        if d >= 48 {
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

fn upscale(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    let image = ImageBuffer::<Rgb<f32>, _>::from_raw(sw, sh, src.to_vec()).unwrap_or_else(|| ImageBuffer::new(sw, sh));
    image::imageops::resize(&image, dw, dh, FilterType::Triangle).into_raw()
}

fn blur3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut s = 0.0f32;
                for k in -1..=1 {
                    let xx = (x as i32 + k).clamp(0, w as i32 - 1) as usize;
                    s += src[(y * w + xx) * 3 + c];
                }
                tmp[(y * w + x) * 3 + c] = s / 3.0;
            }
        }
    }
    let mut dst = vec![0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut s = 0.0f32;
                for k in -1..=1 {
                    let yy = (y as i32 + k).clamp(0, h as i32 - 1) as usize;
                    s += tmp[(yy * w + x) * 3 + c];
                }
                dst[(y * w + x) * 3 + c] = s / 3.0;
            }
        }
    }
    dst
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

fn write_view(rgb: &[f32], w: u32, h: u32, path: &Path) -> Result<(), String> {
    let (dw, dh) = fit_long_side(w, h, 1600);
    let small = shrink(rgb, w, h, dw, dh);
    let mut img: RgbImage = ImageBuffer::new(dw, dh);
    for (i, pixel) in img.pixels_mut().enumerate() {
        let r = to_srgb(small[i * 3]);
        let g = to_srgb(small[i * 3 + 1]);
        let b = to_srgb(small[i * 3 + 2]);
        *pixel = Rgb([r, g, b]);
    }
    img.save_with_format(path, ImageFormat::Jpeg).map_err(|e| format!("Failed to save the view: {e}"))
}

fn to_srgb(x: f32) -> u8 {
    let x = x.max(0.0);
    let y = if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (y.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn shrink(rgb: &[f32], w: u32, h: u32, dw: u32, dh: u32) -> Vec<f32> {
    if dw == w && dh == h {
        return rgb.to_vec();
    }
    let image = ImageBuffer::<Rgb<f32>, _>::from_raw(w, h, rgb.to_vec()).unwrap_or_else(|| ImageBuffer::new(w, h));
    image::imageops::resize(&image, dw, dh, FilterType::Triangle).into_raw()
}

fn fit_long_side(w: u32, h: u32, long_side: u32) -> (u32, u32) {
    let long = w.max(h).max(1);
    if long <= long_side {
        return (w.max(1), h.max(1));
    }
    let scale = long_side as f64 / long as f64;
    (((w as f64) * scale).round().max(1.0) as u32, ((h as f64) * scale).round().max(1.0) as u32)
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
