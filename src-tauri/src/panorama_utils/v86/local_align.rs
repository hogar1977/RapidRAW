use rayon::prelude::*;
use std::cell::RefCell;

pub struct RecordedShift {
    pub x: i32,
    pub y: i32,
    pub w: usize,
    pub h: usize,
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
}

thread_local! {
    static RECORD_SHIFTS: RefCell<bool> = const { RefCell::new(false) };
    static STAGED_SHIFT: RefCell<Option<RecordedShift>> = const { RefCell::new(None) };
    static SHIFT_LOG: RefCell<Vec<Option<RecordedShift>>> = const { RefCell::new(Vec::new()) };
}

pub fn begin_shift_log() {
    RECORD_SHIFTS.with(|flag| *flag.borrow_mut() = true);
    STAGED_SHIFT.with(|slot| *slot.borrow_mut() = None);
    SHIFT_LOG.with(|log| log.borrow_mut().clear());
}

pub fn end_shift_log() -> Vec<Option<RecordedShift>> {
    RECORD_SHIFTS.with(|flag| *flag.borrow_mut() = false);
    SHIFT_LOG.with(|log| std::mem::take(&mut *log.borrow_mut()))
}

pub fn commit_shift(origin_x: i32, origin_y: i32) {
    let recording = RECORD_SHIFTS.with(|flag| *flag.borrow());
    if !recording {
        return;
    }
    let staged = STAGED_SHIFT.with(|slot| slot.borrow_mut().take());
    SHIFT_LOG.with(|log| {
        log.borrow_mut().push(staged.map(|shift| RecordedShift {
            x: origin_x + shift.x,
            y: origin_y + shift.y,
            w: shift.w,
            h: shift.h,
            dx: shift.dx,
            dy: shift.dy,
        }));
    });
}

fn clear_staged_shift() {
    STAGED_SHIFT.with(|slot| *slot.borrow_mut() = None);
}

fn stage_shift(x0: usize, y0: usize, x1: usize, y1: usize, field_x: &[f32], field_y: &[f32], w: usize) {
    let recording = RECORD_SHIFTS.with(|flag| *flag.borrow());
    if !recording || x1 <= x0 || y1 <= y0 {
        return;
    }
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut dx = vec![0f32; cw * ch];
    let mut dy = vec![0f32; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let s = (y0 + y) * w + x0 + x;
            dx[y * cw + x] = field_x[s];
            dy[y * cw + x] = field_y[s];
        }
    }
    STAGED_SHIFT.with(|slot| {
        *slot.borrow_mut() = Some(RecordedShift { x: x0 as i32, y: y0 as i32, w: cw, h: ch, dx, dy });
    });
}

// Nudges the new photo so its overlap sits on the picture already built.
pub fn align_incoming(
    lum_a: &[f32],
    lum_b: &[f32],
    va: &[bool],
    vb: Vec<bool>,
    img: Vec<f32>,
    w: usize,
    h: usize,
) -> (Vec<f32>, Vec<bool>) {
    clear_staged_shift();
    let tile = 128usize;
    let both: Vec<bool> = va.iter().zip(vb.iter()).map(|(a, b)| *a && *b).collect();
    if both.iter().filter(|v| **v).count() < tile * tile {
        super::trace::line("local copied overlap");
        return (img, vb);
    }
    let stride = tile / 2;
    let max_shift = (tile / 4) as i32;
    let sqrt_a: Vec<f32> = lum_a.par_iter().map(|v| v.max(0.0).sqrt()).collect();
    let sqrt_b: Vec<f32> = lum_b.par_iter().map(|v| v.max(0.0).sqrt()).collect();
    let (y0, y1, x0, x1) = overlap_box(&both, w, h);
    if !sparse_disagrees(&sqrt_a, &sqrt_b, &both, w, h, y0, y1, x0, x1, tile, max_shift) {
        super::trace::line("local copied sparse");
        return (img, vb);
    }
    let cys = steps(y0, (y1.saturating_sub(tile)).max(y0), stride);
    let cxs = steps(x0, (x1.saturating_sub(tile)).max(x0), stride);
    let gy = cys.len();
    let gx = cxs.len();
    let rows: Vec<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> = cys
        .par_iter()
        .map(|&ty| {
            if super::trace::halted() {
                return None;
            }
            let mut fx = vec![0f32; gx];
            let mut fy = vec![0f32; gx];
            let mut wt = vec![0f32; gx];
            for (j, &tx) in cxs.iter().enumerate() {
                if ty + tile > h || tx + tile > w {
                    continue;
                }
                let mut cover = 0.0f32;
                let mut mean_a = 0.0f32;
                let mut mean_b = 0.0f32;
                let n = (tile * tile) as f32;
                for y in 0..tile {
                    for x in 0..tile {
                        let p = (ty + y) * w + tx + x;
                        if both[p] {
                            cover += 1.0;
                        }
                        mean_a += sqrt_a[p];
                        mean_b += sqrt_b[p];
                    }
                }
                if cover / n < 0.9 {
                    continue;
                }
                mean_a /= n;
                mean_b /= n;
                let (sa, sb) = tile_std(&sqrt_a, &sqrt_b, w, ty, tx, tile, mean_a, mean_b);
                if sa < 0.01 || sb < 0.01 {
                    continue;
                }
                let (dx, dy, resp) = phase(&sqrt_a, &sqrt_b, w, ty, tx, tile);
                if dx.abs() > max_shift as f32 || dy.abs() > max_shift as f32 || resp < 0.10 {
                    continue;
                }
                let (sad0, sad1, inside) = sad_pair(&sqrt_a, &sqrt_b, w, ty, tx, tile, max_shift as usize, dx, dy, &both);
                if inside < 64 {
                    continue;
                }
                let ratio = sad1 / sad0.max(1e-6);
                if dx.hypot(dy) < 0.75 {
                    wt[j] = resp * 0.15;
                    continue;
                }
                if ratio > 0.8 {
                    continue;
                }
                fx[j] = dx;
                fy[j] = dy;
                wt[j] = resp * (1.0 - ratio).max(0.05);
            }
            Some((fx, fy, wt))
        })
        .collect();
    if rows.iter().any(|row| row.is_none()) {
        super::trace::line("local copied stopped");
        return (img, vb);
    }
    let mut fx = vec![0f32; gy * gx];
    let mut fy = vec![0f32; gy * gx];
    let mut wt = vec![0f32; gy * gx];
    for (i, row) in rows.into_iter().enumerate() {
        let (rx, ry, rw) = row.unwrap();
        fx[i * gx..(i + 1) * gx].copy_from_slice(&rx);
        fy[i * gx..(i + 1) * gx].copy_from_slice(&ry);
        wt[i * gx..(i + 1) * gx].copy_from_slice(&rw);
    }
    let kept = wt.iter().filter(|v| **v > 0.0).count();
    // v39 DIAGNOSTIC (debug only, inert unless LEAN_ALIGN_DUMP=1): dump the
    // per-tile votes and the smoothed field so a fitted field that ramps across
    // the window can be traced to either thin tile coverage (wt -> 0) or to the
    // ridge term shrinking unmeasured cells toward zero.
    if std::env::var("LEAN_ALIGN_DUMP").is_ok() {
        let cys_txt: Vec<String> = cys.iter().map(|v| v.to_string()).collect();
        let cxs_txt: Vec<String> = cxs.iter().map(|v| v.to_string()).collect();
        super::trace::line(&format!("DUMP grid gy={gy} gx={gx} kept={kept} total={}", gy * gx));
        super::trace::line(&format!("DUMP cys=[{}]", cys_txt.join(",")));
        super::trace::line(&format!("DUMP cxs=[{}]", cxs_txt.join(",")));
        super::trace::line(&format!("DUMP stride={stride} tile={tile} max_shift={max_shift}"));
        for i in 0..gy {
            let row_wt: Vec<String> = (0..gx).map(|j| format!("{:.3}", wt[i * gx + j])).collect();
            let row_fx: Vec<String> = (0..gx).map(|j| format!("{:.3}", fx[i * gx + j])).collect();
            super::trace::line(&format!("DUMP row {i} wt=[{}]", row_wt.join(",")));
            super::trace::line(&format!("DUMP row {i} fx=[{}]", row_fx.join(",")));
        }
    }
    if kept < 4 {
        super::trace::line(&format!("local copied too few kept={kept}"));
        return (img, vb);
    }
    let mut live: Vec<f32> = wt.iter().copied().filter(|v| *v > 0.0).collect();
    live.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lam = 0.25 * live[live.len() / 2];
    let num_x = blur_grid(&mul(&fx, &wt), gx, gy, 1.0);
    let num_y = blur_grid(&mul(&fy, &wt), gx, gy, 1.0);
    let den = blur_grid(&wt, gx, gy, 1.0);
    let mut gx_f = vec![0f32; gy * gx];
    let mut gy_f = vec![0f32; gy * gx];
    for i in 0..gx_f.len() {
        gx_f[i] = num_x[i] / (den[i] + lam);
        gy_f[i] = num_y[i] / (den[i] + lam);
    }
    if std::env::var("LEAN_ALIGN_DUMP").is_ok() {
        super::trace::line(&format!("DUMP lam={lam:.5} den_min={:.4} den_max={:.4}", {
            let mn = den.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = den.iter().cloned().fold(0.0f32, f32::max);
            let _ = (mn, mx);
            mn
        }, {
            den.iter().cloned().fold(0.0f32, f32::max)
        }));
        for i in 0..gy {
            let row_den: Vec<String> = (0..gx).map(|j| format!("{:.4}", den[i * gx + j])).collect();
            let row_fit: Vec<String> = (0..gx).map(|j| format!("{:.3}", gx_f[i * gx + j])).collect();
            super::trace::line(&format!("DUMP row {i} den=[{}]", row_den.join(",")));
            super::trace::line(&format!("DUMP row {i} fit=[{}]", row_fit.join(",")));
        }
    }
    // v51 DIAGNOSTIC (inert unless LEAN_ALIGN_RESID=1): for every measured
    // grid cell dump (a) the residual between the raw per-tile shift vote and
    // the smoothed+regularised fit, and (b) the local image gradient at that
    // cell centre. Parallax makes the true displacement field discontinuous at
    // depth boundaries, so if the smoothness fit is the defect the residual
    // should concentrate on high-gradient cells (building silhouettes, ridgeline)
    // and stay near zero inside a single depth layer.
    if std::env::var("LEAN_ALIGN_RESID").is_ok() {
        super::trace::line(&format!(
            "RESID win y0={y0} y1={y1} x0={x0} x1={x1} gy={gy} gx={gx} tile={tile} stride={stride} img={w}x{h}"
        ));
        let cys_txt: Vec<String> = cys.iter().map(|v| v.to_string()).collect();
        let cxs_txt: Vec<String> = cxs.iter().map(|v| v.to_string()).collect();
        super::trace::line(&format!("RESID cys=[{}]", cys_txt.join(",")));
        super::trace::line(&format!("RESID cxs=[{}]", cxs_txt.join(",")));
        for i in 0..gy {
            let cy = cys[i];
            let mut rr: Vec<String> = Vec::with_capacity(gx);
            let mut gg: Vec<String> = Vec::with_capacity(gx);
            let mut ff: Vec<String> = Vec::with_capacity(gx);
            let mut yy: Vec<String> = Vec::with_capacity(gx);
            let mut mm: Vec<String> = Vec::with_capacity(gx);
            let half = tile / 2;
            for j in 0..gx {
                let k = i * gx + j;
                let dxv = fx[k] - gx_f[k];
                let dyv = fy[k] - gy_f[k];
                let r = (dxv * dxv + dyv * dyv).sqrt();
                let cxj = cxs[j];
                let gmag = if cy >= 1 && cxj >= 1 && cy + 1 < h && cxj + 1 < w {
                    sobel_mag(&img, w, h, cy, cxj)
                } else {
                    0.0
                };
                // mean incoming intensity over the tile at this cell, so cells can
                // be split into near-structure vs far-background populations
                let ty0 = cy.saturating_sub(half);
                let tx0 = cxj.saturating_sub(half);
                let ty1 = (cy + half).min(h);
                let tx1 = (cxj + half).min(w);
                let mut acc = 0f64;
                let mut cnt = 0f64;
                for y2 in ty0..ty1 {
                    let base = y2 * w;
                    for x2 in tx0..tx1 {
                        acc += img[base + x2] as f64;
                        cnt += 1.0;
                    }
                }
                let mi = if cnt > 0.0 { (acc / cnt) as f32 } else { 0.0 };
                rr.push(format!("{:.4}", r));
                gg.push(format!("{:.4}", gmag));
                ff.push(format!("{:.4}", fx[k]));
                yy.push(format!("{:.4}", fy[k]));
                mm.push(format!("{:.2}", mi));
            }
            super::trace::line(&format!("RESID row {i} r=[{}]", rr.join(",")));
            super::trace::line(&format!("RESID row {i} g=[{}]", gg.join(",")));
            super::trace::line(&format!("RESID row {i} fx=[{}]", ff.join(",")));
            super::trace::line(&format!("RESID row {i} fy=[{}]", yy.join(",")));
            super::trace::line(&format!("RESID row {i} m=[{}]", mm.join(",")));
            let row_w: Vec<String> = (0..gx).map(|j| format!("{:.4}", wt[i * gx + j])).collect();
            super::trace::line(&format!("RESID row {i} w=[{}]", row_w.join(",")));
        }
    }
    let margin = tile * 2;
    let zy0 = y0.saturating_sub(margin);
    let zx0 = x0.saturating_sub(margin);
    let zy1 = (y1 + margin).min(h);
    let zx1 = (x1 + margin).min(w);
    let dist = distance_near_overlap(&both, w, h, zy0, zx0, zy1, zx1);
    let mut field_x = vec![0f32; w * h];
    let mut field_y = vec![0f32; w * h];
    let mut move_px = vec![false; w * h];
    let mut field_max = 0.0f32;
    for y in zy0..zy1 {
        for x in zx0..zx1 {
            let mapx = (x as f32 - (x0 as f32 + tile as f32 / 2.0)) / stride as f32;
            let mapy = (y as f32 - (y0 as f32 + tile as f32 / 2.0)) / stride as f32;
            let mut sx = sample_grid(&gx_f, gx, gy, mapx, mapy);
            let mut sy = sample_grid(&gy_f, gx, gy, mapx, mapy);
            let i = y * w + x;
            let taper = if vb[i] { (1.0 - dist[i] / tile as f32).clamp(0.0, 1.0) } else { 0.0 };
            sx = (sx * taper).clamp(-(max_shift as f32), max_shift as f32);
            sy = (sy * taper).clamp(-(max_shift as f32), max_shift as f32);
            field_x[i] = sx;
            field_y[i] = sy;
            let mag = sx.hypot(sy);
            field_max = field_max.max(mag);
            move_px[i] = mag > 0.2;
        }
    }
    if !move_px.iter().any(|v| *v) {
        super::trace::line(&format!("local copied under 0.2 kept={kept} field={field_max:.2}"));
        return (img, vb);
    }
    let mut out = img.to_vec();
    let mut ov = vb.to_vec();
    for y in zy0..zy1 {
        for x in zx0..zx1 {
            let i = y * w + x;
            if !move_px[i] {
                continue;
            }
            let sx = x as f32 + field_x[i];
            let sy = y as f32 + field_y[i];
            let pix = cubic_rgb(&img, w, h, sx, sy);
            out[i * 3] = pix[0];
            out[i * 3 + 1] = pix[1];
            out[i * 3 + 2] = pix[2];
            ov[i] = nearest_valid(&vb, w, h, sx, sy);
        }
    }
    let (gain, d0, d1) = improved(lum_a, lum_b, &out, &both, &ov, w, h);
    if !gain {
        super::trace::line(&format!("local copied no gain kept={kept} field={field_max:.2} disagree {d0:.4} -> {d1:.4}"));
        return (img, vb);
    }
    super::trace::line(&format!("local moved kept={kept} field={field_max:.2} disagree {d0:.4} -> {d1:.4}"));
    stage_shift(zx0, zy0, zx1, zy1, &field_x, &field_y, w);
    (out, ov)
}

fn overlap_box(both: &[bool], w: usize, h: usize) -> (usize, usize, usize, usize) {
    let mut y0 = h;
    let mut y1 = 0usize;
    let mut x0 = w;
    let mut x1 = 0usize;
    for y in 0..h {
        for x in 0..w {
            if both[y * w + x] {
                y0 = y0.min(y);
                y1 = y1.max(y + 1);
                x0 = x0.min(x);
                x1 = x1.max(x + 1);
            }
        }
    }
    (y0, y1, x0, x1)
}

fn steps(start: usize, stop: usize, stride: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut i = start;
    while i <= stop {
        v.push(i);
        i += stride.max(1);
    }
    if v.is_empty() {
        v.push(start);
    }
    v
}

// Samples a few tiles and reports whether any of them still wants to move.
fn sparse_disagrees(
    sqrt_a: &[f32],
    sqrt_b: &[f32],
    both: &[bool],
    w: usize,
    h: usize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
    tile: usize,
    max_shift: i32,
) -> bool {
    let pys = lattice(y0, y1.saturating_sub(tile).max(y0));
    let pxs = lattice(x0, x1.saturating_sub(tile).max(x0));
    for &ty in &pys {
        for &tx in &pxs {
            if ty + tile > h || tx + tile > w {
                continue;
            }
            let (_dx, _dy, kind) = vote_tile(sqrt_a, sqrt_b, both, w, ty, tx, tile, max_shift);
            if kind == 2 {
                return true;
            }
        }
    }
    false
}

fn lattice(start: usize, stop: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let span = stop.saturating_sub(start);
    for i in 0..4 {
        let t = start + span * i / 3;
        let t = t.min(stop);
        if v.last().copied() != Some(t) {
            v.push(t);
        }
    }
    if v.is_empty() {
        v.push(start);
    }
    v
}

fn vote_tile(
    sqrt_a: &[f32],
    sqrt_b: &[f32],
    both: &[bool],
    w: usize,
    ty: usize,
    tx: usize,
    tile: usize,
    max_shift: i32,
) -> (f32, f32, u8) {
    let mut cover = 0.0f32;
    let mut mean_a = 0.0f32;
    let mut mean_b = 0.0f32;
    let n = (tile * tile) as f32;
    for y in 0..tile {
        for x in 0..tile {
            let p = (ty + y) * w + tx + x;
            if both[p] {
                cover += 1.0;
            }
            mean_a += sqrt_a[p];
            mean_b += sqrt_b[p];
        }
    }
    if cover / n < 0.9 {
        return (0.0, 0.0, 0);
    }
    mean_a /= n;
    mean_b /= n;
    let (sa, sb) = tile_std(sqrt_a, sqrt_b, w, ty, tx, tile, mean_a, mean_b);
    if sa < 0.01 || sb < 0.01 {
        return (0.0, 0.0, 0);
    }
    let (dx, dy, resp) = phase(sqrt_a, sqrt_b, w, ty, tx, tile);
    if dx.abs() > max_shift as f32 || dy.abs() > max_shift as f32 || resp < 0.10 {
        return (0.0, 0.0, 0);
    }
    let (sad0, sad1, inside) = sad_pair(sqrt_a, sqrt_b, w, ty, tx, tile, max_shift as usize, dx, dy, both);
    if inside < 64 {
        return (0.0, 0.0, 0);
    }
    let ratio = sad1 / sad0.max(1e-6);
    if dx.hypot(dy) < 0.75 {
        return (0.0, resp * 0.15, 1);
    }
    if ratio > 0.8 {
        return (0.0, 0.0, 0);
    }
    (dx, dy, 2)
}

fn tile_std(a: &[f32], b: &[f32], w: usize, y0: usize, x0: usize, tile: usize, mean_a: f32, mean_b: f32) -> (f32, f32) {
    let mut sa = 0.0f32;
    let mut sb = 0.0f32;
    let n = (tile * tile) as f32;
    for y in 0..tile {
        for x in 0..tile {
            let i = (y0 + y) * w + x0 + x;
            let da = a[i] - mean_a;
            let db = b[i] - mean_b;
            sa += da * da;
            sb += db * db;
        }
    }
    ((sa / n).sqrt(), (sb / n).sqrt())
}

fn sad_pair(
    a: &[f32],
    b: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    tile: usize,
    margin: usize,
    dx: f32,
    dy: f32,
    both: &[bool],
) -> (f32, f32, usize) {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut n = 0usize;
    for y in margin..tile - margin {
        for x in margin..tile - margin {
            let i = (y0 + y) * w + x0 + x;
            if !both[i] {
                continue;
            }
            let ta = a[i];
            let tb = b[i];
            let sx = x0 as f32 + x as f32 + dx;
            let sy = y0 as f32 + y as f32 + dy;
            let shifted = sample_scalar(b, w, sx, sy);
            s0 += (ta - tb).abs();
            s1 += (ta - shifted).abs();
            n += 1;
        }
    }
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    (s0 / n as f32, s1 / n as f32, n)
}

fn sample_scalar(src: &[f32], w: usize, x: f32, y: f32) -> f32 {
    let h = src.len() / w;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let at = |yy: i32, xx: i32| {
        let xx = xx.clamp(0, w as i32 - 1) as usize;
        let yy = yy.clamp(0, h as i32 - 1) as usize;
        src[yy * w + xx]
    };
    let a = at(y0, x0);
    let b = at(y0, x0 + 1);
    let c = at(y0 + 1, x0);
    let d = at(y0 + 1, x0 + 1);
    (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy
}

fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(p, q)| p * q).collect()
}

fn blur_grid(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let k = ((6.0 * sigma) as i32) | 1;
    let r = k / 2;
    let mut ker = vec![0f32; k as usize];
    let mut sum = 0.0f32;
    for i in 0..k {
        let x = (i - r) as f32;
        let v = (-0.5 * (x / sigma).powi(2)).exp();
        ker[i as usize] = v;
        sum += v;
    }
    for v in &mut ker {
        *v /= sum;
    }
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0f32;
            for i in 0..k {
                let xx = reflect(x as i32 + i - r, w);
                s += src[y * w + xx] * ker[i as usize];
            }
            tmp[y * w + x] = s;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0f32;
            for i in 0..k {
                let yy = reflect(y as i32 + i - r, h);
                s += tmp[yy * w + x] * ker[i as usize];
            }
            out[y * w + x] = s;
        }
    }
    out
}

fn reflect(i: i32, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n = n as i32;
    let mut x = i;
    while x < 0 || x >= n {
        if x < 0 {
            x = -x - 1;
        } else {
            x = 2 * n - x - 1;
        }
    }
    x as usize
}

fn reflect101(mut p: i32, len: i32) -> i32 {
    if len <= 1 {
        return 0;
    }
    while p < 0 || p >= len {
        if p < 0 {
            p = -p;
        } else {
            p = 2 * len - p - 2;
        }
    }
    p
}

// Blurs using only the given spread, the way a lone sigma is turned into a kernel.
fn gauss_sigma(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let mut k = (sigma * 8.0 + 1.0).round() as i32;
    k |= 1;
    if k < 1 {
        k = 1;
    }
    let r = k / 2;
    let mut ker = vec![0f32; k as usize];
    let mut sum = 0.0f32;
    for i in 0..k {
        let x = (i - r) as f32;
        let v = (-0.5 * (x / sigma).powi(2)).exp();
        ker[i as usize] = v;
        sum += v;
    }
    for v in &mut ker {
        *v /= sum;
    }
    super::sift::blur_with_kernel(src, w, h, &ker, super::sift::BlurEdge::Reflect)
}

fn sample_grid(src: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w.saturating_sub(1)) as f32);
    let y = y.clamp(0.0, (h.saturating_sub(1)) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let a = src[y0 * w + x0];
    let b = src[y0 * w + x1];
    let c = src[y1 * w + x0];
    let d = src[y1 * w + x1];
    (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy
}

// Distance from the overlap, using the same five-pixel steps as the prototype.
fn outside_distance(inside: &[bool], w: usize, h: usize) -> Vec<f32> {
    let hv = (1.0f32 * 65536.0).round() as u32;
    let diag = (1.4f32 * 65536.0).round() as u32;
    let long = (2.1969f32 * 65536.0).round() as u32;
    let dist_max = u32::MAX - long;
    let bw = w + 4;
    let mut tmp = vec![dist_max; bw * (h + 4)];
    for y in 0..h {
        let row = (y + 2) * bw + 2;
        for x in 0..w {
            let j = row + x;
            if inside[y * w + x] {
                tmp[j] = 0;
                continue;
            }
            let mut t0 = tmp[j - bw * 2 - 1].wrapping_add(long);
            let mut t = tmp[j - bw * 2 + 1].wrapping_add(long);
            if t0 > t {
                t0 = t;
            }
            t = tmp[j - bw - 2].wrapping_add(long);
            if t0 > t {
                t0 = t;
            }
            t = tmp[j - bw - 1].wrapping_add(diag);
            if t0 > t {
                t0 = t;
            }
            t = tmp[j - bw].wrapping_add(hv);
            if t0 > t {
                t0 = t;
            }
            t = tmp[j - bw + 1].wrapping_add(diag);
            if t0 > t {
                t0 = t;
            }
            t = tmp[j - bw + 2].wrapping_add(long);
            if t0 > t {
                t0 = t;
            }
            t = tmp[j - 1].wrapping_add(hv);
            if t0 > t {
                t0 = t;
            }
            tmp[j] = if t0 > dist_max { dist_max } else { t0 };
        }
    }
    let scale = 1.0 / 65536.0;
    let mut dist = vec![0f32; w * h];
    for y in (0..h).rev() {
        let row = (y + 2) * bw + 2;
        for x in (0..w).rev() {
            let j = row + x;
            let mut t0 = tmp[j];
            if t0 > hv {
                let mut t = tmp[j + bw * 2 + 1].wrapping_add(long);
                if t0 > t {
                    t0 = t;
                }
                t = tmp[j + bw * 2 - 1].wrapping_add(long);
                if t0 > t {
                    t0 = t;
                }
                t = tmp[j + bw + 2].wrapping_add(long);
                if t0 > t {
                    t0 = t;
                }
                t = tmp[j + bw + 1].wrapping_add(diag);
                if t0 > t {
                    t0 = t;
                }
                t = tmp[j + bw].wrapping_add(hv);
                if t0 > t {
                    t0 = t;
                }
                t = tmp[j + bw - 1].wrapping_add(diag);
                if t0 > t {
                    t0 = t;
                }
                t = tmp[j + bw - 2].wrapping_add(long);
                if t0 > t {
                    t0 = t;
                }
                t = tmp[j + 1].wrapping_add(hv);
                if t0 > t {
                    t0 = t;
                }
                tmp[j] = t0;
            }
            dist[y * w + x] = t0 as f32 * scale;
        }
    }
    dist
}

fn cubic_rgb(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let wx = cubic_w(x - x0 as f32);
    let wy = cubic_w(y - y0 as f32);
    let mut o = [0f32; 3];
    for ky in 0..4 {
        let yy = (y0 - 1 + ky as i32).clamp(0, h as i32 - 1) as usize;
        let wyy = wy[ky];
        for kx in 0..4 {
            let xx = (x0 - 1 + kx as i32).clamp(0, w as i32 - 1) as usize;
            let weight = wyy * wx[kx];
            let p = (yy * w + xx) * 3;
            o[0] += img[p] * weight;
            o[1] += img[p + 1] * weight;
            o[2] += img[p + 2] * weight;
        }
    }
    o
}

fn cubic_w(t: f32) -> [f32; 4] {
    let a = -0.5f32;
    let t2 = t * t;
    let t3 = t2 * t;
    [
        a * (t3 - 2.0 * t2 + t),
        (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0,
        -(a + 2.0) * t3 + (2.0 * a + 3.0) * t2 - a * t,
        a * (-t3 + t2),
    ]
}

fn sobel_mag(lum: &[f32], w: usize, h: usize, y: usize, x: usize) -> f32 {
    let at = |yy: i32, xx: i32| lum[reflect101(yy, h as i32) as usize * w + reflect101(xx, w as i32) as usize];
    let yy = y as i32;
    let xx = x as i32;
    let gx = -at(yy - 1, xx - 1) + at(yy - 1, xx + 1) - 2.0 * at(yy, xx - 1) + 2.0 * at(yy, xx + 1) - at(yy + 1, xx - 1)
        + at(yy + 1, xx + 1);
    let gy = -at(yy - 1, xx - 1) - 2.0 * at(yy - 1, xx) - at(yy - 1, xx + 1) + at(yy + 1, xx - 1) + 2.0 * at(yy + 1, xx)
        + at(yy + 1, xx + 1);
    gx.abs() + gy.abs()
}

fn percentile60(vals: &mut [f32]) -> f32 {
    let n = vals.len();
    let pos = 0.60 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = (pos - lo as f64) as f32;
    let order = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    if lo == hi {
        vals.select_nth_unstable_by(lo, order);
        return vals[lo];
    }
    vals.select_nth_unstable_by(hi, order);
    let hi_v = vals[hi];
    let lo_v = vals[..hi].iter().copied().max_by(order).unwrap_or(hi_v);
    lo_v * (1.0 - frac) + hi_v * frac
}

fn nearest_valid(valid: &[bool], w: usize, h: usize, x: f32, y: f32) -> bool {
    let xx = x.round() as i32;
    let yy = y.round() as i32;
    if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
        return false;
    }
    valid[yy as usize * w + xx as usize]
}

fn distance_near_overlap(both: &[bool], w: usize, h: usize, y0: usize, x0: usize, y1: usize, x1: usize) -> Vec<f32> {
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut inside = vec![false; cw * ch];
    for y in 0..ch {
        let src = (y0 + y) * w + x0;
        inside[y * cw..(y + 1) * cw].copy_from_slice(&both[src..src + cw]);
    }
    let local = outside_distance(&inside, cw, ch);
    let mut dist = vec![1.0e6f32; w * h];
    for y in 0..ch {
        let dst = (y0 + y) * w + x0;
        dist[dst..dst + cw].copy_from_slice(&local[y * cw..(y + 1) * cw]);
    }
    dist
}

fn improved(lum_a: &[f32], lum_b: &[f32], img: &[f32], both: &[bool], vb: &[bool], w: usize, h: usize) -> (bool, f32, f32) {
    let (wr, wg, wb) = super::const_::luma_weights();
    let (by0, by1, bx0, bx1) = overlap_box(both, w, h);
    if by1 <= by0 || bx1 <= bx0 {
        return (false, 0.0, 0.0);
    }
    let margin = 24usize;
    let y0 = by0.saturating_sub(margin);
    let x0 = bx0.saturating_sub(margin);
    let y1 = (by1 + margin).min(h);
    let x1 = (bx1 + margin).min(w);
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut mag = vec![0f32; cw * ch];
    for y in 0..ch {
        let fy = y0 + y;
        for x in 0..cw {
            let fx = x0 + x;
            mag[y * cw + x] = if fy == 0 || fx == 0 || fy + 1 >= h || fx + 1 >= w {
                sobel_mag(lum_a, w, h, fy, fx)
            } else {
                let up = (fy - 1) * w;
                let mid = fy * w;
                let down = (fy + 1) * w;
                let xm = fx - 1;
                let xp = fx + 1;
                let gx = -lum_a[up + xm] + lum_a[up + xp] - 2.0 * lum_a[mid + xm] + 2.0 * lum_a[mid + xp] - lum_a[down + xm]
                    + lum_a[down + xp];
                let gy = -lum_a[up + xm] - 2.0 * lum_a[up + fx] - lum_a[up + xp] + lum_a[down + xm] + 2.0 * lum_a[down + fx]
                    + lum_a[down + xp];
                gx.abs() + gy.abs()
            };
        }
    }
    let mag = gauss_sigma(&mag, cw, ch, 2.0);
    let mut vals = Vec::new();
    for y in by0..by1 {
        for x in bx0..bx1 {
            let i = y * w + x;
            if both[i] {
                vals.push(mag[(y - y0) * cw + (x - x0)]);
            }
        }
    }
    if vals.is_empty() {
        return (false, 0.0, 0.0);
    }
    let thr = percentile60(&mut vals);
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut n = 0.0f32;
    for y in by0..by1 {
        for x in bx0..bx1 {
            let i = y * w + x;
            let g = mag[(y - y0) * cw + (x - x0)];
            if !(both[i] && vb[i] && g > thr) {
                continue;
            }
            let lb = wr * img[i * 3] + wg * img[i * 3 + 1] + wb * img[i * 3 + 2];
            let ref_l = ((lum_a[i] + lum_b[i]) * 0.5).max(0.02);
            d0 += (lum_a[i] - lum_b[i]).abs() / ref_l;
            d1 += (lum_a[i] - lb).abs() / ref_l;
            n += 1.0;
        }
    }
    if n < 1.0 {
        return (false, 0.0, 0.0);
    }
    d0 /= n;
    d1 /= n;
    (d1 <= d0 * 0.97 || (d0 - d1) >= 0.01, d0, d1)
}

fn phase(a: &[f32], b: &[f32], w: usize, y0: usize, x0: usize, tile: usize) -> (f32, f32, f32) {
    if tile == 128 {
        return phase_tile(a, b, w, y0, x0);
    }
    let n = tile;
    let mut fa = vec![[0f32, 0f32]; n * n];
    let mut fb = vec![[0f32, 0f32]; n * n];
    for y in 0..n {
        for x in 0..n {
            let win = (hann(x, n) * hann(y, n)).sqrt();
            let i = (y0 + y) * w + x0 + x;
            fa[y * n + x] = [a[i] * win, 0.0];
            fb[y * n + x] = [b[i] * win, 0.0];
        }
    }
    fft2_inplace(&mut fa, n, false);
    fft2_inplace(&mut fb, n, false);
    cross_power(&mut fa, &fb);
    fft2_inplace(&mut fa, n, true);
    let mut shifted = vec![0f32; n * n];
    shift_peak(&fa, &mut shifted, n);
    read_shift(&shifted, n)
}

fn phase_tile(a: &[f32], b: &[f32], w: usize, y0: usize, x0: usize) -> (f32, f32, f32) {
    let n = 128usize;
    thread_local! {
        static TILE: std::cell::RefCell<(Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<f32>)> = std::cell::RefCell::new((
            vec![[0.0, 0.0]; 128 * 128],
            vec![[0.0, 0.0]; 128 * 128],
            vec![0.0; 128 * 128],
        ));
    }
    TILE.with(|slot| {
        let (fa, fb, shifted) = &mut *slot.borrow_mut();
        let win = hann_row();
        for y in 0..n {
            for x in 0..n {
                let weight = (win[y] * win[x]).sqrt();
                let i = (y0 + y) * w + x0 + x;
                fa[y * n + x] = [a[i] * weight, 0.0];
                fb[y * n + x] = [b[i] * weight, 0.0];
            }
        }
        fft2_inplace(fa, n, false);
        fft2_inplace(fb, n, false);
        cross_power(fa, fb);
        fft2_inplace(fa, n, true);
        shift_peak(fa, shifted, n);
        read_shift(shifted, n)
    })
}

// Divides the cross-power spectrum in double precision, then keeps a unit magnitude.
fn cross_power(fa: &mut [[f32; 2]], fb: &[[f32; 2]]) {
    for (a, b) in fa.iter_mut().zip(fb.iter()) {
        let ar = a[0] as f64;
        let ai = a[1] as f64;
        let br = b[0] as f64;
        let bi = b[1] as f64;
        let re = ar * br + ai * bi;
        let im = ai * br - ar * bi;
        let mag = (re * re + im * im).sqrt().max(1e-12);
        a[0] = (re / mag) as f32;
        a[1] = (im / mag) as f32;
    }
}

fn shift_peak(src: &[[f32; 2]], dst: &mut [f32], n: usize) {
    let half = n / 2;
    for y in 0..n {
        for x in 0..n {
            let sy = (y + half) % n;
            let sx = (x + half) % n;
            dst[sy * n + sx] = src[y * n + x][0];
        }
    }
}

fn read_shift(shifted: &[f32], n: usize) -> (f32, f32, f32) {
    let (py, px, _) = peak_of(shifted, n);
    let (cy, cx, sum) = centroid_box(shifted, n, py, px);
    let center = n as f64 / 2.0;
    let resp = sum / (n as f64 * n as f64);
    ((center - cx) as f32, (center - cy) as f32, resp as f32)
}

fn hann_row() -> &'static [f32] {
    use std::sync::OnceLock;
    static ROW: OnceLock<Vec<f32>> = OnceLock::new();
    ROW.get_or_init(|| (0..128).map(|i| hann(i, 128)).collect())
}

fn fft2_inplace(buf: &mut [[f32; 2]], n: usize, inverse: bool) {
    if !inverse {
        rows_fft(buf, n, false);
        transpose_sq(buf, n);
        rows_fft(buf, n, false);
        transpose_sq(buf, n);
    } else {
        transpose_sq(buf, n);
        rows_fft(buf, n, true);
        transpose_sq(buf, n);
        rows_fft(buf, n, true);
    }
}

fn peak_of(img: &[f32], n: usize) -> (usize, usize, f32) {
    let mut peak = f32::MIN;
    let mut py = 0usize;
    let mut px = 0usize;
    for y in 0..n {
        for x in 0..n {
            let v = img[y * n + x];
            if v > peak {
                peak = v;
                py = y;
                px = x;
            }
        }
    }
    (py, px, peak)
}

fn centroid_box(img: &[f32], n: usize, py: usize, px: usize) -> (f64, f64, f64) {
    let miny = py as i32 - 2;
    let maxy = py as i32 + 2;
    let minx = px as i32 - 2;
    let maxx = px as i32 + 2;
    let miny = miny.max(0) as usize;
    let minx = minx.max(0) as usize;
    let maxy = maxy.min(n as i32 - 1) as usize;
    let maxx = maxx.min(n as i32 - 1) as usize;
    let mut sw = 0.0f64;
    let mut sy = 0.0f64;
    let mut sx = 0.0f64;
    for y in miny..=maxy {
        for x in minx..=maxx {
            let v = img[y * n + x] as f64;
            sw += v;
            sy += v * y as f64;
            sx += v * x as f64;
        }
    }
    if sw.abs() < 1e-12 {
        return (py as f64, px as f64, 0.0);
    }
    (sy / sw, sx / sw, sw)
}

fn hann(i: usize, n: usize) -> f32 {
    0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos()
}

fn rows_fft(buf: &mut [[f32; 2]], n: usize, inverse: bool) {
    for y in 0..n {
        fft(&mut buf[y * n..(y + 1) * n], inverse);
    }
}

fn transpose_sq(buf: &mut [[f32; 2]], n: usize) {
    for y in 0..n {
        for x in (y + 1)..n {
            buf.swap(y * n + x, x * n + y);
        }
    }
}

fn twiddles(inverse: bool) -> &'static [[f32; 2]] {
    use std::sync::OnceLock;
    static FWD: OnceLock<Vec<[f32; 2]>> = OnceLock::new();
    static INV: OnceLock<Vec<[f32; 2]>> = OnceLock::new();
    let slot = if inverse { &INV } else { &FWD };
    slot.get_or_init(|| {
        let mut table = Vec::with_capacity(127);
        let mut len = 2usize;
        while len <= 128 {
            let ang = 2.0 * std::f32::consts::PI / len as f32 * if inverse { 1.0 } else { -1.0 };
            let wlen = [ang.cos(), ang.sin()];
            let mut w = [1.0f32, 0.0];
            for _k in 0..len / 2 {
                table.push(w);
                w = [w[0] * wlen[0] - w[1] * wlen[1], w[0] * wlen[1] + w[1] * wlen[0]];
            }
            len <<= 1;
        }
        table
    })
    .as_slice()
}

fn fft(a: &mut [[f32; 2]], inverse: bool) {
    let n = a.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    if n == 128 {
        let table = twiddles(inverse);
        let mut len = 2usize;
        while len <= n {
            let off = len / 2 - 1;
            let half = len / 2;
            for i in (0..n).step_by(len) {
                for k in 0..half {
                    let w = table[off + k];
                    let u = a[i + k];
                    let v = a[i + k + half];
                    let t = [v[0] * w[0] - v[1] * w[1], v[0] * w[1] + v[1] * w[0]];
                    a[i + k] = [u[0] + t[0], u[1] + t[1]];
                    a[i + k + half] = [u[0] - t[0], u[1] - t[1]];
                }
            }
            len <<= 1;
        }
        return;
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * std::f32::consts::PI / len as f32 * if inverse { 1.0 } else { -1.0 };
        let wlen = [ang.cos(), ang.sin()];
        for i in (0..n).step_by(len) {
            let mut w = [1.0f32, 0.0];
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = a[i + k + len / 2];
                let t = [v[0] * w[0] - v[1] * w[1], v[0] * w[1] + v[1] * w[0]];
                a[i + k] = [u[0] + t[0], u[1] + t[1]];
                a[i + k + len / 2] = [u[0] - t[0], u[1] - t[1]];
                w = [w[0] * wlen[0] - w[1] * wlen[1], w[0] * wlen[1] + w[1] * wlen[0]];
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{percentile60, sobel_mag};

    #[test]
    fn percentile_matches_a_full_sort() {
        let mut vals: Vec<f32> = (0..50).map(|i| ((i * 7) % 13) as f32).collect();
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = 0.60 * (sorted.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = (pos - lo as f64) as f32;
        let expect = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
        let got = percentile60(&mut vals);
        assert!((got - expect).abs() < 1e-5, "{got} {expect}");
    }

    #[test]
    fn interior_gradient_matches_the_edge_formula() {
        let w = 6usize;
        let h = 5usize;
        let lum: Vec<f32> = (0..w * h).map(|i| (i % 9) as f32 * 0.1).collect();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let slow = sobel_mag(&lum, w, h, y, x);
                let up = &lum[(y - 1) * w..];
                let mid = &lum[y * w..];
                let down = &lum[(y + 1) * w..];
                let gx = -up[x - 1] + up[x + 1] - 2.0 * mid[x - 1] + 2.0 * mid[x + 1] - down[x - 1] + down[x + 1];
                let gy = -up[x - 1] - 2.0 * up[x] - up[x + 1] + down[x - 1] + 2.0 * down[x] + down[x + 1];
                let fast = gx.abs() + gy.abs();
                assert!((slow - fast).abs() < 1e-5);
            }
        }
    }
}
