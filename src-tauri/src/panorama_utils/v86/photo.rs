use super::const_::{FIELD_TERMS, GAIN_MAX, GAIN_MIN, GAUGE_WEIGHT, MAX_BLOCKS_PER_PAIR, PHOTO_ITERS, QUIET_TEX};
use super::rng::NumpyRng;
use super::warp::Placed;
use nalgebra::DMatrix;

pub struct Photo {
    pub gains: Vec<f64>,
    pub coef: Vec<f64>,
    pub pedestal: f64,
}

// Matches brightness across the overlaps and removes leftover corner darkening.
pub fn solve_photometry(placed: &[Placed], pairs: &[(usize, usize)], anchor: usize) -> Photo {
    let n = placed.len();
    let k = FIELD_TERMS.len();
    let pedestal = noise_pedestal(placed);
    let cols = n + k + 1;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut rng = NumpyRng::seed0();
    for &(a, b) in pairs {
        let Some((la, lb, uva, uvb)) = pair_blocks(&placed[a], &placed[b], pedestal) else {
            continue;
        };
        let mut idx: Vec<usize> = (0..la.len()).collect();
        if idx.len() > MAX_BLOCKS_PER_PAIR {
            idx = rng.choice(la.len(), MAX_BLOCKS_PER_PAIR);
        }
        let w = 1.0 / (idx.len() as f64).sqrt();
        for i in idx {
            let mut row = vec![0.0; cols];
            row[a] = w;
            row[b] = -w;
            let ba = basis(uva[i][0], uva[i][1]);
            let bb = basis(uvb[i][0], uvb[i][1]);
            for t in 0..k {
                row[n + t] = w * (ba[t] - bb[t]);
            }
            rows.push(row);
            rhs.push(w * (la[i].ln() - lb[i].ln()));
        }
    }
    let overlap_n = rows.len();
    sky_rows(placed, pedestal, n, cols, &mut rows, &mut rhs);
    if rows.is_empty() {
        return Photo { gains: vec![1.0; n], coef: vec![0.0; k + 1], pedestal };
    }
    let m = rows.len();
    for t in 0..k {
        let mut g = vec![0.0; cols];
        let (i, j) = FIELD_TERMS[t];
        g[n + t] = 0.05 * (1 + (i + j) / 2) as f64;
        rows.push(g);
        rhs.push(0.0);
    }
    let mut radial = vec![0.0; cols];
    radial[n + k] = 0.10;
    rows.push(radial);
    rhs.push(0.0);
    let mut gauge = vec![0.0; cols];
    gauge[anchor.min(n - 1)] = GAUGE_WEIGHT;
    rows.push(gauge);
    rhs.push(0.0);
    let mut wts = vec![1.0; rows.len()];
    let mut sol = vec![0.0; cols];
    for _ in 0..PHOTO_ITERS {
        sol = lstsq(&rows, &rhs, &wts, cols);
        let mut res = Vec::with_capacity(m);
        for i in 0..m {
            let mut p = 0.0;
            for c in 0..cols {
                p += rows[i][c] * sol[c];
            }
            res.push((p - rhs[i]).abs());
        }
        huber_weights(&res, &mut wts, 0, overlap_n);
        huber_weights(&res, &mut wts, overlap_n, m);
    }
    let mut xs = sol[..n].to_vec();
    let mean = xs.iter().sum::<f64>() / n as f64;
    for x in &mut xs {
        *x -= mean;
    }
    let gains = xs.iter().map(|x| (-x).exp().clamp(GAIN_MIN, GAIN_MAX)).collect();
    Photo { gains, coef: sol[n..].to_vec(), pedestal }
}

pub fn apply_photometry(placed: &mut Placed, photo: &Photo, index: usize) {
    let g = photo.gains[index] as f32;
    let p = photo.pedestal as f32;
    let use_field = photo.coef.iter().map(|c| c.abs()).sum::<f64>() > 1e-9;
    for i in 0..placed.w * placed.h {
        let mut field = 1.0f32;
        if use_field {
            let u = placed.uv[i * 2] as f64;
            let v = placed.uv[i * 2 + 1] as f64;
            field = field_log(photo, u, v).exp() as f32;
        }
        for ch in 0..3 {
            let l = placed.img[i * 3 + ch];
            let sig = (l - p).max(0.0) / field.max(1e-6);
            placed.img[i * 3 + ch] = sig * g + p;
        }
    }
}

pub fn illumination_corner(photo: &Photo, placed: &Placed) -> f64 {
    let mut um = 0.0f64;
    let mut vm = 0.0f64;
    for i in 0..placed.w * placed.h {
        if !placed.valid[i] {
            continue;
        }
        um = um.max((placed.uv[i * 2] as f64).abs());
        vm = vm.max((placed.uv[i * 2 + 1] as f64).abs());
    }
    field_log(photo, um, vm).exp()
}

fn field_log(photo: &Photo, u: f64, v: f64) -> f64 {
    let b = basis(u, v);
    let mut s = 0.0;
    for (c, coef) in b.iter().zip(photo.coef.iter()) {
        s += c * coef;
    }
    if let Some(radial) = photo.coef.get(b.len()) {
        s += radial * (u * u + v * v);
    }
    s
}

#[derive(Clone, Copy)]
struct SkyPx {
    r2: f64,
    log_l: f64,
}

// Flat areas inside one photo, so a corner both sides of a seam share can still be measured.
fn sky_rows(placed: &[Placed], pedestal: f64, n: usize, cols: usize, rows: &mut Vec<Vec<f64>>, rhs: &mut Vec<f64>) {
    let band = 1.25_f64.ln();
    for frame in placed {
        let samples = quiet_grid(frame, pedestal);
        if samples.len() < 16 {
            continue;
        }
        let mut levels: Vec<f64> = samples.iter().map(|s| s.log_l).collect();
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = levels[levels.len() / 2];
        let kept: Vec<SkyPx> = samples.into_iter().filter(|s| (s.log_l - mid).abs() <= band).collect();
        if kept.len() < 16 {
            continue;
        }
        let mut by_r = kept;
        by_r.sort_by(|a, b| a.r2.partial_cmp(&b.r2).unwrap_or(std::cmp::Ordering::Equal));
        let span = by_r.last().unwrap().r2 - by_r.first().unwrap().r2;
        if span < 0.08 {
            continue;
        }
        let ref_n = (by_r.len() / 4).max(8).min(by_r.len() / 2);
        let refs = &by_r[..ref_n];
        let mut ref_logs: Vec<f64> = refs.iter().map(|s| s.log_l).collect();
        ref_logs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let log_ref = ref_logs[ref_logs.len() / 2];
        let r_ref = refs.iter().map(|s| s.r2).sum::<f64>() / refs.len() as f64;
        let outer: Vec<&SkyPx> = by_r.iter().filter(|s| s.r2 > r_ref + 0.02 && (s.log_l - log_ref).abs() <= band).collect();
        if outer.len() < 8 {
            continue;
        }
        let w = 1.0 / (outer.len() as f64).sqrt();
        for s in outer {
            let mut row = vec![0.0; cols];
            row[n + FIELD_TERMS.len()] = w * (s.r2 - r_ref);
            rows.push(row);
            rhs.push(w * (s.log_l - log_ref));
        }
    }
}

fn quiet_grid(frame: &Placed, pedestal: f64) -> Vec<SkyPx> {
    let mut out = Vec::new();
    if frame.w < 2 || frame.h < 2 {
        return out;
    }
    for y in (1..frame.h).step_by(32) {
        for x in (1..frame.w).step_by(32) {
            let i = y * frame.w + x;
            let left = i - 1;
            let up = i - frame.w;
            if !frame.valid[i] || !frame.valid[left] || !frame.valid[up] {
                continue;
            }
            let l = luma(&frame.img[i * 3..i * 3 + 3]);
            let lx = luma(&frame.img[left * 3..left * 3 + 3]);
            let ly = luma(&frame.img[up * 3..up * 3 + 3]);
            let tex = ((l - lx).abs() + (l - ly).abs()).min(0.5) * 2.0;
            if tex >= QUIET_TEX {
                continue;
            }
            let signal = l as f64 - pedestal;
            if signal <= 0.004 || signal >= 0.6 {
                continue;
            }
            let u = frame.uv[i * 2] as f64;
            let v = frame.uv[i * 2 + 1] as f64;
            out.push(SkyPx { r2: u * u + v * v, log_l: signal.ln() });
        }
    }
    out
}

fn huber_weights(res: &[f64], wts: &mut [f64], start: usize, end: usize) {
    if start >= end {
        return;
    }
    let mut abs: Vec<f64> = res[start..end].to_vec();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = abs[abs.len() / 2];
    let s = 1.4826 * med + 1e-6;
    for i in start..end {
        let hub = (1.5 * s / res[i].max(1e-9)).min(1.0);
        wts[i] = hub.sqrt();
    }
}

fn basis(u: f64, v: f64) -> [f64; 9] {
    let mut o = [0.0; 9];
    for (i, &(a, b)) in FIELD_TERMS.iter().enumerate() {
        o[i] = u.powi(a) * v.powi(b);
    }
    o
}

fn noise_pedestal(placed: &[Placed]) -> f64 {
    let mut vals = Vec::new();
    for p in placed {
        let mut lum = Vec::new();
        for y in (0..p.h).step_by(4) {
            for x in (0..p.w).step_by(4) {
                let i = y * p.w + x;
                if p.valid[i] {
                    lum.push(luma(&p.img[i * 3..i * 3 + 3]));
                }
            }
        }
        if lum.is_empty() {
            continue;
        }
        lum.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((lum.len() as f64) * 0.005) as usize;
        vals.push(lum[k.min(lum.len() - 1)] as f64);
    }
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals[vals.len() / 2]
}

fn pair_blocks(a: &Placed, b: &Placed, pedestal: f64) -> Option<(Vec<f64>, Vec<f64>, Vec<[f64; 2]>, Vec<[f64; 2]>)> {
    let x0 = a.x0.max(b.x0);
    let y0 = a.y0.max(b.y0);
    let x1 = a.x1().min(b.x1());
    let y1 = a.y1().min(b.y1());
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let bw = ((x1 - x0) / 8).max(1) as usize;
    let bh = ((y1 - y0) / 8).max(1) as usize;
    let mut la = vec![0.0; bw * bh];
    let mut lb = vec![0.0; bw * bh];
    let mut uva = vec![[0.0; 2]; bw * bh];
    let mut uvb = vec![[0.0; 2]; bw * bh];
    let mut cnt = vec![0u32; bw * bh];
    let mut mx = vec![0.0f64; bw * bh];
    for y in (y0..y1).step_by(4) {
        for x in (x0..x1).step_by(4) {
            let ia = ((y - a.y0) as usize) * a.w + (x - a.x0) as usize;
            let ib = ((y - b.y0) as usize) * b.w + (x - b.x0) as usize;
            if !a.valid[ia] || !b.valid[ib] {
                continue;
            }
            let bx = ((x - x0) as usize * bw / (x1 - x0) as usize).min(bw - 1);
            let by = ((y - y0) as usize * bh / (y1 - y0) as usize).min(bh - 1);
            let k = by * bw + bx;
            let pa = luma(&a.img[ia * 3..ia * 3 + 3]) as f64;
            let pb = luma(&b.img[ib * 3..ib * 3 + 3]) as f64;
            la[k] += pa;
            lb[k] += pb;
            mx[k] += pa.max(pb);
            uva[k][0] += a.uv[ia * 2] as f64;
            uva[k][1] += a.uv[ia * 2 + 1] as f64;
            uvb[k][0] += b.uv[ib * 2] as f64;
            uvb[k][1] += b.uv[ib * 2 + 1] as f64;
            cnt[k] += 1;
        }
    }
    let mut oa = Vec::new();
    let mut ob = Vec::new();
    let mut oua = Vec::new();
    let mut oub = Vec::new();
    for k in 0..cnt.len() {
        if cnt[k] == 0 {
            continue;
        }
        let c = cnt[k] as f64;
        let pa = la[k] / c - pedestal;
        let pb = lb[k] / c - pedestal;
        let peak = mx[k] / c;
        if pa <= 0.004 || pb <= 0.004 || peak >= 0.6 {
            continue;
        }
        let r = pa / pb.max(1e-9);
        if !(0.5..2.0).contains(&r) {
            continue;
        }
        oa.push(pa);
        ob.push(pb);
        oua.push([uva[k][0] / c, uva[k][1] / c]);
        oub.push([uvb[k][0] / c, uvb[k][1] / c]);
    }
    if oa.len() < 16 {
        return None;
    }
    Some((oa, ob, oua, oub))
}

pub fn luma(px: &[f32]) -> f32 {
    let (r, g, b) = super::const_::luma_weights();
    r * px[0] + g * px[1] + b * px[2]
}

fn lstsq(rows: &[Vec<f64>], rhs: &[f64], wts: &[f64], cols: usize) -> Vec<f64> {
    let mut ata = DMatrix::<f64>::zeros(cols, cols);
    let mut atb = vec![0.0; cols];
    for (row, (&y, &w)) in rows.iter().zip(rhs.iter().zip(wts.iter())) {
        for c in 0..cols {
            let wc = row[c] * w;
            atb[c] += wc * y * w;
            for d in 0..cols {
                ata[(c, d)] += wc * row[d] * w;
            }
        }
    }
    for i in 0..cols {
        ata[(i, i)] += 1e-8;
    }
    let svd = ata.svd(true, true);
    let sol = svd.solve(&nalgebra::DVector::from_column_slice(&atb), 1e-10).unwrap_or_else(|_| nalgebra::DVector::zeros(cols));
    sol.iter().copied().collect()
}
