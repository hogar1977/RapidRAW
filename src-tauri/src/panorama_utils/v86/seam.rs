use super::const_::{QUIET_D, QUIET_TEX, RIM_PX, SEAM_BAND, SEAM_COARSE, SEAM_DIRECT_MAX, SEAM_FINE, VALLEY_PX, WANDER_PULL};
use super::maxflow::min_cut;

// Chooses, in the overlap, which photo to keep so the join hides in a quiet place.
pub fn graphcut_mask(lum_a: &[f32], lum_b: &[f32], va: &[bool], vb: &[bool], w: usize, h: usize) -> Vec<f32> {
    cut_mask(lum_a, lum_b, va, vb, w, h).0
}

// Counts the cells in the fine seam when the overlap is large enough to use the band.
pub fn graphcut_refine_nodes(lum_a: &[f32], lum_b: &[f32], va: &[bool], vb: &[bool], w: usize, h: usize) -> usize {
    cut_mask(lum_a, lum_b, va, vb, w, h).1
}

fn cut_mask(lum_a: &[f32], lum_b: &[f32], va: &[bool], vb: &[bool], w: usize, h: usize) -> (Vec<f32>, usize) {
    let mut mask = vec![0f32; w * h];
    for i in 0..w * h {
        if va[i] {
            mask[i] = 1.0;
        }
    }
    let both: Vec<bool> = va.iter().zip(vb.iter()).map(|(a, b)| *a && *b).collect();
    if !both.iter().any(|v| *v) {
        return (mask, 0);
    }
    let step = SEAM_FINE as usize;
    let (y0, y1, x0, x1) = bounds(&both, w, h, step * 2);
    let gy = sample_axis(y0, y1, step);
    let gx = sample_axis(x0, x1, step);
    if gy.len() * gx.len() <= SEAM_DIRECT_MAX || gy.len().min(gx.len()) < 4 {
        if let Some((lab, _)) = solve_grid(lum_a, lum_b, va, vb, w, &gy, &gx, None, None, None) {
            return (paint(mask, &lab, &both, va, vb, w, h, &gy, &gx), 0);
        }
        return (mask, 0);
    }
    let coarse = SEAM_COARSE as usize;
    let (y0, y1, x0, x1) = bounds(&both, w, h, coarse);
    let gy = sample_axis(y0, y1, step);
    let gx = sample_axis(x0, x1, step);
    let gy_c = anchor_axis(sample_axis(y0, y1, coarse), y0, y1, x0, x1, va, vb, w, false);
    let gy_c = anchor_axis(gy_c, y0, y1, x0, x1, vb, va, w, false);
    let gx_c = anchor_axis(sample_axis(x0, x1, coarse), x0, x1, y0, y1, va, vb, w, true);
    let gx_c = anchor_axis(gx_c, x0, x1, y0, y1, vb, va, w, true);
    let Some((lab_c, _)) = solve_grid(lum_a, lum_b, va, vb, w, &gy_c, &gx_c, None, None, None) else {
        return (mask, 0);
    };
    let nh = gy.len();
    let nw = gx.len();
    let mut coarse_here = vec![false; nh * nw];
    for iy in 0..nh {
        let row = nearest(&gy_c, gy[iy]);
        for ix in 0..nw {
            let col = nearest(&gx_c, gx[ix]);
            coarse_here[iy * nw + ix] = lab_c[row * gx_c.len() + col];
        }
    }
    let mut cut = vec![false; gy_c.len() * gx_c.len()];
    let cw = gx_c.len();
    let ch = gy_c.len();
    for y in 0..ch {
        for x in 0..cw {
            let i = y * cw + x;
            let here = lab_c[i];
            if x + 1 < cw && lab_c[i + 1] != here {
                cut[i] = true;
                cut[i + 1] = true;
            }
            if y + 1 < ch && lab_c[i + cw] != here {
                cut[i] = true;
                cut[i + cw] = true;
            }
        }
    }
    let rad = (SEAM_BAND as usize / 2 / coarse).max(1);
    let band_c = dilate(&cut, cw, ch, rad);
    let mut valid = vec![false; nh * nw];
    let mut va_only = vec![false; nh * nw];
    let mut vb_only = vec![false; nh * nw];
    let mut band = vec![false; nh * nw];
    for iy in 0..nh {
        let row = nearest(&gy_c, gy[iy]);
        for ix in 0..nw {
            let col = nearest(&gx_c, gx[ix]);
            let s = gy[iy] * w + gx[ix];
            let d = iy * nw + ix;
            valid[d] = va[s] || vb[s];
            va_only[d] = va[s] && !vb[s];
            vb_only[d] = vb[s] && !va[s];
            band[d] = band_c[row * cw + col] && valid[d];
        }
    }
    let mut fringe = vec![false; nh * nw];
    for iy in 0..nh {
        for ix in 0..nw {
            let d = iy * nw + ix;
            if band[d] || !valid[d] {
                continue;
            }
            let left = ix > 0 && band[d - 1];
            let right = ix + 1 < nw && band[d + 1];
            let up = iy > 0 && band[d - nw];
            let down = iy + 1 < nh && band[d + nw];
            if left || right || up || down {
                fringe[d] = true;
            }
        }
    }
    if !band.iter().any(|v| *v) || !fringe.iter().any(|v| *v) {
        return (paint(mask, &coarse_here, &both, va, vb, w, h, &gy, &gx), 0);
    }
    let mut active = band.clone();
    let mut term_a = vec![false; nh * nw];
    let mut term_b = vec![false; nh * nw];
    for i in 0..nh * nw {
        active[i] = band[i] || fringe[i];
        term_a[i] = (fringe[i] && coarse_here[i]) || (active[i] && va_only[i]);
        term_b[i] = (fringe[i] && !coarse_here[i]) || (active[i] && vb_only[i]);
    }
    let Some((mut lab, nodes)) = solve_grid(
        lum_a,
        lum_b,
        va,
        vb,
        w,
        &gy,
        &gx,
        Some(&active),
        Some(&term_a),
        Some(&term_b),
    ) else {
        return (paint(mask, &coarse_here, &both, va, vb, w, h, &gy, &gx), 0);
    };
    for i in 0..nh * nw {
        if !band[i] {
            lab[i] = coarse_here[i];
        }
    }
    super::trace::line(&format!("seam refine nodes={nodes}"));
    (paint(mask, &lab, &both, va, vb, w, h, &gy, &gx), nodes)
}

fn solve_grid(
    lum_a: &[f32],
    lum_b: &[f32],
    va: &[bool],
    vb: &[bool],
    w: usize,
    gy: &[usize],
    gx: &[usize],
    active: Option<&[bool]>,
    term_a: Option<&[bool]>,
    term_b: Option<&[bool]>,
) -> Option<(Vec<bool>, usize)> {
    let nh = gy.len();
    let nw = gx.len();
    if nh < 2 || nw < 2 {
        return None;
    }
    let mut a = vec![0f32; nh * nw];
    let mut b = vec![0f32; nh * nw];
    let mut va_g = vec![false; nh * nw];
    let mut vb_g = vec![false; nh * nw];
    for (iy, &y) in gy.iter().enumerate() {
        for (ix, &x) in gx.iter().enumerate() {
            let s = y * w + x;
            let d = iy * nw + ix;
            a[d] = lum_a[s];
            b[d] = lum_b[s];
            va_g[d] = va[s];
            vb_g[d] = vb[s];
        }
    }
    let both: Vec<bool> = va_g.iter().zip(vb_g.iter()).map(|(p, q)| *p && *q).collect();
    let (d, tex, veto) = seam_fields(&a, &b, &both, nw, nh);
    let mut ids = vec![-1i32; nh * nw];
    let mut nodes = 0usize;
    for i in 0..nh * nw {
        let on = va_g[i] || vb_g[i];
        let on = active.map(|m| on && m[i]).unwrap_or(on);
        if on {
            ids[i] = nodes as i32;
            nodes += 1;
        }
    }
    if nodes < 2 {
        return None;
    }
    super::trace::line(&format!("seam grid {nh}x{nw} nodes={nodes}"));
    if super::trace::halted() {
        return None;
    }
    let src = nodes;
    let snk = nodes + 1;
    let (bias, quiet) = seam_steer(&both, gx, gy, &tex, &d, nw, nh);
    let mut edges = Vec::new();
    let scale = 1000.0f32;
    for (dy, dx) in [(0isize, 1isize), (1, 0)] {
        for y in 0..nh as isize - dy {
            for x in 0..nw as isize - dx {
                let p = (y as usize) * nw + x as usize;
                let q = ((y + dy) as usize) * nw + (x + dx) as usize;
                if ids[p] < 0 || ids[q] < 0 {
                    continue;
                }
                let mut base = (d[p] + d[q]) * 0.5 / (1.0 + tex[p] + tex[q]) + (veto[p] + veto[q]) * 0.5 + 0.02;
                if quiet[p] && quiet[q] {
                    base = 0.02;
                }
                let c = base + (bias[p] + bias[q]) * 0.5;
                let ci = (c * scale).round().max(1.0) as i32;
                edges.push((ids[p] as usize, ids[q] as usize, ci));
                edges.push((ids[q] as usize, ids[p] as usize, ci));
            }
        }
    }
    let inf = 1 << 30;
    let mut src_n = 0usize;
    let mut snk_n = 0usize;
    for i in 0..nh * nw {
        if ids[i] < 0 {
            continue;
        }
        let keep_a = term_a.map(|t| t[i]).unwrap_or(va_g[i] && !vb_g[i]);
        let take_b = term_b.map(|t| t[i] && !keep_a).unwrap_or(vb_g[i] && !va_g[i]);
        if keep_a {
            edges.push((src, ids[i] as usize, inf));
            src_n += 1;
        } else if take_b {
            edges.push((ids[i] as usize, snk, inf));
            snk_n += 1;
        }
    }
    if src_n == 0 || snk_n == 0 {
        return None;
    }
    let (seen, _) = min_cut(nodes + 2, &edges, src, snk);
    let mut lab = vec![false; nh * nw];
    for i in 0..nh * nw {
        if ids[i] >= 0 {
            lab[i] = seen[ids[i] as usize];
        }
    }
    if let Some(ta) = term_a {
        for i in 0..nh * nw {
            if ids[i] >= 0 && ta[i] {
                lab[i] = true;
            }
        }
    }
    if let Some(tb) = term_b {
        for i in 0..nh * nw {
            if ids[i] >= 0 && tb[i] && term_a.map(|t| !t[i]).unwrap_or(true) {
                lab[i] = false;
            }
        }
    }
    Some((lab, nodes))
}

fn seam_fields(a: &[f32], b: &[f32], both: &[bool], w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut d = vec![0f32; n];
    let mut tex = vec![0f32; n];
    let mut veto = vec![0f32; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let diff = (a[i] - b[i]).abs() / ((a[i] + b[i]) * 0.5).max(0.02);
            let gx = if x > 0 {
                (a[i] - a[i - 1]).abs() + (b[i] - b[i - 1]).abs()
            } else {
                0.0
            };
            let gy = if y > 0 {
                (a[i] - a[i - w]).abs() + (b[i] - b[i - w]).abs()
            } else {
                0.0
            };
            let t = (gx + gy).min(0.5) * 2.0;
            let agree = (-(diff / 0.2).powi(2)).exp();
            if both[i] {
                d[i] = diff;
                tex[i] = t;
                veto[i] = t * (1.0 - agree);
            }
        }
    }
    (d, tex, veto)
}

fn seam_steer(both: &[bool], xs: &[usize], ys: &[usize], tex: &[f32], d: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<bool>) {
    let mut bias = vec![0f32; w * h];
    let mut quiet = vec![false; w * h];
    if w < 2 || h < 2 {
        return (bias, quiet);
    }
    let dist = distance(both, w, h);
    let step_x = if xs.len() > 1 {
        let mut s = Vec::new();
        for i in 1..xs.len() {
            s.push((xs[i] as i32 - xs[i - 1] as i32).abs());
        }
        s.sort();
        s[s.len() / 2].max(1) as f32
    } else {
        1.0
    };
    let step_y = if ys.len() > 1 {
        let mut s = Vec::new();
        for i in 1..ys.len() {
            s.push((ys[i] as i32 - ys[i - 1] as i32).abs());
        }
        s.sort();
        s[s.len() / 2].max(1) as f32
    } else {
        1.0
    };
    let step = step_x.min(step_y);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !both[i] {
                continue;
            }
            let rim = dist[i] * step < RIM_PX as f32;
            let q = !rim && tex[i] < QUIET_TEX && d[i] < QUIET_D;
            quiet[i] = q;
            if rim {
                bias[i] = 40.0;
            } else if q {
                let mid = (xs[0] + xs[w - 1]) as f32 * 0.5;
                let delta = xs[x] as f32 - mid;
                bias[i] = (WANDER_PULL * (1.0 - (-(delta as f64 / VALLEY_PX).powi(2)).exp())) as f32;
            }
        }
    }
    (bias, quiet)
}

fn distance(mask: &[bool], w: usize, h: usize) -> Vec<f32> {
    let hv = (0.955f32 * 65536.0).round() as u32;
    let diag = (1.3693f32 * 65536.0).round() as u32;
    let dist_max = u32::MAX - diag;
    let bw = w + 2;
    let mut tmp = vec![dist_max; bw * (h + 2)];
    for y in 0..h {
        let row = (y + 1) * bw + 1;
        for x in 0..w {
            let j = row + x;
            if !mask[y * w + x] {
                tmp[j] = 0;
                continue;
            }
            let mut t0 = tmp[j - bw - 1].wrapping_add(diag);
            let mut t = tmp[j - bw].wrapping_add(hv);
            if t0 > t {
                t0 = t;
            }
            t = tmp[j - bw + 1].wrapping_add(diag);
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
        let row = (y + 1) * bw + 1;
        for x in (0..w).rev() {
            let j = row + x;
            let mut t0 = tmp[j];
            if t0 > hv {
                let mut t = tmp[j + bw + 1].wrapping_add(diag);
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

fn paint(mut mask: Vec<f32>, lab: &[bool], both: &[bool], va: &[bool], vb: &[bool], w: usize, h: usize, gy: &[usize], gx: &[usize]) -> Vec<f32> {
    let nw = gx.len();
    for (iy, &y) in gy.iter().enumerate() {
        let y1 = if iy + 1 < gy.len() { gy[iy + 1] } else { h };
        for (ix, &x) in gx.iter().enumerate() {
            let x1 = if ix + 1 < gx.len() { gx[ix + 1] } else { w };
            let keep = lab[iy * nw + ix];
            for yy in y..y1.min(h) {
                for xx in x..x1.min(w) {
                    let i = yy * w + xx;
                    if both[i] {
                        mask[i] = if keep { 1.0 } else { 0.0 };
                    } else if va[i] {
                        mask[i] = 1.0;
                    } else if vb[i] {
                        mask[i] = 0.0;
                    }
                }
            }
        }
    }
    super::trace::line("painted");
    mask
}

fn bounds(both: &[bool], w: usize, h: usize, pad: usize) -> (usize, usize, usize, usize) {
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
    (y0.saturating_sub(pad), (y1 + pad).min(h), x0.saturating_sub(pad), (x1 + pad).min(w))
}

fn nearest(samples: &[usize], q: usize) -> usize {
    if samples.len() <= 1 {
        return 0;
    }
    let mut pos = samples.partition_point(|&s| s < q);
    pos = pos.clamp(1, samples.len() - 1);
    let left = samples[pos - 1];
    let right = samples[pos];
    if q.abs_diff(left) <= q.abs_diff(right) { pos - 1 } else { pos }
}

fn dilate(src: &[bool], w: usize, h: usize, rad: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if !src[y * w + x] {
                continue;
            }
            let y0 = y.saturating_sub(rad);
            let x0 = x.saturating_sub(rad);
            for yy in y0..(y + rad + 1).min(h) {
                for xx in x0..(x + rad + 1).min(w) {
                    out[yy * w + xx] = true;
                }
            }
        }
    }
    out
}

fn anchor_axis(
    mut axis: Vec<usize>,
    origin: usize,
    stop: usize,
    cross0: usize,
    cross1: usize,
    yes: &[bool],
    no: &[bool],
    w: usize,
    along_cols: bool,
) -> Vec<usize> {
    let mut line = Vec::new();
    if along_cols {
        for x in origin..stop.min(w) {
            let any = (cross0..cross1).any(|y| {
                let i = y * w + x;
                yes.get(i).copied().unwrap_or(false) && !no.get(i).copied().unwrap_or(false)
            });
            line.push(any);
        }
    } else {
        for y in origin..stop {
            let any = (cross0..cross1.min(w)).any(|x| {
                let i = y * w + x;
                yes.get(i).copied().unwrap_or(false) && !no.get(i).copied().unwrap_or(false)
            });
            line.push(any);
        }
    }
    let hits: Vec<usize> = line.iter().enumerate().filter(|(_, v)| **v).map(|(i, _)| i).collect();
    if hits.is_empty() {
        return axis;
    }
    let mid = if hits.len() % 2 == 1 {
        hits[hits.len() / 2]
    } else {
        (hits[hits.len() / 2 - 1] + hits[hits.len() / 2]) / 2
    };
    let extra = origin + mid;
    if extra < origin || extra >= stop || axis.contains(&extra) {
        return axis;
    }
    axis.push(extra);
    axis.sort_unstable();
    axis
}

fn sample_axis(a: usize, b: usize, step: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut i = a;
    while i < b {
        v.push(i);
        i += step.max(1);
    }
    if v.is_empty() || *v.last().unwrap() != b.saturating_sub(1) {
        if b > a {
            v.push(b - 1);
        }
    }
    v
}
