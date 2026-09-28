use super::const_::BLEND_DEN;
use rayon::prelude::*;

// Softens the seam mask so the join fades over a few pixels.
pub fn blur_mask(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let k = gauss9();
    let mut tmp = vec![0f32; src.len()];
    if w == 0 || h == 0 {
        return tmp;
    }
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src_row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            row[x] = if x >= 4 && x + 4 < w {
                src_row[x - 4] * k[0]
                    + src_row[x - 3] * k[1]
                    + src_row[x - 2] * k[2]
                    + src_row[x - 1] * k[3]
                    + src_row[x] * k[4]
                    + src_row[x + 1] * k[5]
                    + src_row[x + 2] * k[6]
                    + src_row[x + 3] * k[7]
                    + src_row[x + 4] * k[8]
            } else {
                let mut s = 0.0f32;
                for i in 0..9 {
                    let xx = reflect(x as i32 + i as i32 - 4, w as i32) as usize;
                    s += src_row[xx] * k[i];
                }
                s
            };
        }
    });
    let mut dst = vec![0f32; src.len()];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y >= 4 && y + 4 < h {
            let rows = [
                &tmp[(y - 4) * w..(y - 3) * w],
                &tmp[(y - 3) * w..(y - 2) * w],
                &tmp[(y - 2) * w..(y - 1) * w],
                &tmp[(y - 1) * w..y * w],
                &tmp[y * w..(y + 1) * w],
                &tmp[(y + 1) * w..(y + 2) * w],
                &tmp[(y + 2) * w..(y + 3) * w],
                &tmp[(y + 3) * w..(y + 4) * w],
                &tmp[(y + 4) * w..(y + 5) * w],
            ];
            for x in 0..w {
                row[x] = rows[0][x] * k[0]
                    + rows[1][x] * k[1]
                    + rows[2][x] * k[2]
                    + rows[3][x] * k[3]
                    + rows[4][x] * k[4]
                    + rows[5][x] * k[5]
                    + rows[6][x] * k[6]
                    + rows[7][x] * k[7]
                    + rows[8][x] * k[8];
            }
        } else {
            for x in 0..w {
                let mut s = 0.0f32;
                for i in 0..9 {
                    let yy = reflect(y as i32 + i as i32 - 4, h as i32) as usize;
                    s += tmp[yy * w + x] * k[i];
                }
                row[x] = s;
            }
        }
    });
    dst
}

fn gauss9() -> [f32; 9] {
    let mut raw = [0.0f64; 9];
    let mut sum = 0.0f64;
    for i in 0..9 {
        let x = i as f64 - 4.0;
        raw[i] = (-0.5 * x * x).exp();
        sum += raw[i];
    }
    let mut k = [0.0f32; 9];
    for i in 0..9 {
        k[i] = (raw[i] / sum) as f32;
    }
    k
}

// Blends two overlapping photos so the join fades instead of showing a hard edge.
pub fn pyr_blend(a: &[f32], b: &[f32], mask: &[f32], va: &[bool], vb: &[bool], w: usize, h: usize, levels: usize) -> Vec<f32> {
    let ga = gauss_pyr(a, va, w, h, levels);
    let gb = gauss_pyr(b, vb, w, h, levels);
    super::trace::line("pyramid gauss");
    let mut gm = vec![mask.to_vec()];
    let mut mw = w;
    let mut mh = h;
    for _ in 0..levels {
        let next = pyr_down_gray(&gm.last().unwrap(), mw, mh);
        mw = (mw + 1) / 2;
        mh = (mh + 1) / 2;
        gm.push(next);
    }
    let mut la = Vec::new();
    let mut lb = Vec::new();
    for i in 0..levels {
        let (aw, ah) = ga[i].1;
        let up_a = pyr_up(&ga[i + 1].0, ga[i + 1].1 .0, ga[i + 1].1 .1, aw, ah);
        let up_b = pyr_up(&gb[i + 1].0, gb[i + 1].1 .0, gb[i + 1].1 .1, aw, ah);
        la.push(sub(&ga[i].0, &up_a));
        lb.push(sub(&gb[i].0, &up_b));
    }
    la.push(ga.last().unwrap().0.clone());
    lb.push(gb.last().unwrap().0.clone());
    let mut out = mix(&la[levels], &lb[levels], &gm[levels], ga[levels].1 .0);
    for i in (0..levels).rev() {
        let (aw, ah) = ga[i].1;
        let up = pyr_up(&out, ga[i + 1].1 .0, ga[i + 1].1 .1, aw, ah);
        let mixed = mix(&la[i], &lb[i], &gm[i], aw);
        out = add(&up, &mixed);
        let _ = ah;
    }
    super::trace::line("pyramid expand");
    out
}

pub fn blend_levels(w: u32, h: u32, lo: i32, hi: i32) -> usize {
    let v = ((w.max(h) as f64 / 300.0).log2()).round() as i32;
    v.clamp(lo, hi) as usize
}

fn gauss_pyr(img: &[f32], valid: &[bool], w: usize, h: usize, levels: usize) -> Vec<(Vec<f32>, (usize, usize))> {
    let mut num = vec![0f32; img.len()];
    let mut den = vec![0f32; valid.len()];
    for i in 0..valid.len() {
        let m = if valid[i] { 1.0 } else { 0.0 };
        den[i] = m;
        num[i * 3] = img[i * 3] * m;
        num[i * 3 + 1] = img[i * 3 + 1] * m;
        num[i * 3 + 2] = img[i * 3 + 2] * m;
    }
    let mut nums = vec![(num, (w, h))];
    let mut dens = vec![den];
    let mut cw = w;
    let mut ch = h;
    for _ in 0..levels {
        let n = pyr_down(&nums.last().unwrap().0, cw, ch);
        let d = pyr_down_gray(&dens.last().unwrap(), cw, ch);
        cw = (cw + 1) / 2;
        ch = (ch + 1) / 2;
        nums.push((n, (cw, ch)));
        dens.push(d);
    }
    let mut g = Vec::new();
    for (n, d) in nums.iter().zip(dens.iter()) {
        let (nw, nh) = n.1;
        let mut o = vec![0f32; nw * nh * 3];
        for i in 0..nw * nh {
            if d[i] > BLEND_DEN {
                let s = 1.0 / d[i].max(BLEND_DEN);
                o[i * 3] = n.0[i * 3] * s;
                o[i * 3 + 1] = n.0[i * 3 + 1] * s;
                o[i * 3 + 2] = n.0[i * 3 + 2] * s;
            }
        }
        g.push((o, (nw, nh)));
    }
    g
}

const TAP: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];

fn reflect(mut p: i32, len: i32) -> i32 {
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

fn pyr_down(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    down_sep(src, w, h, 3)
}

fn pyr_down_gray(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    down_sep(src, w, h, 1)
}

// Shrinks a picture by half with a short horizontal pass and a short vertical pass.
fn down_sep(src: &[f32], w: usize, h: usize, nch: usize) -> Vec<f32> {
    let nw = (w + 1) / 2;
    let nh = (h + 1) / 2;
    let mut dst = vec![0f32; nw * nh * nch];
    if w == 0 || h == 0 || nch == 0 {
        return dst;
    }
    let mut narrow = vec![0f32; nw * h * nch];
    narrow.par_chunks_mut(nw * nch).enumerate().for_each(|(y, row)| {
        if super::trace::halted() {
            return;
        }
        let src_row = &src[y * w * nch..(y + 1) * w * nch];
        for x in 0..nw {
            tap_row(src_row, nch, w, x * 2, &mut row[x * nch..(x + 1) * nch]);
        }
    });
    dst.par_chunks_mut(nw * nch).enumerate().for_each(|(y, row)| {
        if super::trace::halted() {
            return;
        }
        let sy = (y * 2) as i32;
        let ys = [
            reflect(sy - 2, h as i32) as usize,
            reflect(sy - 1, h as i32) as usize,
            reflect(sy, h as i32) as usize,
            reflect(sy + 1, h as i32) as usize,
            reflect(sy + 2, h as i32) as usize,
        ];
        for x in 0..nw {
            for c in 0..nch {
                let mut s = 0.0f32;
                for k in 0..5 {
                    s += narrow[(ys[k] * nw + x) * nch + c] * TAP[k];
                }
                row[x * nch + c] = s / 256.0;
            }
        }
    });
    dst
}

fn tap_row(src: &[f32], nch: usize, width: usize, center: usize, out: &mut [f32]) {
    if center >= 2 && center + 2 < width {
        for c in 0..nch {
            let b = center * nch + c;
            out[c] = src[b - 2 * nch] * TAP[0]
                + src[b - nch] * TAP[1]
                + src[b] * TAP[2]
                + src[b + nch] * TAP[3]
                + src[b + 2 * nch] * TAP[4];
        }
        return;
    }
    let n = width as i32;
    let c0 = center as i32;
    for c in 0..nch {
        let mut s = 0.0f32;
        for i in 0..5 {
            let p = reflect(c0 + i as i32 - 2, n) as usize;
            s += src[p * nch + c] * TAP[i];
        }
        out[c] = s;
    }
}

// Expands a picture back to the larger size.
fn pyr_up(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0f32; dw * dh * 3];
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return dst;
    }
    let mut mid = vec![0f32; sh * dw * 3];
    mid.par_chunks_mut(dw * 3).enumerate().for_each(|(sy, row)| {
        if super::trace::halted() {
            return;
        }
        let src_row = &src[sy * sw * 3..(sy + 1) * sw * 3];
        expand_line(src_row, sw, row);
    });
    dst.par_chunks_mut(dw * 3).enumerate().for_each(|(y, row)| {
        if super::trace::halted() {
            return;
        }
        expand_column(row, &mid, sh, dw, y);
    });
    dst
}

fn expand_line(src: &[f32], sw: usize, dst: &mut [f32]) {
    let dw = dst.len() / 3;
    if sw == 0 || dw == 0 {
        dst.fill(0.0);
        return;
    }
    for x in 0..dw.min(2) {
        put_expand(src, sw, dst, x);
    }
    if sw >= 2 {
        let mut i = 1usize;
        while i + 1 < sw {
            let x = i * 2;
            if x >= dw {
                break;
            }
            if x >= 2 {
                let p0 = (i - 1) * 3;
                for c in 0..3 {
                    dst[x * 3 + c] = src[p0 + c] * TAP[0] + src[p0 + 3 + c] * TAP[2] + src[p0 + 6 + c] * TAP[4];
                }
            }
            let xo = x + 1;
            if xo < dw {
                let p0 = i * 3;
                for c in 0..3 {
                    dst[xo * 3 + c] = src[p0 + c] * TAP[1] + src[p0 + 3 + c] * TAP[3];
                }
            }
            i += 1;
        }
    }
    let covered = if sw >= 2 { (2 * (sw - 1)).min(dw) } else { dw.min(2) };
    for x in covered..dw {
        put_expand(src, sw, dst, x);
    }
}

fn put_expand(src: &[f32], sw: usize, dst: &mut [f32], x: usize) {
    for c in 0..3 {
        dst[x * 3 + c] = expand_sample(src, sw, x, c);
    }
}

fn expand_sample(src: &[f32], sw: usize, x: usize, c: usize) -> f32 {
    let at = |i: i32| src[reflect(i, sw as i32) as usize * 3 + c];
    if x % 2 == 0 {
        let sx = (x / 2) as i32;
        at(sx - 1) * TAP[0] + at(sx) * TAP[2] + at(sx + 1) * TAP[4]
    } else {
        let i = (x / 2) as i32;
        at(i) * TAP[1] + at(i + 1) * TAP[3]
    }
}

fn expand_column(row: &mut [f32], mid: &[f32], sh: usize, dw: usize, y: usize) {
    let scale = 4.0 / 256.0;
    row.fill(0.0);
    if y % 2 == 0 {
        let sy = (y / 2) as i32;
        add_src_row(row, mid, sh, dw, sy - 1, TAP[0]);
        add_src_row(row, mid, sh, dw, sy, TAP[2]);
        add_src_row(row, mid, sh, dw, sy + 1, TAP[4]);
    } else {
        let i = (y / 2) as i32;
        add_src_row(row, mid, sh, dw, i, TAP[1]);
        add_src_row(row, mid, sh, dw, i + 1, TAP[3]);
    }
    for v in row.iter_mut() {
        *v *= scale;
    }
}

fn add_src_row(row: &mut [f32], mid: &[f32], sh: usize, dw: usize, sy: i32, weight: f32) {
    let sy = reflect(sy, sh as i32) as usize;
    let src = &mid[sy * dw * 3..(sy + 1) * dw * 3];
    for x in 0..row.len() {
        row[x] += src[x] * weight;
    }
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(p, q)| p - q).collect()
}
fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(p, q)| p + q).collect()
}
fn mix(a: &[f32], b: &[f32], m: &[f32], w: usize) -> Vec<f32> {
    let n = m.len();
    let mut o = vec![0f32; n * 3];
    for i in 0..n {
        let t = m[i];
        for c in 0..3 {
            o[i * 3 + c] = a[i * 3 + c] * t + b[i * 3 + c] * (1.0 - t);
        }
    }
    let _ = w;
    o
}
