use super::const_::{CONTRAST_THRESHOLD, EDGE_THRESHOLD, NFEATURES, NOCTAVE_LAYERS, SIFT_SIGMA};
use rayon::prelude::*;

pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
    pub layer: i32,
}

pub struct Feature {
    pub pt: [f64; 2],
    pub desc: [f32; 128],
}

struct Plane {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

impl Plane {
    fn at(&self, r: i32, c: i32) -> f32 {
        self.data[r as usize * self.w + c as usize]
    }
}

pub fn detect(gray: &[u8], mask: &[u8], width: usize, height: usize) -> Vec<Feature> {
    let t = std::time::Instant::now();
    let mut base = upsample_blur(gray, width, height, SIFT_SIGMA as f32);
    let bw = width * 2;
    let bh = height * 2;
    let n_octaves = (((bw.min(bh) as f64).ln() / 2f64.ln() - 2.0).round() as i32) - (-1);
    let n_octaves = n_octaves.max(1) as usize;
    let layers = NOCTAVE_LAYERS as usize;
    let mut sig = vec![0f32; layers + 3];
    sig[0] = SIFT_SIGMA as f32;
    let k = 2f32.powf(1.0 / NOCTAVE_LAYERS as f32);
    for i in 1..(layers + 3) {
        let prev = k.powi(i as i32 - 1) * SIFT_SIGMA as f32;
        let total = prev * k;
        sig[i] = (total * total - prev * prev).sqrt();
    }
    let mut gauss: Vec<Plane> = Vec::new();
    for o in 0..n_octaves {
        if super::trace::halted() {
            return Vec::new();
        }
        for i in 0..(layers + 3) {
            if o == 0 && i == 0 {
                gauss.push(Plane { w: bw, h: bh, data: std::mem::take(&mut base) });
            } else if i == 0 {
                let src = &gauss[(o - 1) * (layers + 3) + layers];
                gauss.push(halve(src));
            } else {
                let src = &gauss[o * (layers + 3) + i - 1];
                gauss.push(blur(src, sig[i]));
            }
        }
    }
    super::trace::line(&format!("gaussian {:.2}s", t.elapsed().as_secs_f64()));
    let t = std::time::Instant::now();
    let dog = difference_planes(&gauss, n_octaves, layers);
    super::trace::line(&format!("difference {:.2}s", t.elapsed().as_secs_f64()));
    let t = std::time::Instant::now();
    let threshold = (0.5 * CONTRAST_THRESHOLD / (NOCTAVE_LAYERS as f64) * 255.0).floor() as f32;
    let mut kpts = Vec::new();
    for o in 0..n_octaves {
        for i in 1..=layers {
            let idx = o * (layers + 2) + i;
            extrema(&dog, &gauss, o as i32, i as i32, idx, threshold, &mut kpts);
        }
    }
    kpts.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    dedup(&mut kpts);
    if kpts.len() > NFEATURES {
        kpts.truncate(NFEATURES);
    }
    for k in kpts.iter_mut() {
        k.x *= 0.5;
        k.y *= 0.5;
        k.size *= 0.5;
        k.octave -= 1;
    }
    kpts.retain(|k| {
        let x = k.x.round() as isize;
        let y = k.y.round() as isize;
        x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height && mask[y as usize * width + x as usize] != 0
    });
    super::trace::line(&format!("extrema {:.2}s", t.elapsed().as_secs_f64()));
    let t = std::time::Instant::now();
    let feats: Vec<Feature> = kpts.par_iter()
        .map(|k| {
            let scale = if k.octave >= 0 { 1.0 / (1 << k.octave) as f32 } else { (1 << -k.octave) as f32 };
            let img = &gauss[((k.octave + 1) as usize) * (layers + 3) + k.layer as usize];
            let desc = descriptor(img, [k.x * scale, k.y * scale], 360.0 - k.angle, k.size * scale * 0.5);
            Feature { pt: [k.x as f64, k.y as f64], desc }
        })
        .collect();
    super::trace::line(&format!("descriptor {:.2}s", t.elapsed().as_secs_f64()));
    feats
}

pub fn pyramid_bytes(width: u32, height: u32) -> u64 {
    let bw = width as u64 * 2;
    let bh = height as u64 * 2;
    let n_octaves = (((bw.min(bh) as f64).ln() / 2f64.ln() - 2.0).round() as i32 + 1).max(1) as u64;
    let g_layers = (NOCTAVE_LAYERS as u64) + 3;
    let d_layers = (NOCTAVE_LAYERS as u64) + 2;
    let mut pixels = 0u64;
    let mut w = bw;
    let mut h = bh;
    for _ in 0..n_octaves {
        pixels += w * h * (g_layers + d_layers);
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    pixels * 4
}

fn upsample_blur(gray: &[u8], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let bw = w * 2;
    let bh = h * 2;
    let mut dbl = vec![0f32; bw * bh];
    if w > 0 && h > 0 {
        dbl.par_chunks_mut(bw).enumerate().for_each(|(y, row)| {
            let sy = y as f32 * 0.5;
            let y0 = sy.floor() as i32;
            let dy = sy - y0 as f32;
            let y_in = y0 >= 0 && (y0 as usize) + 1 < h;
            for x in 0..bw {
                let sx = x as f32 * 0.5;
                let x0 = sx.floor() as i32;
                let dx = sx - x0 as f32;
                if y_in && x0 >= 0 && (x0 as usize) + 1 < w {
                    let x0 = x0 as usize;
                    let y0 = y0 as usize;
                    let a = gray[y0 * w + x0] as f32 * (1.0 - dx) + gray[y0 * w + x0 + 1] as f32 * dx;
                    let b = gray[(y0 + 1) * w + x0] as f32 * (1.0 - dx) + gray[(y0 + 1) * w + x0 + 1] as f32 * dx;
                    row[x] = a * (1.0 - dy) + b * dy;
                } else {
                    row[x] = sample_reflect(gray, w, h, sx, sy);
                }
            }
        });
    }
    let sig = (sigma * sigma - 0.5 * 0.5 * 4.0).max(0.01).sqrt();
    blur_buf(&dbl, bw, bh, sig)
}

fn sample_reflect(src: &[u8], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let v = |yy: i32, xx: i32| src[reflect101(yy, h as i32) as usize * w + reflect101(xx, w as i32) as usize] as f32;
    let a = v(y0, x0) * (1.0 - dx) + v(y0, x0 + 1) * dx;
    let b = v(y0 + 1, x0) * (1.0 - dx) + v(y0 + 1, x0 + 1) * dx;
    a * (1.0 - dy) + b * dy
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

fn blur(src: &Plane, sigma: f32) -> Plane {
    Plane { w: src.w, h: src.h, data: blur_buf(&src.data, src.w, src.h, sigma) }
}

fn blur_buf(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let mut ksize = (sigma * 8.0 + 1.0).round() as i32;
    ksize |= 1;
    if ksize < 1 {
        ksize = 1;
    }
    let ker = kernel(sigma, ksize as usize);
    blur_with_kernel(src, w, h, &ker, BlurEdge::Reflect)
}

#[derive(Clone, Copy)]
pub(super) enum BlurEdge {
    Reflect,
}

// Blurs each row, then each column, and only the edges ask where a tap lands.
pub(super) fn blur_with_kernel(src: &[f32], w: usize, h: usize, ker: &[f32], edge: BlurEdge) -> Vec<f32> {
    if w == 0 || h == 0 || ker.is_empty() {
        return vec![0f32; w.saturating_mul(h)];
    }
    let horiz = filter_rows(src, w, h, ker, edge);
    if h == 1 {
        return horiz;
    }
    filter_cols(&horiz, w, h, ker, edge)
}

fn filter_rows(src: &[f32], w: usize, h: usize, ker: &[f32], edge: BlurEdge) -> Vec<f32> {
    let r = ker.len() / 2;
    let wide = simd_rows();
    let mut dst = vec![0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src_row = &src[y * w..(y + 1) * w];
        conv_line(src_row, row, ker, r, edge, wide);
    });
    dst
}

fn filter_cols(src: &[f32], w: usize, h: usize, ker: &[f32], edge: BlurEdge) -> Vec<f32> {
    let r = ker.len() / 2;
    let wide = simd_rows();
    let mut dst = vec![0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if ker.len() > 64 {
            for x in 0..w {
                let mut s = 0f32;
                for i in 0..ker.len() {
                    let yy = edge_at(y as i32 + i as i32 - r as i32, h, edge);
                    s += src[yy * w + x] * ker[i];
                }
                row[x] = s;
            }
            return;
        }
        let mut ys = [0usize; 64];
        let k = ker.len();
        for i in 0..k {
            ys[i] = edge_at(y as i32 + i as i32 - r as i32, h, edge);
        }
        if wide && w >= 8 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                conv_cols_wide(src, w, &ys[..k], &ker[..k], row);
            }
            #[cfg(not(target_arch = "x86_64"))]
            conv_cols_scalar(src, w, &ys[..k], &ker[..k], row);
        } else {
            conv_cols_scalar(src, w, &ys[..k], &ker[..k], row);
        }
    });
    dst
}

fn conv_line(src_row: &[f32], row: &mut [f32], ker: &[f32], r: usize, edge: BlurEdge, wide: bool) {
    let w = row.len();
    let k = ker.len();
    let left = r.min(w);
    for x in 0..left {
        row[x] = dot_at(src_row, w, x, ker, r, edge);
    }
    let interior_end = w.saturating_sub(r);
    if wide && interior_end > left && k > 0 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            conv_line_wide(src_row, row, ker, r, left, interior_end);
        }
    } else if interior_end > left {
        for x in left..interior_end {
            let mut s = 0f32;
            let base = x - r;
            for i in 0..k {
                s += src_row[base + i] * ker[i];
            }
            row[x] = s;
        }
    }
    let tail = if wide && interior_end > left {
        left + ((interior_end - left) / 8) * 8
    } else {
        interior_end
    };
    for x in tail..interior_end {
        let mut s = 0f32;
        let base = x - r;
        for i in 0..k {
            s += src_row[base + i] * ker[i];
        }
        row[x] = s;
    }
    for x in interior_end.max(left)..w {
        row[x] = dot_at(src_row, w, x, ker, r, edge);
    }
}

fn dot_at(src_row: &[f32], w: usize, x: usize, ker: &[f32], r: usize, edge: BlurEdge) -> f32 {
    let mut s = 0f32;
    for i in 0..ker.len() {
        s += src_row[edge_at(x as i32 + i as i32 - r as i32, w, edge)] * ker[i];
    }
    s
}

fn conv_cols_scalar(src: &[f32], w: usize, ys: &[usize], ker: &[f32], row: &mut [f32]) {
    for x in 0..row.len() {
        let mut s = 0f32;
        for i in 0..ker.len() {
            s += src[ys[i] * w + x] * ker[i];
        }
        row[x] = s;
    }
}

fn simd_rows() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn conv_line_wide(src_row: &[f32], row: &mut [f32], ker: &[f32], r: usize, start: usize, end: usize) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps};
    let mut x = start;
    while x + 8 <= end {
        let mut acc = _mm256_setzero_ps();
        for i in 0..ker.len() {
            let v = unsafe { _mm256_loadu_ps(src_row.as_ptr().add(x - r + i)) };
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, _mm256_set1_ps(ker[i])));
        }
        unsafe { _mm256_storeu_ps(row.as_mut_ptr().add(x), acc) };
        x += 8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn conv_cols_wide(src: &[f32], w: usize, ys: &[usize], ker: &[f32], row: &mut [f32]) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps};
    let mut x = 0usize;
    let end = row.len();
    while x + 8 <= end {
        let mut acc = _mm256_setzero_ps();
        for i in 0..ker.len() {
            let v = unsafe { _mm256_loadu_ps(src.as_ptr().add(ys[i] * w + x)) };
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, _mm256_set1_ps(ker[i])));
        }
        unsafe { _mm256_storeu_ps(row.as_mut_ptr().add(x), acc) };
        x += 8;
    }
    for x in x..end {
        let mut s = 0f32;
        for i in 0..ker.len() {
            s += src[ys[i] * w + x] * ker[i];
        }
        row[x] = s;
    }
}

fn edge_at(p: i32, len: usize, edge: BlurEdge) -> usize {
    if p >= 0 && (p as usize) < len {
        return p as usize;
    }
    match edge {
        BlurEdge::Reflect => reflect101(p, len as i32) as usize,
    }
}

// Subtracts each blurred picture from the next one.
fn difference_planes(gauss: &[Plane], n_octaves: usize, layers: usize) -> Vec<Plane> {
    let mut dog = Vec::with_capacity(n_octaves * (layers + 2));
    for o in 0..n_octaves {
        for i in 0..(layers + 2) {
            let a = &gauss[o * (layers + 3) + i];
            let b = &gauss[o * (layers + 3) + i + 1];
            let mut data = vec![0f32; a.data.len()];
            let chunk = 8192usize;
            data.par_chunks_mut(chunk).enumerate().for_each(|(n, part)| {
                let start = n * chunk;
                for j in 0..part.len() {
                    part[j] = b.data[start + j] - a.data[start + j];
                }
            });
            dog.push(Plane { w: a.w, h: a.h, data });
        }
    }
    dog
}

fn kernel(sigma: f32, ksize: usize) -> Vec<f32> {
    let center = (ksize / 2) as i32;
    let scale = -0.5 / (sigma * sigma).max(1e-12);
    let mut k = vec![0f32; ksize];
    let mut sum = 0f32;
    for i in 0..ksize {
        let x = (i as i32 - center) as f32;
        let v = (scale * x * x).exp();
        k[i] = v;
        sum += v;
    }
    for v in k.iter_mut() {
        *v /= sum;
    }
    k
}

fn halve(src: &Plane) -> Plane {
    let w = src.w / 2;
    let h = src.h / 2;
    let mut data = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = src.data[(y * 2) * src.w + x * 2];
        }
    }
    Plane { w, h, data }
}

fn extrema(dog: &[Plane], gauss: &[Plane], octv: i32, layer: i32, idx: usize, threshold: f32, out: &mut Vec<KeyPoint>) {
    let img = &dog[idx];
    let prev = &dog[idx - 1];
    let next = &dog[idx + 1];
    let border = 5i32;
    let height = img.h as i32;
    let width = img.w as i32;
    if height <= border * 2 || width <= border * 2 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    let wide = std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let wide = false;
    let rows: Vec<Vec<KeyPoint>> = (border..(height - border))
        .into_par_iter()
        .map(|r| {
            let mut local = Vec::new();
            if wide {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    scan_row_wide(&mut local, img, prev, next, dog, gauss, octv, layer, r, border, width, threshold);
                }
            } else {
                for c in border..(width - border) {
                    accept_pixel(&mut local, img, prev, next, dog, gauss, octv, layer, r, c, threshold);
                }
            }
            local
        })
        .collect();
    for row in rows {
        out.extend(row);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_row_wide(
    local: &mut Vec<KeyPoint>,
    img: &Plane,
    prev: &Plane,
    next: &Plane,
    dog: &[Plane],
    gauss: &[Plane],
    octv: i32,
    layer: i32,
    r: i32,
    border: i32,
    width: i32,
    threshold: f32,
) {
    use std::arch::x86_64::{
        _mm256_and_ps, _mm256_andnot_ps, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_or_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _CMP_GT_OQ, _CMP_LT_OQ,
    };
    let thr = _mm256_set1_ps(threshold);
    let zero = _mm256_setzero_ps();
    let sign = _mm256_set1_ps(-0.0);
    let stride = img.w;
    let mut c = border;
    let limit = width - border;
    while c + 8 <= limit {
        let val = unsafe { _mm256_loadu_ps(img.data.as_ptr().add(r as usize * stride + c as usize)) };
        let hot = _mm256_cmp_ps(_mm256_andnot_ps(sign, val), thr, _CMP_GT_OQ);
        if _mm256_movemask_ps(hot) == 0 {
            c += 8;
            continue;
        }
        let mut above = _mm256_and_ps(hot, _mm256_cmp_ps(val, zero, _CMP_GT_OQ));
        let mut below = _mm256_and_ps(hot, _mm256_cmp_ps(val, zero, _CMP_LT_OQ));
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let nb = unsafe { _mm256_loadu_ps(img.data.as_ptr().add((r + dy) as usize * stride + (c + dx) as usize)) };
                above = _mm256_and_ps(above, _mm256_cmp_ps(val, nb, _CMP_GT_OQ));
                below = _mm256_and_ps(below, _mm256_cmp_ps(val, nb, _CMP_LT_OQ));
            }
        }
        let mask = _mm256_movemask_ps(_mm256_or_ps(above, below));
        if mask != 0 {
            for k in 0..8 {
                if mask & (1 << k) != 0 {
                    accept_pixel(local, img, prev, next, dog, gauss, octv, layer, r, c + k, threshold);
                }
            }
        }
        c += 8;
    }
    for c in c..limit {
        accept_pixel(local, img, prev, next, dog, gauss, octv, layer, r, c, threshold);
    }
}

fn accept_pixel(
    local: &mut Vec<KeyPoint>,
    img: &Plane,
    prev: &Plane,
    next: &Plane,
    dog: &[Plane],
    gauss: &[Plane],
    octv: i32,
    layer: i32,
    r: i32,
    c: i32,
    threshold: f32,
) {
    let val = img.at(r, c);
    if val.abs() <= threshold {
        return;
    }
    let mut is_max = val > 0.0;
    let mut is_min = val < 0.0;
    if !is_max && !is_min {
        return;
    }
    'nb: for dy in -1..=1 {
        for dx in -1..=1 {
            if dy == 0 && dx == 0 {
                continue;
            }
            let p = prev.at(r + dy, c + dx);
            let m = img.at(r + dy, c + dx);
            let n = next.at(r + dy, c + dx);
            if val <= p || val <= m || val <= n {
                is_max = false;
            }
            if val >= p || val >= m || val >= n {
                is_min = false;
            }
            if !is_max && !is_min {
                break 'nb;
            }
        }
    }
    if !(is_max || is_min) {
        return;
    }
    if let Some(k) = adjust(dog, octv, layer, r, c) {
        let ori = orientation(&gauss[((octv as usize) * ((NOCTAVE_LAYERS as usize) + 3)) + k.layer as usize], &k);
        for ang in ori {
            let mut kk = k.clone_pt();
            kk.angle = ang;
            local.push(kk);
        }
    }
}

struct RawKp {
    x: f32,
    y: f32,
    size: f32,
    response: f32,
    octave: i32,
    layer: i32,
    angle: f32,
}

impl RawKp {
    fn clone_pt(&self) -> KeyPoint {
        KeyPoint {
            x: self.x,
            y: self.y,
            size: self.size,
            angle: self.angle,
            response: self.response,
            octave: self.octave,
            layer: self.layer,
        }
    }
}

fn adjust(dog: &[Plane], octv: i32, mut layer: i32, mut r: i32, mut c: i32) -> Option<RawKp> {
    let img_scale = 1.0 / 255.0;
    let deriv_scale = img_scale * 0.5;
    let second = img_scale;
    let cross = img_scale * 0.25;
    let layers = NOCTAVE_LAYERS;
    let mut xi = 0.0f32;
    let mut xr = 0.0f32;
    let mut xc = 0.0f32;
    let mut i = 0;
    while i < 5 {
        let idx = (octv as usize) * ((layers as usize) + 2) + layer as usize;
        let img = &dog[idx];
        let prev = &dog[idx - 1];
        let next = &dog[idx + 1];
        let ddx = (img.at(r, c + 1) - img.at(r, c - 1)) * deriv_scale;
        let ddy = (img.at(r + 1, c) - img.at(r - 1, c)) * deriv_scale;
        let dds = (next.at(r, c) - prev.at(r, c)) * deriv_scale;
        let v2 = img.at(r, c) * 2.0;
        let dxx = (img.at(r, c + 1) + img.at(r, c - 1) - v2) * second;
        let dyy = (img.at(r + 1, c) + img.at(r - 1, c) - v2) * second;
        let dss = (next.at(r, c) + prev.at(r, c) - v2) * second;
        let dxy = (img.at(r + 1, c + 1) - img.at(r + 1, c - 1) - img.at(r - 1, c + 1) + img.at(r - 1, c - 1)) * cross;
        let dxs = (next.at(r, c + 1) - next.at(r, c - 1) - prev.at(r, c + 1) + prev.at(r, c - 1)) * cross;
        let dys = (next.at(r + 1, c) - next.at(r - 1, c) - prev.at(r + 1, c) + prev.at(r - 1, c)) * cross;
        let (sx, sy, ss) = solve3(dxx, dxy, dxs, dyy, dys, dss, ddx, ddy, dds);
        xc = -sx;
        xr = -sy;
        xi = -ss;
        if xc.abs() < 0.5 && xr.abs() < 0.5 && xi.abs() < 0.5 {
            break;
        }
        c += xc.round() as i32;
        r += xr.round() as i32;
        layer += xi.round() as i32;
        if layer < 1 || layer > layers || c < 5 || r < 5 || c >= img.w as i32 - 5 || r >= img.h as i32 - 5 {
            return None;
        }
        i += 1;
    }
    if i >= 5 {
        return None;
    }
    let idx = (octv as usize) * ((layers as usize) + 2) + layer as usize;
    let img = &dog[idx];
    let prev = &dog[idx - 1];
    let next = &dog[idx + 1];
    let ddx = (img.at(r, c + 1) - img.at(r, c - 1)) * deriv_scale;
    let ddy = (img.at(r + 1, c) - img.at(r - 1, c)) * deriv_scale;
    let dds = (next.at(r, c) - prev.at(r, c)) * deriv_scale;
    let t = ddx * xc + ddy * xr + dds * xi;
    let contr = img.at(r, c) * img_scale + t * 0.5;
    if contr.abs() * (layers as f32) < CONTRAST_THRESHOLD as f32 {
        return None;
    }
    let v2 = img.at(r, c) * 2.0;
    let dxx = (img.at(r, c + 1) + img.at(r, c - 1) - v2) * second;
    let dyy = (img.at(r + 1, c) + img.at(r - 1, c) - v2) * second;
    let dxy = (img.at(r + 1, c + 1) - img.at(r + 1, c - 1) - img.at(r - 1, c + 1) + img.at(r - 1, c - 1)) * cross;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    let edge = EDGE_THRESHOLD as f32;
    if det <= 0.0 || tr * tr * edge >= (edge + 1.0) * (edge + 1.0) * det {
        return None;
    }
    let scale = 1 << octv.max(0);
    Some(RawKp {
        x: (c as f32 + xc) * scale as f32,
        y: (r as f32 + xr) * scale as f32,
        size: (SIFT_SIGMA as f32) * 2f32.powf((layer as f32 + xi) / layers as f32) * scale as f32 * 2.0,
        response: contr.abs(),
        octave: octv,
        layer,
        angle: 0.0,
    })
}

fn solve3(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32, b0: f32, b1: f32, b2: f32) -> (f32, f32, f32) {
    let m = [
        [a00, a01, a02],
        [a01, a11, a12],
        [a02, a12, a22],
    ];
    let mut a = m;
    let mut b = [b0, b1, b2];
    for i in 0..3 {
        let mut piv = i;
        for r in (i + 1)..3 {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let d = a[i][i];
        if d.abs() < 1e-12 {
            return (0.0, 0.0, 0.0);
        }
        for r in (i + 1)..3 {
            let f = a[r][i] / d;
            for c in i..3 {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = [0f32; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for c in (i + 1)..3 {
            s -= a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }
    (x[0], x[1], x[2])
}

fn orientation(img: &Plane, k: &RawKp) -> Vec<f32> {
    let mut hist = [0f32; 36];
    let scl = k.size * 0.5 * 1.5;
    let radius = (4.5 * scl).round() as i32;
    let exp_scale = -1.0 / (2.0 * scl * scl);
    let pt = (k.x.round() as i32, k.y.round() as i32);
    for i in -radius..=radius {
        for j in -radius..=radius {
            let r = pt.1 + i;
            let c = pt.0 + j;
            if r <= 0 || c <= 0 || r >= img.h as i32 - 1 || c >= img.w as i32 - 1 {
                continue;
            }
            let dx = img.at(r, c + 1) - img.at(r, c - 1);
            let dy = img.at(r - 1, c) - img.at(r + 1, c);
            let mag = (dx * dx + dy * dy).sqrt();
            let w = ((j * j + i * i) as f32 * exp_scale).exp();
            let mut ang = dy.atan2(dx).to_degrees();
            if ang < 0.0 {
                ang += 360.0;
            }
            let bin = (ang * 36.0 / 360.0).floor() as usize % 36;
            hist[bin] += mag * w;
        }
    }
    let mut smooth = [0f32; 36];
    for i in 0..36 {
        smooth[i] = (hist[(i + 34) % 36] + hist[(i + 2) % 36]) / 16.0
            + (hist[(i + 35) % 36] + hist[(i + 1) % 36]) * 4.0 / 16.0
            + hist[i] * 6.0 / 16.0;
    }
    let maxv = smooth.iter().cloned().fold(0f32, f32::max);
    let mut angs = Vec::new();
    for i in 0..36 {
        let l = smooth[(i + 35) % 36];
        let r = smooth[(i + 1) % 36];
        if smooth[i] > l && smooth[i] > r && smooth[i] >= 0.8 * maxv {
            let mut bin = i as f32 + 0.5 * (l - r) / (l - 2.0 * smooth[i] + r);
            if bin < 0.0 {
                bin += 36.0;
            }
            if bin >= 36.0 {
                bin -= 36.0;
            }
            angs.push(360.0 - bin * 10.0);
            if angs.len() == 2 {
                break;
            }
        }
    }
    if angs.is_empty() {
        angs.push(0.0);
    }
    angs
}

fn descriptor(img: &Plane, ptf: [f32; 2], ori: f32, scl: f32) -> [f32; 128] {
    let d = 4i32;
    let n = 8i32;
    let pt = (ptf[0].round() as i32, ptf[1].round() as i32);
    let cos_t = ori.to_radians().cos();
    let sin_t = ori.to_radians().sin();
    let bins_per_rad = n as f32 / 360.0;
    let exp_scale = -1.0 / (d as f32 * d as f32 * 0.5);
    let hist_width = 3.0 * scl;
    let mut radius = (hist_width * std::f32::consts::SQRT_2 * (d as f32 + 1.0) * 0.5).round() as i32;
    let diag = ((img.w * img.w + img.h * img.h) as f32).sqrt() as i32;
    radius = radius.min(diag);
    let cos_t = cos_t / hist_width;
    let sin_t = sin_t / hist_width;
    let mut hist = vec![0f32; ((d + 2) * (d + 2) * (n + 2)) as usize];
    for i in -radius..=radius {
        for j in -radius..=radius {
            let c_rot = j as f32 * cos_t - i as f32 * sin_t;
            let r_rot = j as f32 * sin_t + i as f32 * cos_t;
            let mut rbin = r_rot + d as f32 / 2.0 - 0.5;
            let mut cbin = c_rot + d as f32 / 2.0 - 0.5;
            let r = pt.1 + i;
            let c = pt.0 + j;
            if rbin > -1.0 && rbin < d as f32 && cbin > -1.0 && cbin < d as f32 && r > 0 && r < img.h as i32 - 1 && c > 0 && c < img.w as i32 - 1 {
                let dx = img.at(r, c + 1) - img.at(r, c - 1);
                let dy = img.at(r - 1, c) - img.at(r + 1, c);
                let mag = (dx * dx + dy * dy).sqrt();
                let mut ang = dy.atan2(dx).to_degrees();
                if ang < 0.0 {
                    ang += 360.0;
                }
                let wgt = ((c_rot * c_rot + r_rot * r_rot) * exp_scale).exp() * mag;
                let mut obin = (ang - ori) * bins_per_rad;
                let r0 = rbin.floor() as i32;
                let c0 = cbin.floor() as i32;
                let mut o0 = obin.floor() as i32;
                rbin -= r0 as f32;
                cbin -= c0 as f32;
                obin -= o0 as f32;
                if o0 < 0 {
                    o0 += n;
                }
                if o0 >= n {
                    o0 -= n;
                }
                let v_r1 = wgt * rbin;
                let v_r0 = wgt - v_r1;
                let v_rc11 = v_r1 * cbin;
                let v_rc10 = v_r1 - v_rc11;
                let v_rc01 = v_r0 * cbin;
                let v_rc00 = v_r0 - v_rc01;
                let packs = [
                    (v_rc00 * (1.0 - obin), 0),
                    (v_rc00 * obin, 1),
                    (v_rc01 * (1.0 - obin), n + 2),
                    (v_rc01 * obin, n + 3),
                    (v_rc10 * (1.0 - obin), (d + 2) * (n + 2)),
                    (v_rc10 * obin, (d + 2) * (n + 2) + 1),
                    (v_rc11 * (1.0 - obin), (d + 3) * (n + 2)),
                    (v_rc11 * obin, (d + 3) * (n + 2) + 1),
                ];
                let idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
                for (val, off) in packs {
                    let p = (idx + off) as usize;
                    if p < hist.len() {
                        hist[p] += val;
                    }
                }
            }
        }
    }
    let mut raw = [0f32; 128];
    for i in 0..d {
        for j in 0..d {
            let idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
            hist[idx as usize] += hist[(idx + n) as usize];
            hist[(idx + 1) as usize] += hist[(idx + n + 1) as usize];
            for k in 0..n {
                raw[(i * d + j) as usize * 8 + k as usize] = hist[(idx + k) as usize];
            }
        }
    }
    let mut nrm2 = raw.iter().map(|v| v * v).sum::<f32>();
    let thr = nrm2.sqrt() * 0.2;
    nrm2 = 0.0;
    for v in raw.iter_mut() {
        *v = v.min(thr);
        nrm2 += *v * *v;
    }
    let scale = 512.0 / nrm2.sqrt().max(1e-12);
    let mut dst = [0f32; 128];
    for i in 0..128 {
        dst[i] = (raw[i] * scale).round().clamp(0.0, 255.0);
    }
    dst
}

fn dedup(kpts: &mut Vec<KeyPoint>) {
    use std::collections::HashMap;
    let mut bins: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    let mut keep = vec![true; kpts.len()];
    let q = |v: f32| (v * 1000.0).round() as i32;
    for i in 0..kpts.len() {
        let key = (q(kpts[i].x), q(kpts[i].y), q(kpts[i].size));
        let mut drop = false;
        'near: for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(prev) = bins.get(&(key.0 + dx, key.1 + dy, key.2 + dz)) else {
                        continue;
                    };
                    for &j in prev {
                        if (kpts[i].x - kpts[j].x).abs() < 1e-3
                            && (kpts[i].y - kpts[j].y).abs() < 1e-3
                            && (kpts[i].size - kpts[j].size).abs() < 1e-3
                        {
                            drop = true;
                            break 'near;
                        }
                    }
                }
            }
        }
        if drop {
            keep[i] = false;
        } else {
            bins.entry(key).or_default().push(i);
        }
    }
    let mut i = 0;
    kpts.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });
}
