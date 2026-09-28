use super::const_::LOWE_RATIO;
use super::sift::Feature;
use rayon::prelude::*;

pub struct PairMatches {
    pub a: usize,
    pub b: usize,
    pub pa: Vec<[f64; 2]>,
    pub pb: Vec<[f64; 2]>,
}

// Compares each photo with the next two in the list. A photo with no real overlap is compared with the rest.
pub fn match_all(feats: &[Vec<Feature>]) -> Vec<PairMatches> {
    let n = feats.len();
    let near = if n <= 8 { 1..=n.saturating_sub(1) } else { 1..=2 };
    let mut todo: Vec<(usize, usize)> = Vec::new();
    for a in 0..n {
        for d in near.clone() {
            if a + d < n {
                todo.push((a, a + d));
            }
        }
    }
    if n > 8 {
        super::trace::line(&format!("matching {} neighbor pairs", todo.len()));
    }
    let mut out = match_pairs(feats, &todo);
    if n <= 8 {
        return out;
    }
    let mut best = vec![0usize; n];
    for pair in &out {
        best[pair.a] = best[pair.a].max(pair.pa.len());
        best[pair.b] = best[pair.b].max(pair.pa.len());
    }
    let mut extra = Vec::new();
    for a in 0..n {
        if best[a] >= 100 {
            continue;
        }
        for b in 0..n {
            if a == b || b.abs_diff(a) <= 2 {
                continue;
            }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if extra.contains(&(lo, hi)) || todo.contains(&(lo, hi)) {
                continue;
            }
            extra.push((lo, hi));
        }
    }
    if !extra.is_empty() {
        super::trace::line(&format!("matching {} extra pairs", extra.len()));
        out.extend(match_pairs(feats, &extra));
    }
    out
}

fn match_pairs(feats: &[Vec<Feature>], pairs: &[(usize, usize)]) -> Vec<PairMatches> {
    let log = super::trace::current();
    pairs
        .par_iter()
        .map(|&(a, b)| {
            let _guard = log.as_ref().map(|log| super::trace::install(log.clone()));
            if feats[a].len() < 6 || feats[b].len() < 6 || super::trace::halted() {
                return PairMatches { a, b, pa: Vec::new(), pb: Vec::new() };
            }
            let (pa, pb) = match_pair(&feats[a], &feats[b]);
            PairMatches { a, b, pa, pb }
        })
        .collect()
}

fn match_pair(a: &[Feature], b: &[Feature]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let ratio2 = LOWE_RATIO * LOWE_RATIO;
    let mut pa = Vec::new();
    let mut pb = Vec::new();
    for fa in a {
        if super::trace::halted() {
            break;
        }
        let mut best = f32::MAX;
        let mut second = f32::MAX;
        let mut best_i = 0usize;
        for (i, fb) in b.iter().enumerate() {
            let d = dist2(&fa.desc, &fb.desc);
            if d < best {
                second = best;
                best = d;
                best_i = i;
            } else if d < second {
                second = d;
            }
        }
        if second < f32::MAX && best < ratio2 * second {
            pa.push(fa.pt);
            pb.push(b[best_i].pt);
        }
    }
    (pa, pb)
}

fn dist2(a: &[f32; 128], b: &[f32; 128]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { dist2_wide(a, b) };
    }
    let mut s0 = 0f32;
    let mut s1 = 0f32;
    let mut s2 = 0f32;
    let mut s3 = 0f32;
    let mut i = 0;
    while i < 128 {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        i += 4;
    }
    s0 + s1 + s2 + s3
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist2_wide(a: &[f32; 128], b: &[f32; 128]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_sub_ps,
        _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
    };
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < 128 {
        let d = _mm256_sub_ps(unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) }, unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) });
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
        i += 8;
    }
    let low = _mm256_castps256_ps128(acc);
    let high = _mm256_extractf128_ps(acc, 1);
    let sum = _mm_add_ps(low, high);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}
