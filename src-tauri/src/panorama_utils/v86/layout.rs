use super::geom::{frame_fov_deg, yaw_pitch_deg};
use nalgebra::Matrix3;

pub struct Layout {
    pub rows: Vec<Vec<usize>>,
    pub yaws: Vec<f64>,
    pub pitches: Vec<f64>,
    pub yaw_span: f64,
    pub pitch_span: f64,
    pub frame_hfov: f64,
    pub frame_vfov: f64,
}

// Groups the photos into rows and columns from how they are aimed.
pub fn detect_layout(rots: &[Matrix3<f64>], f: f64, w: u32, h: u32) -> Layout {
    let (frame_hfov, frame_vfov) = frame_fov_deg(f, w, h);
    let mut yaws = Vec::with_capacity(rots.len());
    let mut pitches = Vec::with_capacity(rots.len());
    for r in rots {
        let (y, p) = yaw_pitch_deg(r);
        yaws.push(y);
        pitches.push(p);
    }
    unwrap_yaws(&mut yaws);
    if rots.is_empty() {
        return Layout {
            rows: Vec::new(),
            yaws,
            pitches,
            yaw_span: 0.0,
            pitch_span: 0.0,
            frame_hfov,
            frame_vfov,
        };
    }
    let mut by_pitch: Vec<usize> = (0..rots.len()).collect();
    by_pitch.sort_by(|&a, &b| pitches[a].partial_cmp(&pitches[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut cur = vec![by_pitch[0]];
    for wdw in by_pitch.windows(2) {
        let a = wdw[0];
        let b = wdw[1];
        if pitches[b] - pitches[a] > 0.4 * frame_vfov {
            rows.push(std::mem::take(&mut cur));
        }
        cur.push(b);
    }
    rows.push(cur);
    for row in &mut rows {
        row.sort_by(|&a, &b| yaws[a].partial_cmp(&yaws[b]).unwrap_or(std::cmp::Ordering::Equal));
    }
    let yaw_span = yaws.iter().cloned().fold(f64::MIN, f64::max) - yaws.iter().cloned().fold(f64::MAX, f64::min);
    let pitch_span = pitches.iter().cloned().fold(f64::MIN, f64::max) - pitches.iter().cloned().fold(f64::MAX, f64::min);
    Layout { rows, yaws, pitches, yaw_span, pitch_span, frame_hfov, frame_vfov }
}

fn unwrap_yaws(yaws: &mut [f64]) {
    if yaws.len() <= 1 {
        if let Some(y) = yaws.get_mut(0) {
            *y = 0.0;
        }
        return;
    }
    let mut ys = yaws.to_vec();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut best_i = 0usize;
    let mut best_g = -1.0;
    for i in 0..ys.len() {
        let g = (ys[(i + 1) % ys.len()] - ys[i]).rem_euclid(360.0);
        if g > best_g {
            best_g = g;
            best_i = i;
        }
    }
    let start = ys[(best_i + 1) % ys.len()];
    for y in yaws.iter_mut() {
        *y = (*y - start).rem_euclid(360.0);
    }
}

// Chooses the order to paste photos, starting in the middle and growing outward.
pub fn composite_order(layout: &Layout, kept: &[usize], fallback: &[usize]) -> Vec<usize> {
    let kept_set: std::collections::HashSet<usize> = kept.iter().copied().collect();
    let rows: Vec<Vec<usize>> = layout
        .rows
        .iter()
        .map(|r| r.iter().copied().filter(|n| kept_set.contains(n)).collect())
        .filter(|r: &Vec<usize>| !r.is_empty())
        .collect();
    let mut pos = std::collections::HashMap::new();
    for (ri, row) in rows.iter().enumerate() {
        for (ci, n) in row.iter().enumerate() {
            pos.insert(*n, (ri, ci));
        }
    }
    let names: Vec<usize> = rows.iter().flatten().copied().collect();
    if names.is_empty() {
        return fallback.iter().copied().filter(|n| kept_set.contains(n)).collect();
    }
    let mut ys: Vec<f64> = names.iter().map(|&n| layout.yaws[n]).collect();
    let mut ps: Vec<f64> = names.iter().map(|&n| layout.pitches[n]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid_y = ys[ys.len() / 2];
    let mid_p = ps[ps.len() / 2];
    let center_d = |n: usize| (layout.yaws[n] - mid_y).abs() + (layout.pitches[n] - mid_p).abs();
    let mut seed = *names.iter().min_by(|a, b| center_d(**a).partial_cmp(&center_d(**b)).unwrap()).unwrap();
    if let Some(&root) = fallback.first() {
        if pos.contains_key(&root) && center_d(root) <= center_d(seed) {
            seed = root;
        }
    }
    let ang = |a: usize, b: usize| (layout.yaws[a] - layout.yaws[b]).abs() + (layout.pitches[a] - layout.pitches[b]).abs();
    let neighbors = |n: usize| -> Vec<usize> {
        let (ri, ci) = pos[&n];
        let row = &rows[ri];
        let mut out = Vec::new();
        if ci > 0 {
            out.push(row[ci - 1]);
        }
        if ci + 1 < row.len() {
            out.push(row[ci + 1]);
        }
        let yaw_n = layout.yaws[n];
        for rj in [ri as i32 - 1, ri as i32 + 1] {
            if rj < 0 || rj as usize >= rows.len() {
                continue;
            }
            let other = &rows[rj as usize];
            let pick = *other.iter().min_by(|a, b| {
                (layout.yaws[**a] - yaw_n).abs().partial_cmp(&(layout.yaws[**b] - yaw_n).abs()).unwrap()
            }).unwrap();
            out.push(pick);
        }
        out
    };
    let mut order = vec![seed];
    let mut seen = std::collections::HashSet::from([seed]);
    let mut unused: std::collections::HashSet<usize> = pos.keys().copied().filter(|n| *n != seed).collect();
    while !unused.is_empty() {
        let mut front = Vec::new();
        for &n in &seen {
            for c in neighbors(n) {
                if unused.contains(&c) {
                    front.push(c);
                }
            }
        }
        let pick = if !front.is_empty() {
            *front.iter().min_by(|a, b| ang(**a, seed).partial_cmp(&ang(**b, seed)).unwrap().then(a.cmp(b))).unwrap()
        } else {
            let rest: Vec<usize> = rows.iter().flatten().copied().filter(|n| unused.contains(n)).collect();
            *rest.iter().min_by(|a, b| ang(**a, seed).partial_cmp(&ang(**b, seed)).unwrap()).unwrap()
        };
        order.push(pick);
        seen.insert(pick);
        unused.remove(&pick);
    }
    for &n in fallback {
        if kept_set.contains(&n) && seen.insert(n) {
            order.push(n);
        }
    }
    order
}
