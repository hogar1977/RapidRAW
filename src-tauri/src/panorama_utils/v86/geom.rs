use super::const_::{self, FOCAL_CANDIDATES, MIN_INL, MIN_SPREAD, RANSAC_ITERS, RANSAC_ITERS_FOCAL};
use super::lens::LensModel;
use super::rng::NumpyRng;
use nalgebra::{Matrix3, Vector3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Perspective,
    Cylindrical,
    Spherical,
}

impl Projection {
    pub fn as_str(self) -> &'static str {
        match self {
            Projection::Perspective => "perspective",
            Projection::Cylindrical => "cylindrical",
            Projection::Spherical => "spherical",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "perspective" => Some(Projection::Perspective),
            "cylindrical" => Some(Projection::Cylindrical),
            "spherical" => Some(Projection::Spherical),
            _ => None,
        }
    }
}

pub struct PairGeom {
    pub rot: Matrix3<f64>,
    pub inliers: usize,
    pub spread: usize,
    pub pa: Vec<[f64; 2]>,
    pub pb: Vec<[f64; 2]>,
}

pub fn bearings(pts: &[[f64; 2]], f: f64, w: u32, h: u32, lens: &LensModel) -> Vec<Vector3<f64>> {
    let hd = ((w as f64) * 0.5).hypot((h as f64) * 0.5);
    pts.iter()
        .map(|p| {
            let mut dx = p[0] - w as f64 / 2.0;
            let mut dy = p[1] - h as f64 / 2.0;
            if !lens.is_identity() {
                let rd = dx.hypot(dy);
                let ru = lens.invert_radius(rd / hd) * hd;
                let s = if rd > 1e-9 { ru / rd.max(1e-9) } else { 1.0 };
                dx *= s;
                dy *= s;
            }
            let v = Vector3::new(dx / f, dy / f, 1.0);
            let n = v.norm();
            if n > 0.0 { v / n } else { v }
        })
        .collect()
}

pub fn kabsch(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> Option<Matrix3<f64>> {
    if a.len() < 2 || a.len() != b.len() {
        return None;
    }
    let mut h = Matrix3::zeros();
    for (av, bv) in a.iter().zip(b.iter()) {
        h += av * bv.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u?;
    let mut vt = svd.v_t?;
    let mut r = vt.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        vt.row_mut(2).scale_mut(-1.0);
        r = vt.transpose() * u.transpose();
    }
    Some(r)
}

pub fn ransac_rotation(
    a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    f: f64,
    pa: &[[f64; 2]],
    w: u32,
    h: u32,
    iters: usize,
) -> Option<(Matrix3<f64>, usize, usize, Vec<bool>)> {
    let mut rng = NumpyRng::seed0();
    let n = a.len();
    if n < 6 || b.len() != n {
        return None;
    }
    let thr = const_::ransac_px(w, h) / f;
    let mut best_count = 0usize;
    let mut best_mask = vec![false; n];
    for _ in 0..iters {
        let sample = rng.choice(n, 3);
        if sample.len() < 3 {
            continue;
        }
        let sa: Vec<_> = sample.iter().map(|&i| a[i]).collect();
        let sb: Vec<_> = sample.iter().map(|&i| b[i]).collect();
        let Some(rot) = kabsch(&sa, &sb) else { continue };
        let mut mask = vec![false; n];
        let mut count = 0usize;
        for i in 0..n {
            let proj = rot * a[i];
            let cosang = proj.dot(&b[i]).clamp(-1.0, 1.0);
            if cosang.acos() < thr {
                mask[i] = true;
                count += 1;
            }
        }
        if count > best_count {
            best_count = count;
            best_mask = mask;
        }
    }
    if best_count < 6 {
        return None;
    }
    let ia: Vec<_> = best_mask.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| a[i]).collect();
    let ib: Vec<_> = best_mask.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| b[i]).collect();
    let rot = kabsch(&ia, &ib)?;
    let mut occ = std::collections::HashSet::new();
    for (i, on) in best_mask.iter().enumerate() {
        if *on {
            let x = pa[i][0];
            let y = pa[i][1];
            let cx = ((x / w as f64) * 6.0) as i32;
            let cy = ((y / h as f64) * 4.0) as i32;
            occ.insert((cx.clamp(0, 5), cy.clamp(0, 3)));
        }
    }
    Some((rot, best_count, occ.len(), best_mask))
}

pub fn strong(inl: usize, spread: usize) -> bool {
    inl >= MIN_INL && (spread >= MIN_SPREAD || inl >= 5 * MIN_INL)
}

pub fn largest_component(names: &[String], pairs: &[(usize, usize, usize, usize)]) -> Vec<usize> {
    let n = names.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for &(a, b, inl, spread) in pairs {
        if !strong(inl, spread) {
            continue;
        }
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        groups[find(&mut parent, i)].push(i);
    }
    groups
        .into_iter()
        .filter(|g| !g.is_empty())
        .max_by_key(|g| (g.len(), usize::MAX - g[0]))
        .unwrap_or_default()
}

pub struct Tree {
    pub root: usize,
    pub parent: Vec<Option<usize>>,
    pub order: Vec<usize>,
}

pub fn spanning_tree(kept: &[usize], edges: &[(usize, usize, usize)]) -> Tree {
    let mut parent_uf: std::collections::HashMap<usize, usize> = kept.iter().copied().map(|i| (i, i)).collect();
    fn find(p: &mut std::collections::HashMap<usize, usize>, mut x: usize) -> usize {
        while p[&x] != x {
            let g = p[&p[&x]];
            p.insert(x, g);
            x = g;
        }
        x
    }
    let mut sorted = edges.to_vec();
    sorted.sort_by(|a, b| b.2.cmp(&a.2));
    let mut adj: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        kept.iter().copied().map(|i| (i, Vec::new())).collect();
    for (a, b, w) in sorted {
        let ra = find(&mut parent_uf, a);
        let rb = find(&mut parent_uf, b);
        if ra != rb {
            parent_uf.insert(ra, rb);
            adj.get_mut(&a).unwrap().push((b, w));
            adj.get_mut(&b).unwrap().push((a, w));
        }
    }
    let strength = |n: usize| adj[&n].iter().map(|(_, w)| *w).sum::<usize>();
    let root = *kept.iter().max_by_key(|n| strength(**n)).unwrap_or(&kept[0]);
    let mut parent = vec![None; kept.iter().copied().max().unwrap_or(0) + 1];
    let mut seen = std::collections::HashSet::new();
    let mut order = vec![root];
    seen.insert(root);
    let mut q = std::collections::VecDeque::from([root]);
    while let Some(u) = q.pop_front() {
        let mut nbrs = adj[&u].clone();
        nbrs.sort_by(|a, b| b.1.cmp(&a.1));
        for (v, _) in nbrs {
            if seen.insert(v) {
                parent[v] = Some(u);
                order.push(v);
                q.push_back(v);
            }
        }
    }
    Tree { root, parent, order }
}

pub fn estimate_focal(
    pairs: &[(&[[f64; 2]], &[[f64; 2]])],
    w: u32,
    h: u32,
    lens: &LensModel,
) -> (f64, f64) {
    let ranked: Vec<_> = pairs.iter().map(|(a, b)| a.len().min(b.len())).collect();
    let mut idx: Vec<usize> = (0..pairs.len()).collect();
    idx.sort_by(|&i, &j| ranked[j].cmp(&ranked[i]));
    idx.retain(|&i| pairs[i].0.len() >= 12);
    idx.truncate(4);
    if idx.is_empty() {
        let f35 = 50.0;
        return (const_::focal_px_from_35eq(w, h, f35), f35);
    }
    let score = |f35: f64| -> (usize, f64) {
        let f = const_::focal_px_from_35eq(w, h, f35);
        let mut tot = 0usize;
        let mut err = 0.0;
        for &i in &idx {
            let (pa, pb) = pairs[i];
            let ba = bearings(pa, f, w, h, lens);
            let bb = bearings(pb, f, w, h, lens);
            let Some((rot, inl, _, mask)) = ransac_rotation(&ba, &bb, f, pa, w, h, RANSAC_ITERS_FOCAL) else {
                continue;
            };
            let mut sum = 0.0;
            let mut n = 0usize;
            for (k, on) in mask.iter().enumerate() {
                if *on {
                    let d = ((rot * ba[k]) - bb[k]).norm() * f;
                    sum += d;
                    n += 1;
                }
            }
            tot += inl;
            if n > 0 {
                err += (sum / n as f64) * inl as f64;
            }
        }
        (tot, if tot > 0 { err / tot as f64 } else { 1e9 })
    };
    let mut best = FOCAL_CANDIDATES[0];
    let mut best_s = score(best);
    for &c in &FOCAL_CANDIDATES[1..] {
        let s = score(c);
        if s.0 > best_s.0 || (s.0 == best_s.0 && s.1 < best_s.1) {
            best = c;
            best_s = s;
        }
    }
    let i = FOCAL_CANDIDATES.iter().position(|c| *c == best).unwrap_or(0);
    let lo = FOCAL_CANDIDATES[i.saturating_sub(1)];
    let hi = FOCAL_CANDIDATES[(i + 1).min(FOCAL_CANDIDATES.len() - 1)];
    let mut best_f = best;
    let mut best_fs = best_s;
    for k in 0..9 {
        let t = k as f64 / 8.0;
        let c = (lo.ln() + (hi.ln() - lo.ln()) * t).exp();
        let s = score(c);
        if s.0 > best_fs.0 || (s.0 == best_fs.0 && s.1 < best_fs.1) {
            best_f = c;
            best_fs = s;
        }
    }
    let _ = ranked;
    (const_::focal_px_from_35eq(w, h, best_f), best_f)
}

pub fn yaw_pitch_deg(rot: &Matrix3<f64>) -> (f64, f64) {
    let ray = rot * Vector3::new(0.0, 0.0, 1.0);
    let yaw = ray.x.atan2(ray.z).to_degrees();
    let pitch = ray.y.atan2(ray.x.hypot(ray.z)).to_degrees();
    (yaw, pitch)
}

pub fn frame_fov_deg(f: f64, w: u32, h: u32) -> (f64, f64) {
    let hf = 2.0 * ((w as f64) * 0.5 / f).atan().to_degrees();
    let vf = 2.0 * ((h as f64) * 0.5 / f).atan().to_degrees();
    (hf, vf)
}

pub fn recommend_projection(yaws: &[f64], pitches: &[f64], frame_hfov: f64, frame_vfov: f64) -> Projection {
    let (ymin, ymax) = span(yaws);
    let (pmin, pmax) = span(pitches);
    let total_h = (ymax - ymin) + frame_hfov;
    let total_v = (pmax - pmin) + frame_vfov;
    if total_h <= const_::PERSPECTIVE_HFOV_MAX && total_v <= const_::PERSPECTIVE_VFOV_MAX {
        Projection::Perspective
    } else if total_h >= const_::SPHERICAL_HFOV_MIN || total_v >= const_::SPHERICAL_VFOV_MIN {
        Projection::Spherical
    } else {
        Projection::Cylindrical
    }
}

fn span(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mut lo = v[0];
    let mut hi = v[0];
    for x in v {
        lo = lo.min(*x);
        hi = hi.max(*x);
    }
    (lo, hi)
}

pub fn proj_forward(x: f64, y: f64, z: f64, projection: Projection) -> (f64, f64) {
    match projection {
        Projection::Perspective => {
            let zc = z.max((const_::PERSPECTIVE_Z_MIN_DEG.to_radians()).cos());
            (x / zc, y / zc)
        }
        Projection::Cylindrical => (x.atan2(z), y / x.hypot(z).max(1e-12)),
        Projection::Spherical => (x.atan2(z), y.atan2(x.hypot(z))),
    }
}

pub fn proj_inverse(theta: f64, hh: f64, projection: Projection) -> Vector3<f64> {
    match projection {
        Projection::Perspective => Vector3::new(theta, hh, 1.0),
        Projection::Cylindrical => {
            let n = (1.0 + hh * hh).sqrt();
            Vector3::new(theta.sin() / n, hh / n, theta.cos() / n)
        }
        Projection::Spherical => Vector3::new(
            theta.sin() * hh.cos(),
            hh.sin(),
            theta.cos() * hh.cos(),
        ),
    }
}

pub fn center_order(yaws: &[f64], pitches: &[f64]) -> Vec<usize> {
    let n = yaws.len();
    if n == 0 {
        return Vec::new();
    }
    let my = yaws.iter().sum::<f64>() / n as f64;
    let mp = pitches.iter().sum::<f64>() / n as f64;
    let mut start = 0usize;
    let mut best = f64::MAX;
    for i in 0..n {
        let d = (yaws[i] - my).hypot(pitches[i] - mp);
        if d < best {
            best = d;
            start = i;
        }
    }
    let mut order = vec![start];
    let mut used = vec![false; n];
    used[start] = true;
    while order.len() < n {
        let mut pick = None;
        let mut best_d = f64::MAX;
        for &u in &order {
            for v in 0..n {
                if used[v] {
                    continue;
                }
                let d = (yaws[u] - yaws[v]).abs() + (pitches[u] - pitches[v]).abs();
                if d < best_d {
                    best_d = d;
                    pick = Some(v);
                }
            }
        }
        if let Some(v) = pick {
            used[v] = true;
            order.push(v);
        } else {
            break;
        }
    }
    order
}

pub fn default_ransac_iters() -> usize {
    RANSAC_ITERS
}
