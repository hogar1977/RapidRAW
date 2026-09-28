use super::const_::BA_HUBER;
use super::geom::{bearings, strong};
use super::lens::LensModel;
use super::rng::NumpyRng;
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

pub fn bundle_rotations(
    pair_rot: &[(usize, usize, Matrix3<f64>, Vec<[f64; 2]>, Vec<[f64; 2]>, usize, usize)],
    root: usize,
    parent: &[Option<usize>],
    order: &[usize],
    w: u32,
    h: u32,
    f: f64,
    lens: &LensModel,
) -> Vec<Matrix3<f64>> {
    let n = parent.len();
    let mut init = vec![Matrix3::identity(); n];
    for &v in order.iter().skip(1) {
        let u = parent[v].unwrap_or(root);
        let rel = pair_rot.iter().find(|(a, b, ..)| (*a == u && *b == v) || (*a == v && *b == u));
        let r_uv = if let Some((a, _, rot, ..)) = rel {
            if *a == u { *rot } else { rot.transpose() }
        } else {
            Matrix3::identity()
        };
        init[v] = init[u] * r_uv.transpose();
    }
    let free: Vec<usize> = order.iter().copied().filter(|&i| i != root).collect();
    if free.is_empty() {
        return init;
    }
    let mut rng = NumpyRng::seed0();
    let mut samples: Vec<(usize, usize, Vec<Vector3<f64>>, Vec<Vector3<f64>>)> = Vec::new();
    let gate = 1.5 * super::const_::ransac_px(w, h) / f;
    for (a, b, rot, pa, pb, inl, spread) in pair_rot {
        if parent.get(*a).is_none() && *a != root {
            continue;
        }
        if !strong(*inl, *spread) {
            continue;
        }
        let ba = bearings(pa, f, w, h, lens);
        let bb = bearings(pb, f, w, h, lens);
        let mut idx = Vec::new();
        for i in 0..ba.len() {
            let cosang = ((rot * ba[i]).dot(&bb[i])).clamp(-1.0, 1.0);
            if cosang.acos() < gate {
                idx.push(i);
            }
        }
        if idx.len() < 6 {
            continue;
        }
        if idx.len() > super::const_::BA_MAX_PER_PAIR {
            let pick = rng.choice(idx.len(), super::const_::BA_MAX_PER_PAIR);
            let mut chosen: Vec<usize> = pick.into_iter().map(|k| idx[k]).collect();
            chosen.sort_unstable();
            idx = chosen;
        }
        samples.push((
            *a,
            *b,
            idx.iter().map(|&i| ba[i]).collect(),
            idx.iter().map(|&i| bb[i]).collect(),
        ));
    }
    let mut x = DVector::zeros(free.len() * 3);
    for (k, &nm) in free.iter().enumerate() {
        let rv = Rotation3::from_matrix_unchecked(init[nm]).scaled_axis();
        x[3 * k] = rv.x;
        x[3 * k + 1] = rv.y;
        x[3 * k + 2] = rv.z;
    }
    if samples.is_empty() {
        return init;
    }
    let unpack = |x: &DVector<f64>| {
        let mut rs = vec![Matrix3::identity(); n];
        rs[root] = Matrix3::identity();
        for (k, &nm) in free.iter().enumerate() {
            let rv = Vector3::new(x[3 * k], x[3 * k + 1], x[3 * k + 2]);
            rs[nm] = Rotation3::from_scaled_axis(rv).into_inner();
        }
        rs
    };
    let residual = |x: &DVector<f64>| {
        let rs = unpack(x);
        let mut out = Vec::new();
        for (a, b, va, vb) in &samples {
            for i in 0..va.len() {
                let d = rs[*a] * va[i] - rs[*b] * vb[i];
                out.push(d.x);
                out.push(d.y);
                out.push(d.z);
            }
        }
        DVector::from_vec(out)
    };
    let mut xv = x;
    for _ in 0..super::const_::BA_MAX_EVALS {
        if super::trace::halted() {
            break;
        }
        let r = residual(&xv);
        if r.nrows() == 0 {
            break;
        }
        let eps = 1e-6;
        let mut j = DMatrix::zeros(r.nrows(), xv.nrows());
        for c in 0..xv.nrows() {
            let mut xp = xv.clone();
            xp[c] += eps;
            let rp = residual(&xp);
            for row in 0..r.nrows() {
                j[(row, c)] = (rp[row] - r[row]) / eps;
            }
        }
        let mut w = DVector::from_element(r.nrows(), 1.0);
        for i in 0..r.nrows() {
            let a = r[i].abs();
            if a > BA_HUBER {
                w[i] = BA_HUBER / a;
            }
        }
        let mut jw = j.clone();
        let mut rw = r.clone();
        for i in 0..r.nrows() {
            let s = (w[i] as f64).sqrt();
            rw[i] *= s;
            for c in 0..xv.nrows() {
                jw[(i, c)] *= s;
            }
        }
        let jt = jw.transpose();
        let mut h = jt.clone() * &jw;
        for i in 0..h.nrows() {
            h[(i, i)] += 1e-6;
        }
        let g = jt * rw;
        let step = match h.lu().solve(&g) {
            Some(s) => s,
            None => break,
        };
        if step.norm() < 1e-8 {
            break;
        }
        xv -= step;
    }
    unpack(&xv)
}
