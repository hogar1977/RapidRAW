use crate::panorama_utils::camera::CameraPose;
use crate::panorama_utils::match_graph::PairEdge;
use crate::panorama_utils::orb::OrbFeature;
use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use std::collections::{HashMap, HashSet};

const MAX_ITERS: usize = 50;
const LAMBDA_INIT: f64 = 1e-2;
/// Huber threshold on bidirectional transfer residual (pixels).
const HUBER_DELTA: f64 = 2.5;
const ROLL_PRIOR_WEIGHT: f64 = 12.0;
const OUTLIER_REJECTION_ITERS: usize = 2;
/// Emphasize vertical registration (kills windowblind ridge steps).
const VERTICAL_RESIDUAL_WEIGHT: f64 = 2.25;

pub fn bundle_adjust(
    poses: &mut [CameraPose],
    features: &[Vec<OrbFeature>],
    edges: &[PairEdge],
    kept: &[usize],
    initial_rotations: &HashMap<usize, Rotation3<f64>>,
) {
    if kept.len() < 2 {
        return;
    }

    for &idx in kept {
        if let Some(r) = initial_rotations.get(&idx) {
            poses[idx].set_rotation(*r);
        }
    }

    let anchor = kept[0];
    poses[anchor].set_rotation(Rotation3::identity());

    let mut param_index: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    for &idx in kept {
        if idx == anchor {
            continue;
        }
        param_index.insert(idx, next);
        next += 3;
    }
    let shared_focal_param = next;
    next += 1;
    let n_params = next;

    let mean_focal = kept.iter().map(|&i| poses[i].focal_px).sum::<f64>() / kept.len() as f64;
    for &idx in kept {
        poses[idx].focal_px = mean_focal;
    }

    let mut x = DVector::zeros(n_params);
    for (&cam, &base) in &param_index {
        x[base] = poses[cam].rotation_vector[0];
        x[base + 1] = poses[cam].rotation_vector[1];
        x[base + 2] = poses[cam].rotation_vector[2];
    }
    x[shared_focal_param] = mean_focal;

    let mut observations = collect_observations(edges, features, kept);
    if observations.is_empty() {
        apply_params(&x, poses, &param_index, shared_focal_param, anchor, kept);
        return;
    }

    for round in 0..=OUTLIER_REJECTION_ITERS {
        let mut lambda = LAMBDA_INIT;
        let mut best = cost(&x, poses, &param_index, shared_focal_param, anchor, kept, &observations);

        for _ in 0..MAX_ITERS {
            let residuals =
                residuals_vec(&x, poses, &param_index, shared_focal_param, anchor, kept, &observations);
            let jac = jacobian(
                &x,
                poses,
                &param_index,
                shared_focal_param,
                anchor,
                kept,
                &observations,
                &residuals,
            );
            let weights = huber_weights(&residuals);
            let mut jtj: DMatrix<f64> = DMatrix::zeros(n_params, n_params);
            let mut jtr: DVector<f64> = DVector::zeros(n_params);
            for i in 0..residuals.len() {
                let w = weights[i];
                let ri = residuals[i];
                for p in 0..n_params {
                    jtr[p] += w * jac[(i, p)] * ri;
                    for q in p..n_params {
                        let v = w * jac[(i, p)] * jac[(i, q)];
                        jtj[(p, q)] += v;
                        if p != q {
                            jtj[(q, p)] += v;
                        }
                    }
                }
            }
            // Soft prior: discourage actual camera roll (not axis-angle z).
            for (&cam, &base) in &param_index {
                let mut tmp = poses[cam];
                tmp.rotation_vector = [x[base], x[base + 1], x[base + 2]];
                let roll = crate::panorama_utils::camera::camera_roll_rad(&tmp.rotation());
                // Push axis-angle toward lower roll via finite-diff on the residual term.
                jtr[base + 2] += ROLL_PRIOR_WEIGHT * roll * 0.15;
                jtj[(base + 2, base + 2)] += ROLL_PRIOR_WEIGHT * 0.15;
                let _ = cam;
            }

            let mut a = jtj;
            for i in 0..n_params {
                a[(i, i)] += lambda * a[(i, i)].abs().max(1e-6);
            }
            let Some(delta) = a.lu().solve(&jtr) else {
                break;
            };
            let x_new = &x - delta;
            let c = cost(
                &x_new,
                poses,
                &param_index,
                shared_focal_param,
                anchor,
                kept,
                &observations,
            );
            if c < best {
                x = x_new;
                best = c;
                lambda = (lambda * 0.4).max(1e-8);
                if jtr.norm() < 1e-5 {
                    break;
                }
            } else {
                lambda = (lambda * 6.0).min(1e6);
            }
        }

        if round == OUTLIER_REJECTION_ITERS {
            break;
        }
        let before = observations.len();
        observations = reject_outlier_observations(
            &x,
            poses,
            &param_index,
            shared_focal_param,
            anchor,
            kept,
            &observations,
        );
        if observations.len() < 8 || observations.len() == before {
            break;
        }
    }

    apply_params(&x, poses, &param_index, shared_focal_param, anchor, kept);
}

struct Obs {
    cam_a: usize,
    cam_b: usize,
    xa: f64,
    ya: f64,
    xb: f64,
    yb: f64,
}

fn collect_observations(
    edges: &[PairEdge],
    features: &[Vec<OrbFeature>],
    kept: &[usize],
) -> Vec<Obs> {
    collect_observations_from(edges, features, kept, false)
}

/// Prefer dense flow matches (parallax-aware) when polishing pitch / local geometry.
fn collect_observations_flow(
    edges: &[PairEdge],
    features: &[Vec<OrbFeature>],
    kept: &[usize],
) -> Vec<Obs> {
    collect_observations_from(edges, features, kept, true)
}

fn collect_observations_from(
    edges: &[PairEdge],
    features: &[Vec<OrbFeature>],
    kept: &[usize],
    prefer_flow: bool,
) -> Vec<Obs> {
    let kept_set: HashSet<usize> = kept.iter().copied().collect();
    let mut out = Vec::new();
    for e in edges {
        if !kept_set.contains(&e.i) || !kept_set.contains(&e.j) {
            continue;
        }
        let pairs: &[(usize, usize)] = if prefer_flow && e.flow_pairs.len() >= 12 {
            &e.flow_pairs
        } else {
            &e.inliers
        };
        let step = if prefer_flow {
            (pairs.len() / 250).max(1)
        } else {
            1
        };
        for (k, &(ia, ib)) in pairs.iter().enumerate() {
            if k % step != 0 {
                continue;
            }
            if ia >= features[e.i].len() || ib >= features[e.j].len() {
                continue;
            }
            let ka = features[e.i][ia].keypoint;
            let kb = features[e.j][ib].keypoint;
            out.push(Obs {
                cam_a: e.i,
                cam_b: e.j,
                xa: ka.x as f64,
                ya: ka.y as f64,
                xb: kb.x as f64,
                yb: kb.y as f64,
            });
        }
    }
    out
}

fn apply_params(
    x: &DVector<f64>,
    poses: &mut [CameraPose],
    param_index: &HashMap<usize, usize>,
    shared_focal_param: usize,
    anchor: usize,
    kept: &[usize],
) {
    poses[anchor].set_rotation(Rotation3::identity());
    let focal = x[shared_focal_param].max(10.0);
    for (&cam, &base) in param_index {
        poses[cam].rotation_vector = [x[base], x[base + 1], x[base + 2]];
    }
    for &idx in kept {
        poses[idx].focal_px = focal;
    }
}

fn residual(poses: &[CameraPose], obs: &Obs) -> f64 {
    // Bidirectional pixel transfer — much stronger FOV/pitch leverage than ray angle.
    let wa = poses[obs.cam_a].world_bearing_from_pixel(obs.xa, obs.ya);
    let wb = poses[obs.cam_b].world_bearing_from_pixel(obs.xb, obs.yb);
    let mut err2 = 0.0;
    let mut n = 0usize;
    if let Some((x, y)) = poses[obs.cam_b].pixel_from_world_bearing(wa) {
        let dx = x - obs.xb;
        let dy = (y - obs.yb) * VERTICAL_RESIDUAL_WEIGHT;
        err2 += dx * dx + dy * dy;
        n += 1;
    }
    if let Some((x, y)) = poses[obs.cam_a].pixel_from_world_bearing(wb) {
        let dx = x - obs.xa;
        let dy = (y - obs.ya) * VERTICAL_RESIDUAL_WEIGHT;
        err2 += dx * dx + dy * dy;
        n += 1;
    }
    if n == 0 {
        return wa.angle(&wb) * poses[obs.cam_a].focal_px.max(1.0);
    }
    (err2 / n as f64).sqrt()
}

fn huber_weights(residuals: &DVector<f64>) -> Vec<f64> {
    residuals
        .iter()
        .map(|&r| {
            let ar = r.abs();
            if ar <= HUBER_DELTA {
                1.0
            } else {
                HUBER_DELTA / ar
            }
        })
        .collect()
}

fn reject_outlier_observations(
    x: &DVector<f64>,
    poses: &[CameraPose],
    param_index: &HashMap<usize, usize>,
    shared_focal_param: usize,
    anchor: usize,
    kept: &[usize],
    observations: &[Obs],
) -> Vec<Obs> {
    with_params(x, poses, param_index, shared_focal_param, anchor, kept, |p| {
        let errs: Vec<f64> = observations.iter().map(|o| residual(p, o)).collect();
        let mut sorted = errs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let thresh = (median * 3.5).max(2.0).min(14.0);
        observations
            .iter()
            .zip(errs.iter())
            .filter(|&(_, e)| *e <= thresh)
            .map(|(o, _)| Obs {
                cam_a: o.cam_a,
                cam_b: o.cam_b,
                xa: o.xa,
                ya: o.ya,
                xb: o.xb,
                yb: o.yb,
            })
            .collect()
    })
}

fn with_params<F, R>(
    x: &DVector<f64>,
    poses: &[CameraPose],
    param_index: &HashMap<usize, usize>,
    shared_focal_param: usize,
    anchor: usize,
    kept: &[usize],
    f: F,
) -> R
where
    F: FnOnce(&[CameraPose]) -> R,
{
    let mut tmp = poses.to_vec();
    apply_params(x, &mut tmp, param_index, shared_focal_param, anchor, kept);
    f(&tmp)
}

fn cost(
    x: &DVector<f64>,
    poses: &[CameraPose],
    param_index: &HashMap<usize, usize>,
    shared_focal_param: usize,
    anchor: usize,
    kept: &[usize],
    observations: &[Obs],
) -> f64 {
    with_params(x, poses, param_index, shared_focal_param, anchor, kept, |p| {
        let data: f64 = observations
            .iter()
            .map(|o| {
                let r = residual(p, o);
                if r.abs() <= HUBER_DELTA {
                    0.5 * r * r
                } else {
                    HUBER_DELTA * (r.abs() - 0.5 * HUBER_DELTA)
                }
            })
            .sum();
        let mut prior = 0.0;
        for (&cam, &base) in param_index {
            let mut tmp = p[cam];
            tmp.rotation_vector = [x[base], x[base + 1], x[base + 2]];
            let roll = crate::panorama_utils::camera::camera_roll_rad(&tmp.rotation());
            prior += 0.5 * ROLL_PRIOR_WEIGHT * roll * roll;
        }
        data + prior
    })
}

fn residuals_vec(
    x: &DVector<f64>,
    poses: &[CameraPose],
    param_index: &HashMap<usize, usize>,
    shared_focal_param: usize,
    anchor: usize,
    kept: &[usize],
    observations: &[Obs],
) -> DVector<f64> {
    with_params(x, poses, param_index, shared_focal_param, anchor, kept, |p| {
        DVector::from_iterator(observations.len(), observations.iter().map(|o| residual(p, o)))
    })
}

fn jacobian(
    x: &DVector<f64>,
    poses: &[CameraPose],
    param_index: &HashMap<usize, usize>,
    shared_focal_param: usize,
    anchor: usize,
    kept: &[usize],
    observations: &[Obs],
    base: &DVector<f64>,
) -> DMatrix<f64> {
    let n = x.len();
    let m = observations.len();
    let mut jac = DMatrix::zeros(m, n);
    let eps = 1e-5;
    for p in 0..n {
        let mut xp = x.clone();
        xp[p] += eps;
        let rp = residuals_vec(&xp, poses, param_index, shared_focal_param, anchor, kept, observations);
        for i in 0..m {
            jac[(i, p)] = (rp[i] - base[i]) / eps;
        }
    }
    jac
}

pub fn refine_two_camera_yaw(yaw_rad: f64, focal_px: f64, width: u32, height: u32) -> [CameraPose; 2] {
    let mut poses = [
        CameraPose::pinhole(focal_px, width, height),
        CameraPose::pinhole(focal_px, width, height),
    ];
    poses[1].rotation_vector = [0.0, yaw_rad, 0.0];
    let _ = Vector3::<f64>::zeros();
    poses
}


/// Enforce strictly monotonic yaws along sorted `kept` indices (1×N pan direction).
pub fn enforce_monotonic_yaws(yaws: &mut [f64], sign: f64) {
    enforce_monotonic_yaws_with_step(yaws, sign, 0.35f64.to_radians());
}

pub fn enforce_monotonic_yaws_with_step(yaws: &mut [f64], sign: f64, min_step: f64) {
    if yaws.len() < 2 {
        return;
    }
    let step_min = min_step.max(0.05f64.to_radians());
    let s = if sign >= 0.0 { 1.0 } else { -1.0 };
    for i in 0..yaws.len() - 1 {
        let step = yaws[i + 1] - yaws[i];
        if step * s < step_min {
            yaws[i + 1] = yaws[i] + step_min * s;
        }
    }
}

fn pan_sign_from_yaws(yaws: &[f64]) -> f64 {
    if yaws.len() < 2 {
        return -1.0;
    }
    let mut pos = 0usize;
    let mut neg = 0usize;
    for w in yaws.windows(2) {
        let d = w[1] - w[0];
        if d > 0.0 {
            pos += 1;
        } else if d < 0.0 {
            neg += 1;
        }
    }
    if pos >= neg {
        1.0
    } else {
        -1.0
    }
}

fn median_abs_step(yaws: &[f64]) -> f64 {
    if yaws.len() < 2 {
        return 3.0f64.to_radians();
    }
    let mut steps: Vec<f64> = yaws.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    steps[steps.len() / 2].max(1.0f64.to_radians())
}

/// After single-row flatten: polish yaw only (shared focal stays at EXIF bootstrap).
/// Joint focal+yaw LM was collapsing the row (focal→∞, yaws→min_step) on parallax sets.
pub fn refine_single_row_yaws(
    poses: &mut [CameraPose],
    features: &[Vec<OrbFeature>],
    edges: &[PairEdge],
    kept: &[usize],
) {
    use crate::panorama_utils::camera::{rotation_from_yaw_pitch, yaw_pitch_from_rotation};

    if kept.len() < 2 {
        return;
    }
    let observations = collect_observations(edges, features, kept);
    if observations.is_empty() {
        return;
    }

    let mut order = kept.to_vec();
    order.sort_unstable();
    let anchor = order[0];
    let (_, shared_pitch) = yaw_pitch_from_rotation(&poses[anchor].rotation());

    let mut yaw_index: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    for &idx in &order {
        if idx == anchor {
            continue;
        }
        yaw_index.insert(idx, next);
        next += 1;
    }
    let n_params = next;
    if n_params == 0 {
        return;
    }

    let mut init_yaws: Vec<f64> = Vec::with_capacity(order.len());
    init_yaws.push(0.0);
    for &idx in order.iter().skip(1) {
        let (yaw, _) = yaw_pitch_from_rotation(&poses[idx].rotation());
        init_yaws.push(yaw);
    }
    let sign = pan_sign_from_yaws(&init_yaws);
    let min_step = median_abs_step(&init_yaws) * 0.45;
    enforce_monotonic_yaws_with_step(&mut init_yaws, sign, min_step);

    let mean_focal = kept.iter().map(|&i| poses[i].focal_px).sum::<f64>() / kept.len() as f64;
    for &idx in kept {
        poses[idx].focal_px = mean_focal;
    }

    let mut x = DVector::zeros(n_params);
    for (i, &idx) in order.iter().enumerate() {
        if let Some(&yi) = yaw_index.get(&idx) {
            x[yi] = init_yaws[i];
        }
    }

    let prior_yaws = init_yaws.clone();
    const YAW_PRIOR_WEIGHT: f64 = 40.0;

    let project_x = |x: &mut DVector<f64>| {
        let mut yaws = vec![0.0f64; order.len()];
        for (i, &idx) in order.iter().enumerate() {
            if idx == anchor {
                yaws[i] = 0.0;
            } else if let Some(&yi) = yaw_index.get(&idx) {
                yaws[i] = x[yi];
            }
        }
        enforce_monotonic_yaws_with_step(&mut yaws, sign, min_step);
        let prior_span = (prior_yaws.last().unwrap_or(&0.0) - prior_yaws[0]).abs();
        let span = (yaws.last().unwrap_or(&0.0) - yaws[0]).abs();
        if prior_span > 1e-6 && span < prior_span * 0.5 {
            let scale = (prior_span * 0.5) / span.max(1e-9);
            let y0 = yaws[0];
            for y in yaws.iter_mut().skip(1) {
                *y = y0 + (*y - y0) * scale;
            }
            enforce_monotonic_yaws_with_step(&mut yaws, sign, min_step);
        }
        for (i, &idx) in order.iter().enumerate() {
            if let Some(&yi) = yaw_index.get(&idx) {
                x[yi] = yaws[i];
            }
        }
    };

    let apply_yaw = |x: &DVector<f64>, poses: &mut [CameraPose]| {
        poses[anchor].set_rotation(rotation_from_yaw_pitch(0.0, shared_pitch));
        poses[anchor].focal_px = mean_focal;
        for (&cam, &yi) in &yaw_index {
            poses[cam].set_rotation(rotation_from_yaw_pitch(x[yi], shared_pitch));
            poses[cam].focal_px = mean_focal;
        }
    };

    let cost_yaw = |x: &DVector<f64>, poses: &[CameraPose]| -> f64 {
        let mut tmp = poses.to_vec();
        apply_yaw(x, &mut tmp);
        let mut c: f64 = observations
            .iter()
            .map(|o| {
                let r = residual(&tmp, o);
                if r.abs() <= HUBER_DELTA {
                    0.5 * r * r
                } else {
                    HUBER_DELTA * (r.abs() - 0.5 * HUBER_DELTA)
                }
            })
            .sum();
        for (i, &idx) in order.iter().enumerate() {
            if let Some(&yi) = yaw_index.get(&idx) {
                let d = x[yi] - prior_yaws[i];
                c += 0.5 * YAW_PRIOR_WEIGHT * d * d;
            }
        }
        c
    };

    let mut lambda = LAMBDA_INIT;
    project_x(&mut x);
    let mut best = cost_yaw(&x, poses);
    for _ in 0..MAX_ITERS {
        let mut tmp = poses.to_vec();
        apply_yaw(&x, &mut tmp);
        let base_res = DVector::from_iterator(
            observations.len(),
            observations.iter().map(|o| residual(&tmp, o)),
        );
        let mut jac: DMatrix<f64> = DMatrix::zeros(observations.len(), n_params);
        let eps = 1e-5;
        for p in 0..n_params {
            let mut xp = x.clone();
            xp[p] += eps;
            project_x(&mut xp);
            let mut tp = poses.to_vec();
            apply_yaw(&xp, &mut tp);
            for i in 0..observations.len() {
                jac[(i, p)] = (residual(&tp, &observations[i]) - base_res[i]) / eps;
            }
        }
        let weights = huber_weights(&base_res);
        let mut jtj: DMatrix<f64> = DMatrix::zeros(n_params, n_params);
        let mut jtr: DVector<f64> = DVector::zeros(n_params);
        for i in 0..base_res.len() {
            let w = weights[i];
            let ri = base_res[i];
            for p in 0..n_params {
                jtr[p] += w * jac[(i, p)] * ri;
                for q in p..n_params {
                    let v = w * jac[(i, p)] * jac[(i, q)];
                    jtj[(p, q)] += v;
                    if p != q {
                        jtj[(q, p)] += v;
                    }
                }
            }
        }
        for (i, &idx) in order.iter().enumerate() {
            if let Some(&yi) = yaw_index.get(&idx) {
                jtr[yi] += YAW_PRIOR_WEIGHT * (x[yi] - prior_yaws[i]);
                jtj[(yi, yi)] += YAW_PRIOR_WEIGHT;
            }
        }
        let mut a = jtj;
        for i in 0..n_params {
            a[(i, i)] += lambda * a[(i, i)].abs().max(1e-6);
        }
        let Some(delta) = a.lu().solve(&jtr) else {
            break;
        };
        let mut x_new = &x - delta;
        project_x(&mut x_new);
        let c = cost_yaw(&x_new, poses);
        if c < best {
            x = x_new;
            best = c;
            lambda = (lambda * 0.4).max(1e-8);
            if jtr.norm() < 1e-5 {
                break;
            }
        } else {
            lambda = (lambda * 6.0).min(1e6);
        }
    }
    project_x(&mut x);
    apply_yaw(&x, poses);
}


/// Freeze yaw from the chain; polish per-camera pitch (and tiny roll) so ridges meet.
/// Prior LM on yaw collapsed the pan span — pitch-only is safe for 1×N handheld.
pub fn refine_single_row_pitches(
    poses: &mut [CameraPose],
    features: &[Vec<OrbFeature>],
    edges: &[PairEdge],
    kept: &[usize],
) {
    use crate::panorama_utils::camera::{rotation_from_yaw_pitch, yaw_pitch_from_rotation};

    if kept.len() < 2 {
        return;
    }
    let observations = collect_observations_flow(edges, features, kept);
    if observations.is_empty() {
        return;
    }

    let mut order = kept.to_vec();
    order.sort_unstable();
    let mut yaws = Vec::with_capacity(order.len());
    let mut pitches0 = Vec::with_capacity(order.len());
    for &idx in &order {
        let (y, p) = yaw_pitch_from_rotation(&poses[idx].rotation());
        yaws.push(y);
        pitches0.push(p);
    }
    let shared = {
        let mut s = pitches0.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s[s.len() / 2]
    };
    const MAX_PITCH_DELTA: f64 = 2.5 * std::f64::consts::PI / 180.0;
    const PITCH_PRIOR: f64 = 25.0;

    // Params: pitch offset from shared for each camera except anchor (index 0 in order).
    let n_pitch = order.len() - 1;
    if n_pitch == 0 {
        return;
    }
    let mut x = DVector::zeros(n_pitch);
    for i in 1..order.len() {
        x[i - 1] = (pitches0[i] - shared).clamp(-MAX_PITCH_DELTA, MAX_PITCH_DELTA);
    }

    let apply = |x: &DVector<f64>, poses: &mut [CameraPose]| {
        for (i, &idx) in order.iter().enumerate() {
            let pitch = if i == 0 {
                shared
            } else {
                shared + x[i - 1].clamp(-MAX_PITCH_DELTA, MAX_PITCH_DELTA)
            };
            poses[idx].set_rotation(rotation_from_yaw_pitch(yaws[i], pitch));
        }
    };

    let cost_fn = |x: &DVector<f64>, poses: &[CameraPose]| -> f64 {
        let mut tmp = poses.to_vec();
        apply(x, &mut tmp);
        let mut c: f64 = observations
            .iter()
            .map(|o| {
                let r = residual(&tmp, o);
                if r.abs() <= HUBER_DELTA {
                    0.5 * r * r
                } else {
                    HUBER_DELTA * (r.abs() - 0.5 * HUBER_DELTA)
                }
            })
            .sum();
        for i in 0..n_pitch {
            c += 0.5 * PITCH_PRIOR * x[i] * x[i];
        }
        c
    };

    let mut lambda = LAMBDA_INIT;
    let mut best = cost_fn(&x, poses);
    for _ in 0..MAX_ITERS {
        let mut tmp = poses.to_vec();
        apply(&x, &mut tmp);
        let base_res = DVector::from_iterator(
            observations.len(),
            observations.iter().map(|o| residual(&tmp, o)),
        );
        let mut jac: DMatrix<f64> = DMatrix::zeros(observations.len(), n_pitch);
        let eps = 1e-5;
        for p in 0..n_pitch {
            let mut xp = x.clone();
            xp[p] = (xp[p] + eps).clamp(-MAX_PITCH_DELTA, MAX_PITCH_DELTA);
            let mut tp = poses.to_vec();
            apply(&xp, &mut tp);
            for i in 0..observations.len() {
                jac[(i, p)] = (residual(&tp, &observations[i]) - base_res[i]) / eps;
            }
        }
        let weights = huber_weights(&base_res);
        let mut jtj: DMatrix<f64> = DMatrix::zeros(n_pitch, n_pitch);
        let mut jtr: DVector<f64> = DVector::zeros(n_pitch);
        for i in 0..base_res.len() {
            let w = weights[i];
            let ri = base_res[i];
            for p in 0..n_pitch {
                jtr[p] += w * jac[(i, p)] * ri;
                for q in p..n_pitch {
                    let v = w * jac[(i, p)] * jac[(i, q)];
                    jtj[(p, q)] += v;
                    if p != q {
                        jtj[(q, p)] += v;
                    }
                }
            }
        }
        for p in 0..n_pitch {
            jtr[p] += PITCH_PRIOR * x[p];
            jtj[(p, p)] += PITCH_PRIOR;
        }
        let mut a = jtj;
        for i in 0..n_pitch {
            a[(i, i)] += lambda * a[(i, i)].abs().max(1e-6);
        }
        let Some(delta) = a.lu().solve(&jtr) else {
            break;
        };
        let mut x_new = &x - delta;
        for i in 0..n_pitch {
            x_new[i] = x_new[i].clamp(-MAX_PITCH_DELTA, MAX_PITCH_DELTA);
        }
        let c = cost_fn(&x_new, poses);
        if c < best {
            x = x_new;
            best = c;
            lambda = (lambda * 0.4).max(1e-8);
            if jtr.norm() < 1e-5 {
                break;
            }
        } else {
            lambda = (lambda * 6.0).min(1e6);
        }
    }
    apply(&x, poses);
    for (i, &idx) in order.iter().enumerate() {
        let pitch = if i == 0 {
            shared
        } else {
            shared + x[i - 1]
        };
        crate::panorama_utils::debug_log::write(&format!(
            "pitch_polish[{}] yaw={:.3}° pitch={:.3}° (shared={:.3}°)",
            idx,
            yaws[i].to_degrees(),
            pitch.to_degrees(),
            shared.to_degrees()
        ));
    }
}
