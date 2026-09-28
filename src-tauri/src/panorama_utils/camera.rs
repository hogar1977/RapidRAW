use crate::lens_correction::{
    find_best_lens_match, resolve_lens_params, LensDatabase, LensDistortionParams,
};
use nalgebra::{Matrix3, Rotation3, SVD, Unit, Vector3};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const DEFAULT_FOCAL_MM_35EQ: f64 = 50.0;
/// Reject EXIF focals outside this 35mm-equivalent band (mm).
const MIN_FOCAL_MM_35EQ: f64 = 8.0;
const MAX_FOCAL_MM_35EQ: f64 = 400.0;
const SENSOR_WIDTH_35MM: f64 = 36.0;
const RANSAC_ITERS: usize = 2000;
const RANSAC_ANGLE_THRESH_RAD: f64 = 0.045;
const MIN_ROTATION_INLIERS: usize = 6;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LensModel {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    /// 0 = poly3, 1 = ptlens
    pub model: u32,
}

impl Default for LensModel {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            model: 0,
        }
    }
}

impl LensModel {
    pub fn from_params(p: &LensDistortionParams) -> Self {
        Self {
            k1: p.k1,
            k2: p.k2,
            k3: p.k3,
            model: p.model,
        }
    }

    pub fn is_identity(self) -> bool {
        self.k1.abs() < 1e-9 && self.k2.abs() < 1e-9 && self.k3.abs() < 1e-9
    }

    /// Radial scale: undistorted_r_norm → distorted_r_norm / undistorted (forward).
    fn forward_scale(self, ru_norm: f64) -> f64 {
        let r2 = ru_norm * ru_norm;
        if self.model == 1 {
            let a = self.k1;
            let b = self.k2;
            let c = self.k3;
            let d = 1.0 - a - b - c;
            a * ru_norm * r2 + b * r2 + c * ru_norm + d
        } else {
            1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        }
    }

    /// Invert forward distortion: distorted normalized radius → undistorted.
    fn invert_radius(self, rd_norm: f64) -> f64 {
        if self.is_identity() || rd_norm < 1e-12 {
            return rd_norm;
        }
        let mut ru = rd_norm;
        for _ in 0..12 {
            let s = self.forward_scale(ru);
            let pred = ru * s;
            let err = pred - rd_norm;
            if err.abs() < 1e-10 {
                break;
            }
            // d(pred)/d(ru) ≈ s + ru * s'
            let eps = 1e-6;
            let s2 = self.forward_scale(ru + eps);
            let ds = (s2 - s) / eps;
            let deriv = (s + ru * ds).max(1e-6);
            ru = (ru - err / deriv).max(0.0);
        }
        ru
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraPose {
    pub rotation_vector: [f64; 3],
    pub focal_px: f64,
    pub width: u32,
    pub height: u32,
    pub lens: LensModel,
}

impl CameraPose {
    pub fn pinhole(focal_px: f64, width: u32, height: u32) -> Self {
        Self {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px,
            width,
            height,
            lens: LensModel::default(),
        }
    }

    pub fn rotation(&self) -> Rotation3<f64> {
        let v = Vector3::new(
            self.rotation_vector[0],
            self.rotation_vector[1],
            self.rotation_vector[2],
        );
        let angle = v.norm();
        if angle < 1e-12 {
            Rotation3::identity()
        } else {
            Rotation3::from_axis_angle(&Unit::new_normalize(v), angle)
        }
    }

    pub fn set_rotation(&mut self, r: Rotation3<f64>) {
        let aa = r.scaled_axis();
        self.rotation_vector = [aa.x, aa.y, aa.z];
    }

    fn half_diag(&self) -> f64 {
        let cx = self.width as f64 * 0.5;
        let cy = self.height as f64 * 0.5;
        (cx * cx + cy * cy).sqrt().max(1.0)
    }

    /// Distorted image pixel → ideal camera bearing (lens undistorted).
    pub fn pixel_to_bearing(&self, x: f64, y: f64) -> Vector3<f64> {
        let cx = self.width as f64 * 0.5;
        let cy = self.height as f64 * 0.5;
        let mut dx = x - cx;
        let mut dy = cy - y; // y-up
        if !self.lens.is_identity() {
            let hd = self.half_diag();
            let rd = (dx * dx + dy * dy).sqrt();
            if rd > 1e-9 {
                let rd_norm = rd / hd;
                let ru_norm = self.lens.invert_radius(rd_norm);
                let scale = ru_norm / rd_norm;
                dx *= scale;
                dy *= scale;
            }
        }
        Vector3::new(dx, dy, self.focal_px).normalize()
    }

    /// Ideal bearing → distorted image pixel (for sampling the raw frame).
    pub fn bearing_to_pixel(&self, bearing: Vector3<f64>) -> Option<(f64, f64)> {
        if bearing.z <= 1e-8 {
            return None;
        }
        let cx = self.width as f64 * 0.5;
        let cy = self.height as f64 * 0.5;
        let mut dx = bearing.x / bearing.z * self.focal_px;
        let mut dy = bearing.y / bearing.z * self.focal_px;
        if !self.lens.is_identity() {
            let hd = self.half_diag();
            let ru = (dx * dx + dy * dy).sqrt();
            if ru > 1e-9 {
                let ru_norm = ru / hd;
                let scale = self.lens.forward_scale(ru_norm);
                dx *= scale;
                dy *= scale;
            }
        }
        let x = dx + cx;
        let y = cy - dy;
        Some((x, y))
    }

    pub fn world_bearing_from_pixel(&self, x: f64, y: f64) -> Vector3<f64> {
        self.rotation() * self.pixel_to_bearing(x, y)
    }

    pub fn pixel_from_world_bearing(&self, world: Vector3<f64>) -> Option<(f64, f64)> {
        let local = self.rotation().inverse() * world;
        self.bearing_to_pixel(local)
    }

    pub fn with_scaled_size(&self, new_w: u32, new_h: u32) -> Self {
        let scale = new_w as f64 / self.width.max(1) as f64;
        Self {
            rotation_vector: self.rotation_vector,
            focal_px: self.focal_px * scale,
            width: new_w,
            height: new_h,
            lens: self.lens,
        }
    }
}

pub fn fov_rad_from_focal_mm_35eq(focal_mm: f64) -> f64 {
    let f = if focal_mm > 1e-3 {
        focal_mm
    } else {
        DEFAULT_FOCAL_MM_35EQ
    };
    2.0 * (SENSOR_WIDTH_35MM / (2.0 * f)).atan()
}

pub fn focal_px_from_fov(fov_rad: f64, width: u32) -> f64 {
    (width as f64 * 0.5) / (fov_rad * 0.5).tan()
}

fn parse_mm(s: &str) -> Option<f64> {
    let cleaned = s.replace(" mm", "").replace("mm", "").trim().to_string();
    if cleaned.contains('/') {
        let parts: Vec<&str> = cleaned.split('/').collect();
        if parts.len() == 2 {
            let n = parts[0].trim().parse::<f64>().ok()?;
            let d = parts[1].trim().parse::<f64>().ok()?;
            if d.abs() > 1e-9 {
                return Some(n / d);
            }
        }
        return None;
    }
    cleaned.parse::<f64>().ok()
}

fn sanitize_focal_mm_35eq(v: f64) -> Option<f64> {
    if v.is_finite() && (MIN_FOCAL_MM_35EQ..=MAX_FOCAL_MM_35EQ).contains(&v) {
        Some(v)
    } else {
        None
    }
}

/// Bootstrap 35mm-equivalent focal length for panorama FOV.
/// Missing / absurd EXIF → [`DEFAULT_FOCAL_MM_35EQ`] (BA may refine `focal_px` later).
pub fn parse_focal_mm_35eq(exif: &HashMap<String, String>) -> f64 {
    let fl = exif.get("FocalLength").and_then(|s| parse_mm(s));
    let fl35 = exif
        .get("FocalLengthIn35mmFilm")
        .and_then(|s| parse_mm(s))
        .and_then(sanitize_focal_mm_35eq);
    let scale = exif
        .get("ScaleFactor35efl")
        .or_else(|| exif.get("ScaleFactor35Efl"))
        .and_then(|s| parse_mm(s))
        .filter(|s| *s > 0.5 && *s < 10.0);

    let candidate = match (fl, fl35) {
        // Prefer a real, distinct 35mm-equivalent tag whenever present.
        (Some(a), Some(b)) if (a - b).abs() >= 0.05 => Some(b),
        (_, Some(b)) => {
            // Tag present; if identical to native, may be bogus on some bodies.
            if let Some(a) = fl {
                if (a - b).abs() < 0.05 {
                    if let Some(s) = scale {
                        sanitize_focal_mm_35eq(a * s)
                    } else if a < 24.0 {
                        // Short identical tags often mean APS-C wrote native twice.
                        sanitize_focal_mm_35eq(a * 1.5)
                    } else {
                        sanitize_focal_mm_35eq(a)
                    }
                } else {
                    Some(b)
                }
            } else {
                Some(b)
            }
        }
        (Some(a), None) => {
            if let Some(s) = scale {
                sanitize_focal_mm_35eq(a * s)
            } else {
                sanitize_focal_mm_35eq(a)
            }
        }
        _ => None,
    };

    candidate.and_then(sanitize_focal_mm_35eq).unwrap_or(DEFAULT_FOCAL_MM_35EQ)
}

fn parse_focal_mm_native(exif: &HashMap<String, String>) -> f64 {
    exif.get("FocalLength")
        .and_then(|s| parse_mm(s))
        .filter(|v| *v > 1.0)
        .unwrap_or_else(|| parse_focal_mm_35eq(exif))
}

/// Always-on lens profile for panorama (ignores editor lensDistortionEnabled).
/// FOV comes from EXIF 35mm-eq only; lensfun is used for distortion coeffs, not cropfactor FOV.
pub fn pose_from_exif_with_lens(
    exif: &HashMap<String, String>,
    width: u32,
    height: u32,
    lens_db: Option<&LensDatabase>,
) -> CameraPose {
    let focal_mm_35 = parse_focal_mm_35eq(exif);
    let mut lens = LensModel::default();

    if let Some(db) = lens_db {
        let maker = exif
            .get("LensMake")
            .or_else(|| exif.get("Make"))
            .map(|s| s.as_str())
            .unwrap_or("");
        let lens_model = exif.get("LensModel").map(|s| s.as_str()).unwrap_or("");
        let camera_model = exif.get("Model").map(|s| s.as_str()).unwrap_or("");
        let native_fl = parse_focal_mm_native(exif) as f32;

        if let Some((lmaker, lmodel)) = find_best_lens_match(db, maker, lens_model, camera_model) {
            if let Some(params) = resolve_lens_params(db, &lmaker, &lmodel, native_fl, None, None) {
                lens = LensModel::from_params(&params);
                log::info!(
                    "Panorama lens profile: {} {} (model={}, k=[{:.4},{:.4},{:.4}]) focal35={:.1} native={:.1}",
                    lmaker,
                    lmodel,
                    lens.model,
                    lens.k1,
                    lens.k2,
                    lens.k3,
                    focal_mm_35,
                    native_fl
                );
            }
        }
    }

    let fov = fov_rad_from_focal_mm_35eq(focal_mm_35);
    let focal_px = focal_px_from_fov(fov, width);
    log::info!(
        "Panorama FOV bootstrap: FocalLength={:?} FocalLengthIn35mmFilm={:?} ScaleFactor35efl={:?} => focal35={:.2}mm focal_px={:.1} ({}x{})",
        exif.get("FocalLength"),
        exif.get("FocalLengthIn35mmFilm"),
        exif.get("ScaleFactor35efl").or_else(|| exif.get("ScaleFactor35Efl")),
        focal_mm_35,
        focal_px,
        width,
        height
    );
    CameraPose {
        rotation_vector: [0.0, 0.0, 0.0],
        focal_px,
        width,
        height,
        lens,
    }
}

pub fn pose_from_exif(exif: &HashMap<String, String>, width: u32, height: u32) -> CameraPose {
    pose_from_exif_with_lens(exif, width, height, None)
}

pub fn rotation_from_bearings_kabsch(
    bearings_a: &[Vector3<f64>],
    bearings_b: &[Vector3<f64>],
) -> Option<Rotation3<f64>> {
    if bearings_a.len() < 2 || bearings_a.len() != bearings_b.len() {
        return None;
    }
    let mut h = Matrix3::zeros();
    for (a, b) in bearings_a.iter().zip(bearings_b.iter()) {
        h += a * b.transpose();
    }
    let svd = SVD::new(h, true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut r = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v = v_t.transpose();
        v.set_column(2, &(-v.column(2)));
        r = v * u.transpose();
    }
    Some(Rotation3::from_matrix_unchecked(r))
}

pub fn estimate_relative_rotation_ransac(
    pts_a: &[(f64, f64)],
    pts_b: &[(f64, f64)],
    pose_a: &CameraPose,
    pose_b: &CameraPose,
) -> Option<(Rotation3<f64>, Vec<usize>)> {
    if pts_a.len() < 3 || pts_a.len() != pts_b.len() {
        return None;
    }

    let bearings_a: Vec<Vector3<f64>> = pts_a
        .iter()
        .map(|&(x, y)| pose_a.pixel_to_bearing(x, y))
        .collect();
    let bearings_b: Vec<Vector3<f64>> = pts_b
        .iter()
        .map(|&(x, y)| pose_b.pixel_to_bearing(x, y))
        .collect();

    let mut rng = rand::rng();
    let n = bearings_a.len();
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_r: Option<Rotation3<f64>> = None;

    for _ in 0..RANSAC_ITERS {
        let mut sample = [0usize; 3];
        sample[0] = rng.random_range(0..n);
        loop {
            sample[1] = rng.random_range(0..n);
            if sample[1] != sample[0] {
                break;
            }
        }
        loop {
            sample[2] = rng.random_range(0..n);
            if sample[2] != sample[0] && sample[2] != sample[1] {
                break;
            }
        }

        let ba = [
            bearings_a[sample[0]],
            bearings_a[sample[1]],
            bearings_a[sample[2]],
        ];
        let bb = [
            bearings_b[sample[0]],
            bearings_b[sample[1]],
            bearings_b[sample[2]],
        ];
        let Some(r) = rotation_from_bearings_kabsch(&ba, &bb) else {
            continue;
        };

        let mut inliers = Vec::new();
        for i in 0..n {
            let mapped = r * bearings_a[i];
            if mapped.angle(&bearings_b[i]) < RANSAC_ANGLE_THRESH_RAD {
                inliers.push(i);
            }
        }
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_r = Some(r);
        }
    }

    if best_inliers.len() < MIN_ROTATION_INLIERS {
        return None;
    }

    let ba: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| bearings_a[i]).collect();
    let bb: Vec<Vector3<f64>> = best_inliers.iter().map(|&i| bearings_b[i]).collect();
    let r = rotation_from_bearings_kabsch(&ba, &bb).or(best_r)?;

    let mut refined = Vec::new();
    for i in 0..n {
        let mapped = r * bearings_a[i];
        if mapped.angle(&bearings_b[i]) < RANSAC_ANGLE_THRESH_RAD {
            refined.push(i);
        }
    }
    if refined.len() < MIN_ROTATION_INLIERS {
        return None;
    }
    let ba: Vec<Vector3<f64>> = refined.iter().map(|&i| bearings_a[i]).collect();
    let bb: Vec<Vector3<f64>> = refined.iter().map(|&i| bearings_b[i]).collect();
    let r = rotation_from_bearings_kabsch(&ba, &bb)?;
    Some((r, refined))
}

pub fn yaw_pitch_from_rotation(r: &Rotation3<f64>) -> (f64, f64) {
    let forward = r * Vector3::new(0.0, 0.0, 1.0);
    let yaw = forward.x.atan2(forward.z);
    let pitch = forward.y.asin().clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    (yaw, pitch)
}

pub fn level_horizon(poses: &mut [CameraPose], kept: &[usize]) {
    if kept.is_empty() {
        return;
    }
    let mut avg_up = Vector3::zeros();
    for &i in kept {
        avg_up += poses[i].rotation() * Vector3::new(0.0, 1.0, 0.0);
    }
    avg_up /= kept.len() as f64;
    if avg_up.norm() < 1e-8 {
        return;
    }
    avg_up.normalize_mut();

    let world_up = Vector3::new(0.0, 1.0, 0.0);
    let axis = avg_up.cross(&world_up);
    let axis_norm = axis.norm();
    if axis_norm < 1e-8 {
        return;
    }
    let angle = avg_up.angle(&world_up);
    let correct =
        Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle);
    for &i in kept {
        let leveled = correct * poses[i].rotation();
        poses[i].set_rotation(leveled);
    }
}

/// Roll angle of a camera relative to world up, preserving look direction.
pub fn camera_roll_rad(r: &Rotation3<f64>) -> f64 {
    let forward = r * Vector3::new(0.0, 0.0, 1.0);
    let up = r * Vector3::new(0.0, 1.0, 0.0);
    let world_up = Vector3::new(0.0, 1.0, 0.0);
    let right = world_up.cross(&forward);
    if right.norm() < 1e-8 {
        return 0.0;
    }
    let leveled_up = forward.cross(&right.normalize()).normalize();
    let sin_r = leveled_up.cross(&up).dot(&forward);
    let cos_r = leveled_up.dot(&up);
    sin_r.atan2(cos_r)
}

/// Decompose relative rotation into (yaw, pitch, roll) in radians about Y, X, Z.
pub fn relative_yaw_pitch_roll(r: &Rotation3<f64>) -> (f64, f64, f64) {
    let (yaw, pitch) = yaw_pitch_from_rotation(r);
    let roll = camera_roll_rad(r);
    (yaw, pitch, roll)
}

/// Remove per-camera roll while preserving each look-at direction.
/// Global `level_horizon` alone cannot fix a horseshoe caused by differential roll.
pub fn zero_camera_roll(poses: &mut [CameraPose], kept: &[usize]) {
    let world_up = Vector3::new(0.0, 1.0, 0.0);
    for &i in kept {
        let r = poses[i].rotation();
        let forward = (r * Vector3::new(0.0, 0.0, 1.0)).normalize();
        let mut right = world_up.cross(&forward);
        if right.norm() < 1e-6 {
            continue;
        }
        right.normalize_mut();
        let up = forward.cross(&right).normalize();
        let mat = Matrix3::from_columns(&[right, up, forward]);
        if mat.determinant() < 0.0 {
            continue;
        }
        poses[i].set_rotation(Rotation3::from_matrix_unchecked(mat));
    }
}

pub fn rotation_from_yaw_pitch(yaw: f64, pitch: f64) -> Rotation3<f64> {
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    // Look-at: (sin(yaw)*cos(pitch), sin(pitch), cos(yaw)*cos(pitch))
    let forward = Vector3::new(sy * cp, sp, cy * cp).normalize();
    let world_up = Vector3::new(0.0, 1.0, 0.0);
    let mut right = world_up.cross(&forward);
    if right.norm() < 1e-8 {
        right = Vector3::new(1.0, 0.0, 0.0);
    } else {
        right.normalize_mut();
    }
    let up = forward.cross(&right).normalize();
    Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[right, up, forward]))
}

/// True when look-directions form one pitch cluster (1×N row), not a multi-row grid.
pub fn is_likely_single_row(poses: &[CameraPose], kept: &[usize]) -> bool {
    if kept.len() < 2 {
        return true;
    }
    let mut pitches: Vec<f64> = kept
        .iter()
        .map(|&i| yaw_pitch_from_rotation(&poses[i].rotation()).1)
        .collect();
    pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pitch_span = pitches.last().unwrap() - pitches.first().unwrap();
    if pitch_span.to_degrees() > 14.0 {
        return false;
    }
    let mut clusters = 1usize;
    for w in pitches.windows(2) {
        if (w[1] - w[0]).to_degrees() > 6.0 {
            clusters += 1;
        }
    }
    clusters <= 1
}

/// Force a single-row motion model: shared pitch, zero roll, keep per-camera yaw.
/// No-ops (aside from roll removal) when poses look like a multi-row set.
pub fn enforce_single_row(poses: &mut [CameraPose], kept: &[usize]) {
    if kept.is_empty() {
        return;
    }
    if !is_likely_single_row(poses, kept) {
        log::info!("Panorama: multi-row detected — keeping per-camera pitch");
        zero_camera_roll(poses, kept);
        return;
    }
    log::info!("Panorama: single-row detected — sharing pitch, zeroing roll");
    let mut pitches: Vec<f64> = kept
        .iter()
        .map(|&i| yaw_pitch_from_rotation(&poses[i].rotation()).1)
        .collect();
    pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_pitch = pitches[pitches.len() / 2];
    for &i in kept {
        let (yaw, _) = yaw_pitch_from_rotation(&poses[i].rotation());
        poses[i].set_rotation(rotation_from_yaw_pitch(yaw, median_pitch));
    }
}

/// Keep correspondences whose flow agrees with a dominant translation axis.
/// Works for horizontal rows and vertical multi-row links.
pub fn filter_panorama_flow_matches(
    pts_a: &[(f64, f64)],
    pts_b: &[(f64, f64)],
) -> Vec<usize> {
    if pts_a.len() != pts_b.len() || pts_a.len() < 4 {
        return (0..pts_a.len()).collect();
    }
    let mut dxs: Vec<f64> = Vec::with_capacity(pts_a.len());
    let mut dys: Vec<f64> = Vec::with_capacity(pts_a.len());
    for (a, b) in pts_a.iter().zip(pts_b.iter()) {
        dxs.push(b.0 - a.0);
        dys.push(b.1 - a.1);
    }
    let mut dx_sorted = dxs.clone();
    let mut dy_sorted = dys.clone();
    dx_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dy_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_dx = dx_sorted[dx_sorted.len() / 2];
    let median_dy = dy_sorted[dy_sorted.len() / 2];

    let horizontal = median_dx.abs() >= median_dy.abs();
    let primary = if horizontal { median_dx } else { median_dy };
    if primary.abs() < 1.0 {
        return (0..pts_a.len()).collect();
    }
    let sign = primary.signum();
    let max_ortho = (primary.abs() * 0.55).clamp(25.0, 180.0);
    let min_primary = (primary.abs() * 0.25).max(8.0);

    let mut kept = Vec::new();
    for (i, (a, b)) in pts_a.iter().zip(pts_b.iter()).enumerate() {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let (along, ortho) = if horizontal { (dx, dy) } else { (dy, dx) };
        if along * sign < min_primary {
            continue;
        }
        if ortho.abs() > max_ortho {
            continue;
        }
        kept.push(i);
    }
    if kept.len() >= 6 {
        kept
    } else {
        (0..pts_a.len()).collect()
    }
}

/// OpenCV `detail::waveCorrect` (Brown & Lowe via OpenCV motion_estimators.cpp).
/// Builds an orthonormal frame from the X-axis moment matrix and reorients all cameras.
pub fn wave_correct(poses: &mut [CameraPose], kept: &[usize], horizontal: bool) {
    if kept.len() < 2 {
        return;
    }
    // Moment of camera X axes (OpenCV always uses col(0), for both HORIZ and VERT).
    let mut moment = Matrix3::zeros();
    for &i in kept {
        let col = poses[i].rotation().matrix().column(0).clone_owned();
        moment += &col * col.transpose();
    }
    let svd = SVD::new(moment, true, true);
    let Some(u) = svd.u else {
        return;
    };
    // OpenCV eigen() returns descending eigenvalues; SVD U columns match that order.
    // HORIZ → smallest eigenvector; VERT → largest.
    let mut rg1 = if horizontal {
        u.column(2).clone_owned()
    } else {
        u.column(0).clone_owned()
    };
    if rg1.norm() < 1e-12 {
        return;
    }
    rg1.normalize_mut();

    let mut img_k = Vector3::zeros();
    for &i in kept {
        img_k += poses[i].rotation().matrix().column(2);
    }
    let mut rg0 = rg1.cross(&img_k);
    let rg0_norm = rg0.norm();
    if rg0_norm < 1e-12 {
        return;
    }
    rg0 /= rg0_norm;

    // Orientation consistency (OpenCV confidence flip).
    if horizontal {
        let mut conf = 0.0;
        for &i in kept {
            conf += rg0.dot(&poses[i].rotation().matrix().column(0));
        }
        if conf < 0.0 {
            rg0 = -rg0;
            rg1 = -rg1;
        }
    } else {
        let mut conf = 0.0;
        for &i in kept {
            conf -= rg1.dot(&poses[i].rotation().matrix().column(0));
        }
        if conf < 0.0 {
            rg0 = -rg0;
            rg1 = -rg1;
        }
    }
    let rg2 = rg0.cross(&rg1);
    let r_correct = Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
        rg0.transpose(),
        rg1.transpose(),
        rg2.transpose(),
    ]));
    for &i in kept {
        let leveled = r_correct * poses[i].rotation();
        poses[i].set_rotation(leveled);
    }
}

