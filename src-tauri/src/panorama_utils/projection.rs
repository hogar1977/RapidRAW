use crate::panorama_utils::camera::{yaw_pitch_from_rotation, CameraPose};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Projection {
    Spherical,
    Cylindrical,
    Perspective,
}

impl Projection {
    pub fn as_str(self) -> &'static str {
        match self {
            Projection::Spherical => "spherical",
            Projection::Cylindrical => "cylindrical",
            Projection::Perspective => "perspective",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "spherical" => Some(Projection::Spherical),
            "cylindrical" => Some(Projection::Cylindrical),
            "perspective" => Some(Projection::Perspective),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectionCanvas {
    pub projection: Projection,
    pub width: u32,
    pub height: u32,
    pub hfov: f64,
    pub vfov: f64,
    pub yaw0: f64,
    pub pitch0: f64,
}

impl ProjectionCanvas {
    pub fn from_poses(poses: &[CameraPose], projection: Projection, long_side: u32) -> Self {
        let mut min_yaw = f64::INFINITY;
        let mut max_yaw = f64::NEG_INFINITY;
        let mut min_pitch = f64::INFINITY;
        let mut max_pitch = f64::NEG_INFINITY;

        for pose in poses {
            let (yaw, pitch) = yaw_pitch_from_rotation(&pose.rotation());
            let half_h = (pose.width as f64 * 0.5 / pose.focal_px).atan();
            let half_v = (pose.height as f64 * 0.5 / pose.focal_px).atan();
            min_yaw = min_yaw.min(yaw - half_h);
            max_yaw = max_yaw.max(yaw + half_h);
            min_pitch = min_pitch.min(pitch - half_v);
            max_pitch = max_pitch.max(pitch + half_v);
        }

        if !min_yaw.is_finite() {
            min_yaw = -0.5;
            max_yaw = 0.5;
            min_pitch = -0.3;
            max_pitch = 0.3;
        }

        let pad = 0.02;
        let hfov = (max_yaw - min_yaw + pad).max(0.1);
        let vfov = (max_pitch - min_pitch + pad).max(0.1);
        let yaw0 = (min_yaw + max_yaw) * 0.5;
        let pitch0 = (min_pitch + max_pitch) * 0.5;

        let aspect = hfov / vfov;
        let (width, height) = if aspect >= 1.0 {
            let w = long_side;
            let h = ((long_side as f64) / aspect).round().max(1.0) as u32;
            (w, h)
        } else {
            let h = long_side;
            let w = ((long_side as f64) * aspect).round().max(1.0) as u32;
            (w, h)
        };

        ProjectionCanvas {
            projection,
            width,
            height,
            hfov,
            vfov,
            yaw0,
            pitch0,
        }
    }

    pub fn pixel_to_ray(&self, x: f64, y: f64) -> Vector3<f64> {
        let u = (x + 0.5) / self.width as f64 - 0.5;
        let v = (y + 0.5) / self.height as f64 - 0.5;
        match self.projection {
            Projection::Spherical => {
                let yaw = self.yaw0 + u * self.hfov;
                let pitch = self.pitch0 - v * self.vfov;
                Vector3::new(
                    yaw.sin() * pitch.cos(),
                    pitch.sin(),
                    yaw.cos() * pitch.cos(),
                )
            }
            Projection::Cylindrical => {
                let yaw = self.yaw0 + u * self.hfov;
                let y_cyl = -v * (self.vfov.tan());
                Vector3::new(yaw.sin(), y_cyl, yaw.cos()).normalize()
            }
            Projection::Perspective => {
                let x_p = u * 2.0 * (self.hfov * 0.5).tan();
                let y_p = -v * 2.0 * (self.vfov * 0.5).tan();
                let local = Vector3::new(x_p, y_p, 1.0).normalize();
                let cy = self.yaw0.cos();
                let sy = self.yaw0.sin();
                let cp = self.pitch0.cos();
                let sp = self.pitch0.sin();
                let after_pitch =
                    Vector3::new(local.x, local.y * cp - local.z * sp, local.y * sp + local.z * cp);
                Vector3::new(
                    after_pitch.x * cy + after_pitch.z * sy,
                    after_pitch.y,
                    -after_pitch.x * sy + after_pitch.z * cy,
                )
            }
        }
    }

    pub fn ray_to_pixel(&self, ray: Vector3<f64>) -> Option<(f64, f64)> {
        let n = ray.normalize();
        match self.projection {
            Projection::Spherical => {
                let yaw = n.x.atan2(n.z);
                let pitch = n.y.asin();
                let u = (yaw - self.yaw0) / self.hfov + 0.5;
                let v = (self.pitch0 - pitch) / self.vfov + 0.5;
                Some((u * self.width as f64 - 0.5, v * self.height as f64 - 0.5))
            }
            Projection::Cylindrical => {
                let yaw = n.x.atan2(n.z);
                let y_cyl = n.y / (n.x.hypot(n.z)).max(1e-9);
                let u = (yaw - self.yaw0) / self.hfov + 0.5;
                let v = (-y_cyl) / self.vfov.tan().max(1e-6) + 0.5;
                Some((u * self.width as f64 - 0.5, v * self.height as f64 - 0.5))
            }
            Projection::Perspective => {
                let cy = self.yaw0.cos();
                let sy = self.yaw0.sin();
                let cp = self.pitch0.cos();
                let sp = self.pitch0.sin();
                let after_yaw = Vector3::new(n.x * cy - n.z * sy, n.y, n.x * sy + n.z * cy);
                let local = Vector3::new(
                    after_yaw.x,
                    after_yaw.y * cp + after_yaw.z * sp,
                    -after_yaw.y * sp + after_yaw.z * cp,
                );
                if local.z <= 1e-8 {
                    return None;
                }
                let x_p = local.x / local.z;
                let y_p = local.y / local.z;
                let u = x_p / (2.0 * (self.hfov * 0.5).tan()) + 0.5;
                let v = -y_p / (2.0 * (self.vfov * 0.5).tan()) + 0.5;
                Some((u * self.width as f64 - 0.5, v * self.height as f64 - 0.5))
            }
        }
    }
}

pub fn recommend_projection(poses: &[CameraPose]) -> Projection {
    if poses.is_empty() {
        return Projection::Cylindrical;
    }
    let mut min_yaw = f64::INFINITY;
    let mut max_yaw = f64::NEG_INFINITY;
    let mut min_pitch = f64::INFINITY;
    let mut max_pitch = f64::NEG_INFINITY;
    for pose in poses {
        let (yaw, pitch) = yaw_pitch_from_rotation(&pose.rotation());
        min_yaw = min_yaw.min(yaw);
        max_yaw = max_yaw.max(yaw);
        min_pitch = min_pitch.min(pitch);
        max_pitch = max_pitch.max(pitch);
    }
    let yaw_span = (max_yaw - min_yaw).to_degrees();
    let pitch_span = (max_pitch - min_pitch).to_degrees();
    // Single-row landscape pans prefer cylindrical once beyond a modest FOV;
    // perspective compresses the sides and fights seam continuity on 1×N sets.
    if pitch_span >= 22.0 || yaw_span >= 160.0 {
        Projection::Spherical
    } else if pitch_span < 14.0 && yaw_span <= 28.0 {
        Projection::Perspective
    } else if pitch_span < 14.0 {
        Projection::Cylindrical
    } else if yaw_span <= 65.0 {
        Projection::Perspective
    } else {
        Projection::Cylindrical
    }
}
