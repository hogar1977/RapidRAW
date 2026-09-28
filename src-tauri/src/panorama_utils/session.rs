use crate::panorama_utils::v86::geom::Projection;
use crate::panorama_utils::v86::lens::LensModel;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroppedImage {
    pub filename: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedCrop {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Default for NormalizedCrop {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0, width: 1.0, height: 1.0 }
    }
}

pub struct WorkingFrame {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<f32>,
}

pub struct PanoramaSession {
    pub source_paths: Vec<String>,
    pub kept_indices: Vec<usize>,
    pub dropped: Vec<DroppedImage>,
    pub preview_png_base64: String,
    pub overlay_png_base64: String,
    pub winner_map_png_base64: String,
    pub filenames: Vec<String>,
    pub recommended_projection: Projection,
    pub selected_projection: Projection,
    pub crop: NormalizedCrop,
    pub preview_width: u32,
    pub preview_height: u32,
    pub temp_dir: Option<PathBuf>,
    pub composite: Vec<f32>,
    pub composite_width: u32,
    pub composite_height: u32,
    pub frames: Vec<WorkingFrame>,
    pub rotations: Vec<Matrix3<f64>>,
    pub focal_px: f64,
    pub focal35: f64,
    pub lens: LensModel,
    pub gains: Vec<f64>,
    pub coef: Vec<f64>,
    pub pedestal: f64,
    pub half: bool,
    pub log_origin: std::time::Instant,
}

impl PanoramaSession {
    pub fn clear_temps(&mut self) {
        if let Some(dir) = self.temp_dir.take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

pub fn drop_session(session: &mut Option<PanoramaSession>) {
    if let Some(mut s) = session.take() {
        s.clear_temps();
    }
}
