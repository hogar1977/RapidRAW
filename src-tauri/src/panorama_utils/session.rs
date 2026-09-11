use crate::panorama_utils::projection::Projection;
use crate::panorama_utils::camera::CameraPose;
use crate::panorama_utils::local_warp::ImageMesh;
use image::Rgb32FImage;
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
        Self {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PanoramaSession {
    pub source_paths: Vec<String>,
    pub kept_indices: Vec<usize>,
    pub poses: Vec<CameraPose>,
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
    pub preview_images: Vec<Rgb32FImage>,
    pub low_detail_masks: Vec<image::GrayImage>,
    /// Local CP meshes at preview resolution (global image index).
    pub local_meshes: Vec<ImageMesh>,
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
