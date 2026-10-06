use super::finish::{self, Ribbon};
use nalgebra::Matrix3;
use rapidraw_lib::panorama_stitching;
use rapidraw_lib::panorama_utils::v86::lens::LensModel;
use rapidraw_lib::panorama_utils::v86::warp::{self, warp_one_nudged, CanvasGeom, Placed};

// A developed picture. Later file types decode into this same frame.
pub struct LinearFrame {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<f32>,
    pub exposure_ev: Option<f64>,
}

pub trait PixelExec: Send + Sync {
    fn develop_raw(&self, path: &str) -> Result<LinearFrame, String>;
    fn warp(
        &self,
        rgb: &mut [f32],
        width: u32,
        height: u32,
        rot: &Matrix3<f64>,
        focal: f64,
        lens: &LensModel,
        geom: &CanvasGeom,
        nudge: &(dyn Fn(f64, f64) -> (f64, f64) + Sync),
    ) -> Placed;
    fn blend_ribbon(&self, job: &mut Ribbon<'_>);
}

pub struct CpuExec;

pub fn pick_exec() -> Box<dyn PixelExec> {
    Box::new(CpuExec)
}

impl PixelExec for CpuExec {
    fn develop_raw(&self, path: &str) -> Result<LinearFrame, String> {
        let frame = panorama_stitching::develop_panorama_frame(path)?;
        Ok(LinearFrame {
            name: frame.name,
            width: frame.width,
            height: frame.height,
            rgb: frame.rgb,
            exposure_ev: frame.exposure_ev,
        })
    }

    fn warp(
        &self,
        rgb: &mut [f32],
        width: u32,
        height: u32,
        rot: &Matrix3<f64>,
        focal: f64,
        lens: &LensModel,
        geom: &CanvasGeom,
        nudge: &(dyn Fn(f64, f64) -> (f64, f64) + Sync),
    ) -> Placed {
        warp::devignette(rgb, width, height, lens);
        let call = |x, y| nudge(x, y);
        warp_one_nudged(rgb, width, height, rot, focal, lens, geom, &call)
    }

    fn blend_ribbon(&self, job: &mut Ribbon<'_>) {
        finish::run_ribbon(job);
    }
}
