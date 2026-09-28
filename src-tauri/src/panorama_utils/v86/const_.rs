pub const NFEATURES: usize = 12_000;
pub const NOCTAVE_LAYERS: i32 = 3;
pub const CONTRAST_THRESHOLD: f64 = 0.04;
pub const EDGE_THRESHOLD: f64 = 10.0;
pub const SIFT_SIGMA: f64 = 1.6;
pub const LOWE_RATIO: f32 = 0.75;
pub const MIN_PAIR_POINTS: usize = 6;
pub const RANSAC_ITERS: usize = 500;
pub const RANSAC_ITERS_FOCAL: usize = 200;
pub const MIN_INL: usize = 12;
pub const MIN_SPREAD: usize = 3;
pub const BA_HUBER: f64 = 0.002;
pub const BA_MAX_EVALS: usize = 200;
pub const BA_MAX_PER_PAIR: usize = 600;
pub const LEVEL_XATOL: f64 = 1e-6;
pub const LEVEL_FATOL: f64 = 1e-9;
pub const LEVEL_ROLL_W: f64 = 0.25;
pub const PERSPECTIVE_HFOV_MAX: f64 = 65.0;
pub const PERSPECTIVE_VFOV_MAX: f64 = 50.0;
pub const SPHERICAL_HFOV_MIN: f64 = 160.0;
pub const SPHERICAL_VFOV_MIN: f64 = 75.0;
pub const PERSPECTIVE_Z_MIN_DEG: f64 = 80.0;
pub const CANVAS_PX_CAP: f64 = 500e6;
pub const WARP_PAD: i32 = 8;
pub const WARP_CHUNK: i32 = 256;
pub const GAIN_MIN: f64 = 0.50;
pub const GAIN_MAX: f64 = 2.00;
pub const FIELD_RIDGE_BASE: f64 = 0.05;
pub const GAUGE_WEIGHT: f64 = 1e3;
pub const PHOTO_ITERS: usize = 3;
pub const MAX_BLOCKS_PER_PAIR: usize = 4000;
pub const LOCAL_TILE: i32 = 128;
pub const LOCAL_STRIDE: i32 = 64;
pub const LOCAL_MAX_SHIFT: f64 = 32.0;
pub const SEAM_FINE: i32 = 4;
pub const SEAM_COARSE: i32 = 32;
pub const SEAM_BAND: i32 = 128;
pub const SEAM_DIRECT_MAX: usize = 80_000;
pub const RIM_PX: f64 = 48.0;
pub const QUIET_TEX: f32 = 0.08;
pub const QUIET_D: f32 = 0.25;
pub const WANDER_PULL: f64 = 0.8;
pub const VALLEY_PX: f64 = 36.0;
pub const BLEND_DEN: f32 = 1e-4;
pub const CROP_DS: usize = 4;
pub const DISPLAY_LONG_SIDE: u32 = 1920;
pub const POINT_LONG_SIDE: u32 = 2000;
pub const FF_DIAG_MM: f64 = 43.266615305567875;

pub const FIELD_TERMS: [(i32, i32); 9] = [
    (0, 2),
    (0, 4),
    (0, 6),
    (2, 0),
    (2, 2),
    (2, 4),
    (4, 0),
    (4, 2),
    (6, 0),
];

pub const FOCAL_CANDIDATES: [f64; 11] = [16.0, 20.0, 24.0, 28.0, 35.0, 45.0, 60.0, 85.0, 120.0, 170.0, 240.0];

pub fn focal_px_from_35eq(width: u32, height: u32, focal35: f64) -> f64 {
    focal35 * ((width as f64).hypot(height as f64)) / FF_DIAG_MM
}

pub fn ransac_px(w: u32, h: u32) -> f64 {
    2.0 * (w.max(h) as f64) / 3000.0
}

pub fn gray_weights() -> (f32, f32, f32) {
    (0.299, 0.587, 0.114)
}

pub fn luma_weights() -> (f32, f32, f32) {
    (0.2126, 0.7152, 0.0722)
}
