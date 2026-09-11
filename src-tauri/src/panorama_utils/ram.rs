#[derive(Debug, Clone)]
pub struct RamGuardStatus {
    pub can_proceed: bool,
    pub estimated_bytes: u64,
    pub available_bytes: u64,
    pub message: Option<String>,
}

pub fn estimate_panorama_bytes(
    width: u32,
    height: u32,
    num_images: usize,
    canvas_width: u32,
    canvas_height: u32,
) -> u64 {
    // Peak during save: all source frames in memory (f32 RGB) + one canvas + mask/winners.
    let frame = (width as u64).saturating_mul(height as u64).saturating_mul(12);
    let sources = frame.saturating_mul(num_images as u64);
    let canvas = (canvas_width as u64)
        .saturating_mul(canvas_height as u64)
        .saturating_mul(12); // output RGB f32
    let aux = (canvas_width as u64)
        .saturating_mul(canvas_height as u64)
        .saturating_mul(3); // mask + winners + scratch
    sources.saturating_add(canvas).saturating_add(aux)
}

pub fn max_safe_bytes(budget_bytes: u64) -> u64 {
    #[cfg(target_os = "android")]
    let ratio = 0.50_f64;
    #[cfg(not(target_os = "android"))]
    let ratio = 0.85_f64;
    (budget_bytes as f64 * ratio) as u64
}

/// Memory budget for the stitch: prefer MemAvailable, but never treat the machine as
/// having less than ~40% of total RAM — preview buffers already sit in the process and
/// would otherwise make a 32 GB box look like an 8 GB one.
pub fn stitch_memory_budget_bytes() -> (u64, u64) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available = sys.available_memory();
    let total = sys.total_memory();
    let floor = ((total as f64) * 0.40) as u64;
    let budget = available.max(floor);
    (budget, total)
}

pub fn available_memory_bytes() -> u64 {
    stitch_memory_budget_bytes().0
}

pub fn check_ram(
    width: u32,
    height: u32,
    num_images: usize,
    canvas_width: u32,
    canvas_height: u32,
) -> RamGuardStatus {
    let estimated = estimate_panorama_bytes(width, height, num_images, canvas_width, canvas_height);
    let (budget, total) = stitch_memory_budget_bytes();
    let limit = max_safe_bytes(budget);
    if estimated > limit {
        RamGuardStatus {
            can_proceed: false,
            estimated_bytes: estimated,
            available_bytes: budget,
            message: Some(format!(
                "Required memory (~{:.1} GB) exceeds the safe limit (~{:.1} GB; {:.1} GB free of {:.1} GB total). Select fewer frames or a smaller output.",
                estimated as f64 / (1024.0 * 1024.0 * 1024.0),
                limit as f64 / (1024.0 * 1024.0 * 1024.0),
                budget as f64 / (1024.0 * 1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            )),
        }
    } else {
        RamGuardStatus {
            can_proceed: true,
            estimated_bytes: estimated,
            available_bytes: budget,
            message: None,
        }
    }
}

/// Full-res canvas long side: match source vertical resolution, not `width * N` (which
/// blows up to 30k² when FOV is mis-estimated).
pub fn save_canvas_long_side(
    poses: &[crate::panorama_utils::camera::CameraPose],
    projection: crate::panorama_utils::projection::Projection,
) -> u32 {
    use crate::panorama_utils::projection::ProjectionCanvas;
    if poses.is_empty() {
        return 4000;
    }
    let target_short = poses
        .iter()
        .map(|p| p.height.min(p.width))
        .max()
        .unwrap_or(4000);
    let probe = ProjectionCanvas::from_poses(poses, projection, target_short.max(512));
    let aspect = (probe.width as f64 / probe.height.max(1) as f64).max(1e-6);
    let long = if aspect >= 1.0 {
        (target_short as f64 * aspect).ceil() as u32
    } else {
        target_short
    };
    long.clamp(2000, 24000)
}
