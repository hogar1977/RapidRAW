//! Temporary verbose panorama diagnostics.
//!
//! ## How to remove later
//! 1. Set [`ENABLED`] to `false`, **or**
//! 2. Delete this file, remove `pub mod debug_log;` from `mod.rs`, and delete every
//!    `debug_log::` / `pano_dbg` call site in `panorama_stitching.rs` (search `PANO_DEBUG`).
//!
//! One log file is written per stitch activation next to the first source image:
//! `RapidRAW_pano_YYYYMMDD_HHMMSS.log`

use image::{DynamicImage, GrayImage, Rgb32FImage};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Master switch — flip to `false` to silence without deleting call sites.
pub const ENABLED: bool = true;

/// Process-wide current log handle (avoids thread-local RefCell re-entrancy panics).
static CURRENT: OnceLock<Mutex<Option<Arc<PanoDebugLog>>>> = OnceLock::new();

fn current_slot() -> &'static Mutex<Option<Arc<PanoDebugLog>>> {
    CURRENT.get_or_init(|| Mutex::new(None))
}

pub struct PanoDebugLog {
    pub path: PathBuf,
    started: Instant,
    file: Mutex<File>,
}

impl PanoDebugLog {
    /// Create a new per-run log beside the first source file (fallback: cwd).
    pub fn start(source_paths: &[String]) -> Option<Arc<Self>> {
        if !ENABLED {
            return None;
        }
        let ts = timestamp_stamp();
        let dir = source_paths
            .first()
            .and_then(|p| Path::new(p).parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        let path = dir.join(format!("RapidRAW_pano_{}.log", ts));
        let mut file = match File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                log::warn!("PANO_DEBUG: failed to create {}: {}", path.display(), e);
                return None;
            }
        };
        let _ = writeln!(
            file,
            "RapidRAW panorama debug log\n\
             created={}\n\
             log_path={}\n\
             NOTE: Temporary diagnostic aid. Disable via panorama_utils/debug_log.rs ENABLED=false.\n\
             ============================================================",
            chrono_like_now(),
            path.display()
        );
        let handle = Arc::new(Self {
            path: path.clone(),
            started: Instant::now(),
            file: Mutex::new(file),
        });
        if let Ok(mut slot) = current_slot().lock() {
            *slot = Some(handle.clone());
        }
        log::info!("PANO_DEBUG: writing {}", path.display());
        Some(handle)
    }

    pub fn path_str(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }

    pub fn write(&self, msg: &str) {
        let elapsed = self.started.elapsed().as_secs_f32();
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "[{:8.3}s] {}", elapsed, msg);
            let _ = f.flush();
        }
    }

    pub fn section(&self, title: &str) {
        self.write(&format!("---- {} ----", title));
    }

    pub fn finish(self: &Arc<Self>, status: &str) {
        self.write(&format!(
            "DONE status={} total={:.3}s",
            status,
            self.started.elapsed().as_secs_f32()
        ));
        self.write(&format!("log_file={}", self.path.display()));
        if let Ok(mut slot) = current_slot().lock() {
            let clear = slot
                .as_ref()
                .map(|cur| Arc::ptr_eq(cur, self))
                .unwrap_or(false);
            if clear {
                *slot = None;
            }
        }
    }
}

pub fn clear_current() {
    if let Ok(mut slot) = current_slot().lock() {
        *slot = None;
    }
}

pub fn write(msg: &str) {
    if !ENABLED {
        return;
    }
    // Clone Arc out of the mutex before writing so we never hold CURRENT while
    // touching the file lock (and never nest RefCell/Mutex borrows).
    let handle = current_slot().lock().ok().and_then(|s| s.clone());
    if let Some(log) = handle {
        log.write(msg);
    }
}

pub fn section(title: &str) {
    if !ENABLED {
        return;
    }
    let handle = current_slot().lock().ok().and_then(|s| s.clone());
    if let Some(log) = handle {
        log.section(title);
    }
}

pub fn log_paths(source_paths: &[String]) {
    section("SOURCE PATHS");
    for (i, p) in source_paths.iter().enumerate() {
        write(&format!("[{}] {}", i, p));
    }
}

pub fn log_rgb32f_stats(label: &str, img: &Rgb32FImage) {
    let (w, h) = img.dimensions();
    let n = (w as usize) * (h as usize);
    if n == 0 {
        write(&format!("{}: empty {}x{}", label, w, h));
        return;
    }
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan = 0usize;
    let mut inf = 0usize;
    let mut nonzero = 0usize;
    for p in img.pixels() {
        for c in p.0 {
            if c.is_nan() {
                nan += 1;
                continue;
            }
            if c.is_infinite() {
                inf += 1;
                continue;
            }
            min_v = min_v.min(c);
            max_v = max_v.max(c);
            sum += c as f64;
            if c.abs() > 1e-12 {
                nonzero += 1;
            }
        }
    }
    let mean = sum / ((n * 3) as f64);
    write(&format!(
        "{}: {}x{} min={:.6} max={:.6} mean={:.6} nonzero_channels={} nan={} inf={}",
        label, w, h, min_v, max_v, mean, nonzero, nan, inf
    ));
}

pub fn log_gray_stats(label: &str, img: &GrayImage) {
    let (w, h) = img.dimensions();
    let mut min_v = u8::MAX;
    let mut max_v = u8::MIN;
    let mut sum = 0u64;
    let mut covered = 0usize;
    for p in img.pixels() {
        let v = p[0];
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        sum += v as u64;
        if v > 0 {
            covered += 1;
        }
    }
    let n = (w as usize) * (h as usize).max(1);
    write(&format!(
        "{}: {}x{} min={} max={} mean={:.2} covered={} ({:.1}%)",
        label,
        w,
        h,
        min_v,
        max_v,
        sum as f64 / n as f64,
        covered,
        100.0 * covered as f64 / n as f64
    ));
}

pub fn log_dynamic_stats(label: &str, img: &DynamicImage) {
    match img {
        DynamicImage::ImageRgb32F(rgb) => log_rgb32f_stats(label, rgb),
        other => {
            let rgb = other.to_rgb32f();
            log_rgb32f_stats(label, &rgb);
        }
    }
}

fn timestamp_stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Ok(out) = std::process::Command::new("date")
        .args(["+%Y%m%d_%H%M%S"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    format!("{}", secs)
}

fn chrono_like_now() -> String {
    if let Ok(out) = std::process::Command::new("date")
        .args(["--iso-8601=seconds"])
        .output()
    {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout).trim().to_string();
        }
    }
    format!("unix_s={}", timestamp_stamp())
}
