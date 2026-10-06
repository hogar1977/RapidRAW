mod align;
mod finish;
mod jpeg;

use crate::sets;
use rapidraw_lib::panorama_utils::v86::trace::{self, StitchLog};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub fn run(wanted: &[String]) {
    let root = PathBuf::from("/home/dalibor/Projects/RapidRAW");
    let out_root = root.join("lean_pano_proto/out/v10");
    let tmp = root.join("lean_pano_proto/tmp/v10");
    let db = root.join("src-tauri/lensfun_db");
    let _ = std::fs::create_dir_all(&out_root);
    let _ = std::fs::create_dir_all(&tmp);
    let sets = sets::select(&root.join("temporary_pano"), wanted);
    if sets.is_empty() {
        eprintln!("no panorama sets found");
        std::process::exit(1);
    }
    let mut failed = false;
    for (stem, paths) in &sets {
        let dir = out_root.join(stem);
        let _ = std::fs::create_dir_all(&dir);
        let log_path = dir.join(format!("{stem}_Pano.pano.log"));
        let tiff_path = dir.join(format!("{stem}_Pano.tiff"));
        let preview_path = dir.join(format!("{stem}_preview.jpg"));
        let view_path = dir.join(format!("{stem}_view.jpg"));
        println!("v10 {stem} ({} files)", paths.len());
        match run_set(&log_path, &tiff_path, &preview_path, &view_path, paths, &db, &tmp, &stem) {
            Ok(()) => println!("saved {}", tiff_path.display()),
            Err(error) => {
                eprintln!("{stem} failed: {error}");
                failed = true;
                if stem.contains("0140") && wanted.is_empty() {
                    eprintln!("stopping after 0140 so the later sets are not run on a bad mapping");
                    break;
                }
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}

fn run_set(log_path: &Path, tiff_path: &Path, preview_path: &Path, view_path: &Path, paths: &[String], db: &Path, tmp: &Path, stem: &str) -> Result<(), String> {
    let stop = Arc::new(AtomicBool::new(false));
    let log = StitchLog::create(log_path, stop)?;
    let _guard = trace::install(log);
    trace::line(&format!("merge started files={}", paths.len()));
    let solved = align::solve_paths(paths, db)?;
    trace::line(&format!("lens crop={:.3} focal35={:.1} calib={:.3}", solved.crop_factor, solved.focal35, solved.calib));
    let preview = align::preview_at(solved, None)?;
    let cropped = crop_preview(&preview.result.rgb, preview.result.width, preview.result.height, preview.result.crop_x, preview.result.crop_y, preview.result.crop_w, preview.result.crop_h);
    finish::write_view(&cropped.0, cropped.1, cropped.2, preview_path)?;
    trace::line(&format!("preview ready {}", preview_path.display()));
    trace::line("save started");
    finish::render(paths, &preview, tmp, tiff_path, view_path, stem)?;
    trace::line(&format!("saved {}", tiff_path.display()));
    Ok(())
}

fn crop_preview(rgb: &[f32], w: u32, h: u32, x: f64, y: f64, cw: f64, ch: f64) -> (Vec<f32>, u32, u32) {
    let x0 = (x * w as f64).round().clamp(0.0, w as f64) as u32;
    let y0 = (y * h as f64).round().clamp(0.0, h as f64) as u32;
    let x1 = ((x + cw) * w as f64).round().clamp(x0 as f64 + 1.0, w as f64) as u32;
    let y1 = ((y + ch) * h as f64).round().clamp(y0 as f64 + 1.0, h as f64) as u32;
    let out_w = x1 - x0;
    let out_h = y1 - y0;
    let mut out = vec![0f32; out_w as usize * out_h as usize * 3];
    for row in 0..out_h {
        let s = (((y0 + row) * w + x0) * 3) as usize;
        let d = (row * out_w * 3) as usize;
        let n = out_w as usize * 3;
        out[d..d + n].copy_from_slice(&rgb[s..s + n]);
    }
    (out, out_w, out_h)
}
