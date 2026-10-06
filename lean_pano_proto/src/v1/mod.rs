mod align;
mod finish;

use crate::sets;
use rapidraw_lib::panorama_utils::v86::trace::{self, StitchLog};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub fn run(wanted: &[String]) {
    let root = PathBuf::from("/home/dalibor/Projects/RapidRAW");
    let out_root = root.join("lean_pano_proto/out/v1");
    let tmp = root.join("lean_pano_proto/tmp/v1");
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
        let view_path = dir.join(format!("{stem}_view.jpg"));
        println!("v1 {stem} ({} files)", paths.len());
        match run_set(&log_path, &tiff_path, &view_path, paths, &db, &tmp) {
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

fn run_set(log_path: &Path, tiff_path: &Path, view_path: &Path, paths: &[String], db: &Path, tmp: &Path) -> Result<(), String> {
    let stop = Arc::new(AtomicBool::new(false));
    let log = StitchLog::create(log_path, stop)?;
    let _guard = trace::install(log);
    trace::line(&format!("merge started files={}", paths.len()));
    let alignment = align::align_jpegs(paths, db)?;
    trace::line(&format!("lens crop={:.3} focal35={:.1} calib={:.3}", alignment.crop_factor, alignment.focal35, alignment.calib));
    trace::line("preview ready");
    trace::line("save started");
    finish::render(paths, &alignment, tmp, tiff_path, view_path)?;
    trace::line(&format!("saved {}", tiff_path.display()));
    Ok(())
}
