use crate::app_settings::load_settings;
use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::formats::is_raw_file;
use crate::image_processing::{apply_linear_to_srgb, apply_srgb_to_linear};
use crate::lens_correction::{find_best_lens_match, resolve_lens_params, CalibrationElement, LensDatabase};
use crate::panorama_utils::camera::parse_focal_mm_35eq;
use crate::panorama_utils::overlay::generate_overlay;
use crate::panorama_utils::ram::stitch_memory_budget_bytes;
use crate::panorama_utils::session::{drop_session, DroppedImage, NormalizedCrop, PanoramaSession, WorkingFrame};
use crate::panorama_utils::v86::const_::DISPLAY_LONG_SIDE;
use crate::panorama_utils::v86::geom::Projection;
use crate::panorama_utils::v86::lens::{LensKind, LensModel};
use crate::panorama_utils::v86::memory::peak_bytes;
use crate::panorama_utils::v86::photo::Photo;
use crate::panorama_utils::v86::stitch::{self, InputFrame, StitchResult};
use crate::panorama_utils::v86::trace::{self, StitchLog};
use base64::{engine::general_purpose, Engine as _};
use image::{DynamicImage, ImageFormat, Rgb32FImage};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter};

// CYBERTIMON: required by hdr_deghosting; do not remove until HDR is migrated.
pub const BRIEF_DESCRIPTOR_SIZE: usize = 256;
// CYBERTIMON: required by hdr_deghosting; do not remove until HDR is migrated.
pub type Descriptor = [u8; BRIEF_DESCRIPTOR_SIZE / 8];

// CYBERTIMON: required by hdr_deghosting; do not remove until HDR is migrated.
#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    pub x: u32,
    pub y: u32,
}

// CYBERTIMON: required by hdr_deghosting; do not remove until HDR is migrated.
pub struct Feature {
    pub keypoint: KeyPoint,
    pub descriptor: Descriptor,
}

// CYBERTIMON: required by hdr_deghosting; do not remove until HDR is migrated.
#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub index1: usize,
    pub index2: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LensProbe {
    full_bytes: u64,
    half_bytes: u64,
    limit_bytes: u64,
    scale: String,
    size_known: bool,
    lens_maker: String,
    lens_model: String,
    crop_factor: f64,
    focal35: f64,
    native_mm: f64,
    disagreements: Vec<String>,
    calib_crop: f64,
}

#[tauri::command]
pub fn probe_panorama_lenses(paths: Vec<String>, state: tauri::State<'_, AppState>) -> Result<LensProbe, String> {
    if paths.len() < 2 {
        return Err("Please select at least two images to stitch.".into());
    }
    let lens_db = state.lens_db.lock().unwrap().clone();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut size_known = true;
    let mut maker = String::new();
    let mut model = String::new();
    let mut camera_model = String::new();
    let mut focal35 = 50.0;
    let mut natives = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let (real, _) = parse_virtual_path(path);
        let bytes = fs::read(&real).map_err(|e| format!("Could not read {}: {}", real.display(), e))?;
        let exif = crate::exif_processing::read_exif_data_from_bytes(real.to_string_lossy().as_ref(), &bytes);
        if i == 0 {
            focal35 = parse_focal_mm_35eq(&exif);
            maker = exif.get("LensMake").or_else(|| exif.get("Make")).cloned().unwrap_or_default();
            model = exif.get("LensModel").cloned().unwrap_or_default();
            camera_model = exif.get("Model").cloned().unwrap_or_default();
        }
        let native = first_number(exif.get("FocalLength").map(|s| s.as_str()).unwrap_or(""));
        natives.push((file_name(path), native));
        match read_size(real.as_path(), &bytes) {
            Some((w, h)) if i == 0 => {
                width = w;
                height = h;
            }
            Some(_) => {}
            None => size_known = false,
        }
    }
    let (limit, _) = stitch_memory_budget_bytes();
    let (full_bytes, half_bytes, scale) = if size_known && width > 0 && height > 0 {
        let n = paths.len() as u32;
        let full = peak_bytes(n, width, height);
        let half = peak_bytes(n, (width / 2).max(1), (height / 2).max(1));
        let scale = if full <= limit {
            "full"
        } else if half <= limit {
            "half"
        } else {
            "blocked"
        };
        (full, half, scale.to_string())
    } else {
        (0, 0, "blocked".to_string())
    };
    let (lens_maker, lens_model, crop, calib) = lens_summary(lens_db.as_deref(), &maker, &model, &camera_model);
    let crop = if crop > 0.0 { crop } else { 1.0 };
    let native_mm = focal35 / crop;
    let mut disagreements = Vec::new();
    for (name, native) in natives {
        if let Some(mm) = native {
            if (mm - native_mm).abs() > 0.5 {
                disagreements.push(format!("{name}: {mm:.1} mm vs {native_mm:.1} mm"));
            }
        }
    }
    Ok(LensProbe {
        full_bytes,
        half_bytes,
        limit_bytes: limit,
        scale,
        size_known,
        lens_maker,
        lens_model,
        crop_factor: crop,
        focal35,
        native_mm,
        disagreements,
        calib_crop: calib,
    })
}

#[tauri::command]
pub async fn stitch_panorama(
    paths: Vec<String>,
    crop_factor: f64,
    focal35: f64,
    estimate_intrinsics: bool,
    scale: String,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two images to stitch.".to_string());
    }
    let source_paths: Vec<String> = paths.iter().map(|p| parse_virtual_path(p).0.to_string_lossy().into_owned()).collect();
    let session_handle = state.panorama_session.clone();
    let lens_db = state.lens_db.lock().unwrap().clone();
    let stop = arm_stop(&state);
    let log_path = pano_log_path(&source_paths[0]);
    let log = StitchLog::create(&log_path, stop.clone())?;
    log.line(&format!("merge started files={} scale={scale}", source_paths.len()));
    let task = tokio::task::spawn_blocking(move || {
        let _guard = trace::install(log);
        if trace::halted() {
            trace::note_stop();
            return Ok(());
        }
        match run_stitch(&source_paths, crop_factor, focal35, estimate_intrinsics, &scale, &app_handle, lens_db) {
            Ok(session) => {
                let payload = complete_payload(&session);
                {
                    let mut slot = session_handle.lock().unwrap();
                    if trace::halted() {
                        drop_session(&mut slot);
                        trace::note_stop();
                        return Ok(());
                    }
                    *slot = Some(session);
                }
                if trace::halted() {
                    if let Ok(mut slot) = session_handle.try_lock() {
                        drop_session(&mut slot);
                    }
                    trace::note_stop();
                    return Ok(());
                }
                trace::line("preview ready");
                let _ = app_handle.emit("panorama-complete", payload);
                Ok(())
            }
            Err(e) if e == "stopped" || trace::halted() => {
                trace::note_stop();
                Ok(())
            }
            Err(e) => {
                trace::line(&format!("failed: {e}"));
                log::error!("Panorama failed:\n{}", e);
                let _ = app_handle.emit("panorama-error", e.clone());
                Err(e)
            }
        }
    });
    match task.await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(join_err) => Err(format!("Panorama task failed: {}", join_err)),
    }
}

#[tauri::command]
pub async fn reproject_panorama(projection: String, app_handle: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let proj = Projection::parse(&projection).ok_or_else(|| format!("Unknown projection: {}", projection))?;
    let session_handle = state.panorama_session.clone();
    let stop = arm_stop(&state);
    let log_path = {
        let guard = state.panorama_session.lock().unwrap();
        guard.as_ref().and_then(|session| session.source_paths.first().map(|p| pano_log_path(p)))
    };
    let log = match log_path {
        Some(path) => StitchLog::append(&path, stop.clone())?,
        None => return Err("No panorama session. Run stitch first.".into()),
    };
    log.line(&format!("reproject {projection}"));
    let task = tokio::task::spawn_blocking(move || {
        let _guard = trace::install(log);
        if trace::halted() {
            trace::note_stop();
            return Ok(());
        }
        let mut guard = session_handle.lock().unwrap();
        let session = guard.as_mut().ok_or_else(|| "No panorama session. Run stitch first.".to_string())?;
        session.selected_projection = proj;
        let _ = app_handle.emit("panorama-progress", "Reprojecting preview...");
        let frames: Vec<InputFrame> = session.frames.iter().map(|f| InputFrame { name: f.name.clone(), width: f.width, height: f.height, rgb: f.rgb.clone() }).collect();
        let preview = stitch::shrink_frames(&frames, crate::panorama_utils::v86::const_::DISPLAY_LONG_SIDE);
        let scale = preview[0].width as f64 / frames[0].width.max(1) as f64;
        let photo = Photo { gains: session.gains.clone(), coef: session.coef.clone(), pedestal: session.pedestal };
        let rendered = match stitch::rerender(
            &preview,
            &session.rotations,
            &session.kept_indices,
            &session.lens,
            session.focal_px * scale,
            session.focal35,
            &photo,
            proj,
            session.recommended_projection,
            session.half,
            &|msg| { let _ = app_handle.emit("panorama-progress", msg); },
        ) {
            Ok(rendered) => rendered,
            Err(e) if e == "stopped" || trace::halted() => {
                trace::note_stop();
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        if trace::halted() {
            trace::note_stop();
            return Ok(());
        }
        fill_preview(session, &rendered)?;
        let payload = complete_payload(session);
        trace::line("preview ready");
        let _ = app_handle.emit("panorama-complete", payload);
        Ok(())
    });
    match task.await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(join_err) => Err(format!("Reproject task failed: {}", join_err)),
    }
}

#[tauri::command]
pub async fn cancel_panorama(state: tauri::State<'_, AppState>) -> Result<(), String> {
    state.panorama_stop.lock().unwrap().store(true, Ordering::SeqCst);
    if let Ok(mut guard) = state.panorama_session.try_lock() {
        drop_session(&mut guard);
    }
    Ok(())
}

fn arm_stop(state: &AppState) -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let mut slot = state.panorama_stop.lock().unwrap();
    slot.store(true, Ordering::SeqCst);
    *slot = flag.clone();
    flag
}

fn pano_log_path(first: &str) -> PathBuf {
    let (path, _) = parse_virtual_path(first);
    let parent = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("panorama");
    parent.join(format!("{stem}_Pano.pano.log"))
}

#[tauri::command]
pub async fn save_panorama(first_path_str: String, crop: Option<NormalizedCrop>, state: tauri::State<'_, AppState>, app_handle: tauri::AppHandle) -> Result<String, String> {
    let stop = arm_stop(&state);
    let session_slot = state.panorama_session.clone();
    let mut session = {
        let mut guard = state.panorama_session.lock().unwrap();
        guard.take().ok_or_else(|| "No panorama session found to save.".to_string())?
    };
    let log_path = match session.source_paths.first() {
        Some(path) => pano_log_path(path),
        None => {
            if let Ok(mut guard) = session_slot.lock() {
                *guard = Some(session);
            }
            return Err("No panorama session found to save.".to_string());
        }
    };
    let log = match StitchLog::append_since(&log_path, stop, session.log_origin) {
        Ok(log) => log,
        Err(err) => {
            if let Ok(mut guard) = session_slot.lock() {
                *guard = Some(session);
            }
            return Err(err);
        }
    };
    let crop = crop.unwrap_or(session.crop.clone());
    let task = tokio::task::spawn_blocking(move || {
        let _guard = trace::install(log);
        trace::line("save started");
        let result = save_composite(&mut session, &first_path_str, &crop, &app_handle);
        match &result {
            Ok(_) => {}
            Err(e) if e == "stopped" || trace::halted() => trace::note_stop(),
            Err(e) => trace::line(&format!("failed: {e}")),
        }
        let mut session = session;
        session.clear_temps();
        result
    });
    match task.await {
        Ok(Ok(path)) => Ok(path),
        Ok(Err(e)) => Err(e),
        Err(join_err) => Err(format!("Save panorama failed: {}", join_err)),
    }
}

fn run_stitch(
    source_paths: &[String],
    crop_factor: f64,
    focal35: f64,
    estimate: bool,
    scale: &str,
    app: &AppHandle,
    lens_db: Option<std::sync::Arc<LensDatabase>>,
) -> Result<PanoramaSession, String> {
    let (w, h) = header_size(&source_paths[0])?;
    let n = source_paths.len() as u32;
    let (limit, _) = stitch_memory_budget_bytes();
    let full = peak_bytes(n, w, h);
    let half_need = peak_bytes(n, (w / 2).max(1), (h / 2).max(1));
    let half = if scale == "half" || (scale == "full" && full > limit && half_need <= limit) {
        true
    } else if full <= limit && scale != "half" {
        false
    } else if half_need <= limit {
        true
    } else {
        return Err("Not enough memory to stitch these photos.".into());
    };
    if half && scale == "full" {
        let _ = app.emit("panorama-progress", "Using half size so the stitch fits in memory.");
    }
    let settings = load_settings(app.clone()).unwrap_or_default();
    let mut frames = Vec::new();
    let mut gains_ev = Vec::new();
    let mut exif0 = HashMap::new();
    for (i, path) in source_paths.iter().enumerate() {
        trace::gate()?;
        let name = file_name(path);
        let _ = app.emit("panorama-progress", format!("Loading {}/{}: {}", i + 1, source_paths.len(), name));
        let (rgb, fw, fh, exif) = load_linear(path, &settings)?;
        if i == 0 {
            exif0 = exif.clone();
        }
        gains_ev.push(exposure_value(&exif));
        let (rgb, fw, fh) = if half {
            let img = Rgb32FImage::from_raw(fw, fh, rgb).ok_or_else(|| "Could not read pixels.".to_string())?;
            let sw = (fw / 2).max(1);
            let sh = (fh / 2).max(1);
            let small = image::imageops::resize(&img, sw, sh, image::imageops::FilterType::Triangle);
            (small.into_raw(), sw, sh)
        } else {
            (rgb, fw, fh)
        };
        frames.push(InputFrame { name: name.clone(), width: fw, height: fh, rgb });
        trace::line(&format!("loaded {}/{} {name} {fw}x{fh} half={half}", i + 1, source_paths.len()));
    }
    apply_exposure(&mut frames, &gains_ev);
    let (mut lens, calib, crops) = build_lens(lens_db.as_deref(), &exif0, crop_factor, estimate);
    if !estimate {
        lens.k = if crop_factor.abs() < 1e-9 { 1.0 } else { calib / crop_factor };
    }
    let progress = |msg: &str| {
        let _ = app.emit("panorama-progress", msg.to_string());
    };
    trace::line(&format!(
        "lens crop={crop_factor:.3} focal35={focal35:.1} estimate={estimate} calib={calib:.3}"
    ));
    let rendered = stitch::stitch(&frames, lens, focal35, estimate, &crops, calib, half, &progress)?;
    trace::line(&format!(
        "composite {}x{} kept={} dropped={}",
        rendered.width,
        rendered.height,
        rendered.kept.len(),
        rendered.dropped.len()
    ));
    let mut session = PanoramaSession {
        source_paths: source_paths.to_vec(),
        kept_indices: rendered.kept.clone(),
        dropped: rendered.dropped.iter().map(|(filename, reason)| DroppedImage { filename: filename.clone(), reason: reason.clone() }).collect(),
        preview_png_base64: String::new(),
        overlay_png_base64: String::new(),
        winner_map_png_base64: String::new(),
        filenames: rendered.kept.iter().filter_map(|&i| frames.get(i).map(|f| f.name.clone())).collect(),
        recommended_projection: rendered.recommended,
        selected_projection: rendered.used,
        crop: NormalizedCrop { x: rendered.crop_x, y: rendered.crop_y, width: rendered.crop_w, height: rendered.crop_h },
        preview_width: 0,
        preview_height: 0,
        temp_dir: None,
        composite: rendered.rgb.clone(),
        composite_width: rendered.width,
        composite_height: rendered.height,
        frames: frames.into_iter().map(|f| WorkingFrame { name: f.name, width: f.width, height: f.height, rgb: f.rgb }).collect(),
        rotations: rendered.rotations.clone(),
        focal_px: rendered.focal_px,
        focal35: rendered.focal35,
        lens: rendered.lens,
        gains: rendered.photo.gains.clone(),
        coef: rendered.photo.coef.clone(),
        pedestal: rendered.photo.pedestal,
        half,
        log_origin: trace::started().unwrap_or_else(std::time::Instant::now),
    };
    fill_preview(&mut session, &rendered)?;
    Ok(session)
}

fn fill_preview(session: &mut PanoramaSession, rendered: &StitchResult) -> Result<(), String> {
    session.composite = rendered.rgb.clone();
    session.composite_width = rendered.width;
    session.composite_height = rendered.height;
    session.crop = NormalizedCrop { x: rendered.crop_x, y: rendered.crop_y, width: rendered.crop_w, height: rendered.crop_h };
    session.selected_projection = rendered.used;
    session.filenames = rendered.kept.iter().filter_map(|&i| session.frames.get(i).map(|f| f.name.clone())).collect();
    let (dw, dh) = display_size(rendered.width, rendered.height);
    let img = Rgb32FImage::from_raw(rendered.width, rendered.height, rendered.rgb.clone()).ok_or_else(|| "Could not build preview.".to_string())?;
    let small = image::imageops::resize(&img, dw, dh, image::imageops::FilterType::Lanczos3);
    let display = apply_linear_to_srgb(DynamicImage::ImageRgb32F(small));
    session.preview_png_base64 = encode_png(&display.to_rgb8())?;
    let winners = resize_winners(&rendered.winners, rendered.width, rendered.height, dw, dh);
    let overlay = generate_overlay(&winners, dw, dh)?;
    session.overlay_png_base64 = general_purpose::STANDARD.encode(overlay.boundary_png);
    session.winner_map_png_base64 = general_purpose::STANDARD.encode(overlay.winner_map_png);
    session.preview_width = dw;
    session.preview_height = dh;
    Ok(())
}

fn save_composite(session: &mut PanoramaSession, first_path_str: &str, crop: &NormalizedCrop, app: &AppHandle) -> Result<String, String> {
    let mut frames: Vec<InputFrame> = session
        .frames
        .iter_mut()
        .map(|f| InputFrame {
            name: f.name.clone(),
            width: f.width,
            height: f.height,
            rgb: std::mem::take(&mut f.rgb),
        })
        .collect();
    let progress = |msg: &str| {
        let _ = app.emit("panorama-save-progress", serde_json::json!({ "percent": 5, "message": msg }));
    };
    let _ = app.emit("panorama-save-progress", serde_json::json!({ "percent": 5, "message": "Rendering full resolution..." }));
    let rendered = stitch::compose(
        &frames,
        &session.rotations,
        &session.kept_indices,
        session.lens,
        session.focal_px,
        session.focal35,
        session.selected_projection,
        session.half,
        &progress,
    );
    for (dst, src) in session.frames.iter_mut().zip(frames.iter_mut()) {
        if dst.rgb.is_empty() {
            dst.rgb = std::mem::take(&mut src.rgb);
        }
    }
    let rendered = rendered?;
    let _ = app.emit("panorama-save-progress", serde_json::json!({ "percent": 10, "message": "Cropping panorama..." }));
    trace::line(&format!("save crop {}x{}", rendered.width, rendered.height));
    let w = rendered.width;
    let h = rendered.height;
    let x0 = (crop.x * w as f64).round().clamp(0.0, w as f64) as u32;
    let y0 = (crop.y * h as f64).round().clamp(0.0, h as f64) as u32;
    let x1 = ((crop.x + crop.width) * w as f64).round().clamp(x0 as f64 + 1.0, w as f64) as u32;
    let y1 = ((crop.y + crop.height) * h as f64).round().clamp(y0 as f64 + 1.0, h as f64) as u32;
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut cropped = vec![0f32; (cw as usize) * (ch as usize) * 3];
    for y in 0..ch {
        for x in 0..cw {
            let s = (((y0 + y) * w + (x0 + x)) * 3) as usize;
            let d = ((y * cw + x) * 3) as usize;
            cropped[d] = rendered.rgb[s];
            cropped[d + 1] = rendered.rgb[s + 1];
            cropped[d + 2] = rendered.rgb[s + 2];
        }
    }
    let _ = app.emit("panorama-save-progress", serde_json::json!({ "percent": 70, "message": "Encoding image..." }));
    trace::line(&format!("save encode {cw}x{ch}"));
    let img = Rgb32FImage::from_raw(cw, ch, cropped).ok_or_else(|| "Could not crop the panorama.".to_string())?;
    let display = apply_linear_to_srgb(DynamicImage::ImageRgb32F(img));
    let rgb16 = display.to_rgb16();
    let _ = app.emit("panorama-save-progress", serde_json::json!({ "percent": 95, "message": "Writing TIFF..." }));
    let (first_path, _) = parse_virtual_path(first_path_str);
    let parent_dir = first_path.parent().ok_or_else(|| "Could not determine parent directory.".to_string())?;
    let stem = first_path.file_stem().and_then(|s| s.to_str()).unwrap_or("panorama");
    let output_path = parent_dir.join(format!("{stem}_Pano.tiff"));
    rgb16.save_with_format(&output_path, ImageFormat::Tiff).map_err(|e| format!("Failed to save panorama: {}", e))?;
    let (real_path, _) = parse_virtual_path(first_path_str);
    let _ = crate::exif_processing::write_rrexif_sidecar(&real_path.to_string_lossy(), &output_path);
    trace::line(&format!("saved {}", output_path.display()));
    let _ = app.emit("panorama-save-progress", serde_json::json!({ "percent": 100, "message": "Save complete" }));
    Ok(output_path.to_string_lossy().to_string())
}

fn complete_payload(session: &PanoramaSession) -> serde_json::Value {
    serde_json::json!({
        "base64": session.preview_png_base64.clone(),
        "overlayBase64": format!("data:image/png;base64,{}", session.overlay_png_base64),
        "winnerMapBase64": format!("data:image/png;base64,{}", session.winner_map_png_base64),
        "dropped": session.dropped,
        "recommendedProjection": session.recommended_projection.as_str(),
        "selectedProjection": session.selected_projection.as_str(),
        "crop": session.crop,
        "previewWidth": session.preview_width,
        "previewHeight": session.preview_height,
        "filenames": session.filenames,
    })
}

fn load_linear(path: &str, settings: &crate::app_settings::AppSettings) -> Result<(Vec<f32>, u32, u32, HashMap<String, String>), String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {}: {}", file_name(path), e))?;
    let exif = crate::exif_processing::read_exif_data_from_bytes(path, &bytes);
    if let Some((rgb, w, h)) = prepared_linear(path) {
        trace::line(&format!("prepared {} {w}x{h}", file_name(path)));
        return Ok((rgb, w, h, exif));
    }
    let mut dynamic = crate::image_loader::load_base_image_from_bytes(&bytes, path, false, settings, None)
        .map_err(|e| format!("Failed to decode {}: {}", file_name(path), e))?;
    if !is_raw_file(path) {
        dynamic = apply_srgb_to_linear(dynamic);
    }
    let rgb = dynamic.to_rgb32f();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h, exif))
}

// Reads a float picture from PANORAMA_LINEAR_DIR when that folder is set.
fn prepared_linear(path: &str) -> Option<(Vec<f32>, u32, u32)> {
    let dir = std::env::var("PANORAMA_LINEAR_DIR").ok()?;
    let name = file_name(path);
    let stem = Path::new(&name).file_stem()?.to_string_lossy();
    let bytes = fs::read(Path::new(&dir).join(format!("{stem}.f32"))).ok()?;
    if bytes.len() < 8 {
        return None;
    }
    let w = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
    let h = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let count = (w as usize).checked_mul(h as usize)?.checked_mul(3)?;
    if bytes.len() != 8 + count * 4 {
        return None;
    }
    let rgb = bytes[8..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Some((rgb, w, h))
}

fn apply_exposure(frames: &mut [InputFrame], evs: &[Option<f64>]) {
    let mut known: Vec<f64> = evs.iter().filter_map(|v| *v).filter(|v| *v > 0.0).collect();
    if known.is_empty() {
        return;
    }
    known.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ref_ev = known[known.len() / 2];
    for (frame, ev) in frames.iter_mut().zip(evs.iter()) {
        let Some(ev) = ev.filter(|v| *v > 0.0) else { continue };
        let g = (ref_ev / ev) as f32;
        for p in frame.rgb.iter_mut() {
            *p *= g;
        }
    }
}

fn exposure_value(exif: &HashMap<String, String>) -> Option<f64> {
    let shutter = exif.get("ExposureTime").and_then(|s| parse_shutter(s))?;
    let iso = exif.get("PhotographicSensitivity").or_else(|| exif.get("ISOSpeed")).and_then(|s| first_number(s))?;
    let fnum = exif.get("FNumber").or_else(|| exif.get("ApertureValue")).and_then(|s| first_number(s))?;
    if fnum <= 0.0 {
        return None;
    }
    Some(shutter * iso / (fnum * fnum))
}

fn build_lens(db: Option<&LensDatabase>, exif: &HashMap<String, String>, crop_factor: f64, estimate: bool) -> (LensModel, f64, Vec<f64>) {
    let Some(db) = db else {
        return (LensModel::identity(), crop_factor.max(1.0), Vec::new());
    };
    let maker = exif.get("LensMake").or_else(|| exif.get("Make")).map(|s| s.as_str()).unwrap_or("");
    let model = exif.get("LensModel").map(|s| s.as_str()).unwrap_or("");
    let camera = exif.get("Model").map(|s| s.as_str()).unwrap_or("");
    let Some((lmaker, lmodel)) = find_best_lens_match(db, maker, model, camera) else {
        return (LensModel::identity(), 1.0, Vec::new());
    };
    let native = first_number(exif.get("FocalLength").map(|s| s.as_str()).unwrap_or("")).unwrap_or(parse_focal_mm_35eq(exif)) as f32;
    let aperture = exif.get("FNumber").and_then(|s| first_number(s)).map(|v| v as f32);
    let mut lens = LensModel::identity();
    if let Some(params) = resolve_lens_params(db, &lmaker, &lmodel, native, aperture, None) {
        if params.model == 1 {
            lens.kind = LensKind::PtLens;
            lens.a = params.k1;
            lens.b = params.k2;
            lens.c = params.k3;
        } else if params.k1.abs() + params.k2.abs() + params.k3.abs() > 0.0 {
            lens.kind = LensKind::Poly3;
            lens.a = params.k1;
        }
        lens.vig_k1 = params.vig_k1;
        lens.vig_k2 = params.vig_k2;
        lens.vig_k3 = params.vig_k3;
        lens.has_vig = params.vig_k1.abs() + params.vig_k2.abs() + params.vig_k3.abs() > 1e-8;
    }
    if let Some(vig) = vig_at_longest_distance(db, &lmaker, &lmodel, native) {
        lens.vig_k1 = vig.0;
        lens.vig_k2 = vig.1;
        lens.vig_k3 = vig.2;
        lens.has_vig = vig.0.abs() + vig.1.abs() + vig.2.abs() > 1e-8;
    }
    let calib = lens_crop(db, &lmaker, &lmodel).unwrap_or(1.0);
    let cam = camera_crop(db, exif.get("Make").map(|s| s.as_str()).unwrap_or(""), camera).unwrap_or(calib);
    let mut crops = vec![calib, cam];
    crops.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    crops.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
    if !estimate {
        lens.k = if crop_factor.abs() < 1e-9 { 1.0 } else { calib / crop_factor };
    } else {
        lens.k = if cam.abs() < 1e-9 { 1.0 } else { calib / cam };
    }
    (lens, calib, crops)
}

fn lens_summary(db: Option<&LensDatabase>, maker: &str, model: &str, camera: &str) -> (String, String, f64, f64) {
    let Some(db) = db else {
        return (String::new(), String::new(), 1.0, 1.0);
    };
    let Some((lmaker, lmodel)) = find_best_lens_match(db, maker, model, camera) else {
        let crop = camera_crop(db, maker, camera).unwrap_or(1.0);
        return (String::new(), String::new(), crop, crop);
    };
    let calib = lens_crop(db, &lmaker, &lmodel).unwrap_or(1.0);
    let crop = camera_crop(db, maker, camera).unwrap_or(calib);
    (lmaker, lmodel, crop, calib)
}

fn lens_crop(db: &LensDatabase, maker: &str, model: &str) -> Option<f64> {
    let lenses = db.lenses.iter().filter(|l| l.get_maker().eq_ignore_ascii_case(maker)).collect::<Vec<_>>();
    lenses.iter().find(|l| l.get_display_name(&lenses) == model).and_then(|l| l.cropfactor).map(|v| v as f64)
}

fn camera_crop(db: &LensDatabase, maker: &str, model: &str) -> Option<f64> {
    db.cameras.iter().find(|c| c.get_maker().eq_ignore_ascii_case(maker) && c.get_model().eq_ignore_ascii_case(model)).map(|c| c.cropfactor as f64)
}

fn vig_at_longest_distance(db: &LensDatabase, maker: &str, model: &str, focal: f32) -> Option<(f64, f64, f64)> {
    let lenses = db.lenses.iter().filter(|l| l.get_maker().eq_ignore_ascii_case(maker)).collect::<Vec<_>>();
    let lens = lenses.iter().find(|l| l.get_display_name(&lenses) == model)?;
    let vigs: Vec<_> = lens.calibration.as_ref()?.elements.iter().filter_map(|e| match e {
        CalibrationElement::Vignetting(v) => Some(v),
        _ => None,
    }).collect();
    if vigs.is_empty() {
        return None;
    }
    let best_f = vigs.iter().min_by(|a, b| (a.focal - focal).abs().partial_cmp(&(b.focal - focal).abs()).unwrap_or(std::cmp::Ordering::Equal))?.focal;
    let group: Vec<_> = vigs.into_iter().filter(|v| (v.focal - best_f).abs() < 0.05).collect();
    let best = group.iter().max_by(|a, b| a.distance.unwrap_or(0.0).partial_cmp(&b.distance.unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal))?;
    Some((best.k1.unwrap_or(0.0) as f64, best.k2.unwrap_or(0.0) as f64, best.k3.unwrap_or(0.0) as f64))
}

fn header_size(path: &str) -> Result<(u32, u32), String> {
    let (real, _) = parse_virtual_path(path);
    let bytes = fs::read(&real).map_err(|e| e.to_string())?;
    read_size(real.as_path(), &bytes).ok_or_else(|| "Image size is unknown.".into())
}

fn read_size(path: &Path, bytes: &[u8]) -> Option<(u32, u32)> {
    if let Ok(dim) = image::image_dimensions(path) {
        if dim.0 > 0 && dim.1 > 0 {
            return Some(dim);
        }
    }
    let exif = crate::exif_processing::read_exif(bytes)?;
    let w = exif_uint(&exif, exif::Tag::PixelXDimension).or_else(|| exif_uint(&exif, exif::Tag::ImageWidth))?;
    let h = exif_uint(&exif, exif::Tag::PixelYDimension).or_else(|| exif_uint(&exif, exif::Tag::ImageLength))?;
    if w == 0 || h == 0 { None } else { Some((w, h)) }
}

fn exif_uint(exif: &exif::Exif, tag: exif::Tag) -> Option<u32> {
    let field = exif.get_field(tag, exif::In::PRIMARY).or_else(|| exif.get_field(tag, exif::In::THUMBNAIL))?;
    field.value.get_uint(0).map(|v| v as u32)
}

fn display_size(w: u32, h: u32) -> (u32, u32) {
    let long = w.max(h).max(1);
    if long <= DISPLAY_LONG_SIDE {
        return (w.max(1), h.max(1));
    }
    let s = DISPLAY_LONG_SIDE as f64 / long as f64;
    (((w as f64) * s).round().max(1.0) as u32, ((h as f64) * s).round().max(1.0) as u32)
}

fn resize_winners(src: &[u16], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u16> {
    let mut out = vec![u16::MAX; (dw as usize) * (dh as usize)];
    for y in 0..dh {
        let sy = ((y as u64 * sh as u64) / dh as u64) as u32;
        for x in 0..dw {
            let sx = ((x as u64 * sw as u64) / dw as u64) as u32;
            out[(y * dw + x) as usize] = src[(sy * sw + sx) as usize];
        }
    }
    out
}

fn encode_png(img: &image::RgbImage) -> Result<String, String> {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, ImageFormat::Png).map_err(|e| format!("Failed to encode panorama preview: {}", e))?;
    Ok(format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(buf.get_ref())))
}

// Writes the seven frames the stitcher loads, before exposure matching.
pub fn write_loaded_frames(src_dir: &str, out_dir: &str) -> Result<(), String> {
    let settings: crate::app_settings::AppSettings = serde_json::from_str(
        &fs::read_to_string("/home/dalibor/.local/share/io.github.CyberTimon.RapidRAW/settings.json").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    for n in 8715..=8721 {
        let path = format!("{src_dir}/DSCF{n}.RAF");
        let (rgb, w, h, _) = load_linear(&path, &settings)?;
        let mut file = fs::File::create(format!("{out_dir}/DSCF{n}.f32")).map_err(|e| e.to_string())?;
        file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(&h.to_le_bytes()).map_err(|e| e.to_string())?;
        for chunk in rgb.chunks(1_048_576) {
            let mut buf = Vec::with_capacity(chunk.len() * 4);
            for v in chunk {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            file.write_all(&buf).map_err(|e| e.to_string())?;
        }
        let mean = rgb.iter().map(|v| *v as f64).sum::<f64>() / rgb.len() as f64;
        println!("DSCF{n} {w}x{h} mean={mean:.4}");
    }
    Ok(())
}

fn file_name(path: &str) -> String {
    Path::new(path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string())
}

fn first_number(s: &str) -> Option<f64> {
    let mut num = String::new();
    let mut seen = false;
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            num.push(c);
            seen = true;
        } else if seen {
            break;
        }
    }
    num.parse().ok()
}

fn parse_shutter(s: &str) -> Option<f64> {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("1/") {
        let denom = first_number(rest)?;
        if denom > 0.0 { Some(1.0 / denom) } else { None }
    } else {
        first_number(t)
    }
}
