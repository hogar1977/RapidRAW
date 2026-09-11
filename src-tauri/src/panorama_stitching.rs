use crate::app_settings::load_settings;
use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::formats::is_raw_file;
use crate::image_processing::{apply_linear_to_srgb, apply_srgb_to_linear, downscale_f32_image};
use crate::panorama_utils::auto_crop::max_inscribed_aabb;
use crate::panorama_utils::blend::blend_panorama;
use crate::panorama_utils::bundle_adjust::{
    bundle_adjust, refine_single_row_pitches, refine_single_row_yaws,
};
use crate::panorama_utils::local_warp::build_local_meshes;
use crate::panorama_utils::camera::{
    enforce_single_row, is_likely_single_row, log_alignment_residuals, pose_from_exif_with_lens,
    wave_correct, CameraPose, yaw_pitch_from_rotation,
};
use crate::panorama_utils::debug_log; // PANO_DEBUG — temporary
use crate::panorama_utils::match_graph::build_match_graph;
use crate::panorama_utils::orb::{detect_and_compute, generate_steered_brief_pairs};
use crate::panorama_utils::overlay::generate_overlay;
use crate::panorama_utils::processing::generate_low_detail_mask;
use crate::panorama_utils::projection::{recommend_projection, Projection, ProjectionCanvas};
use crate::panorama_utils::ram::{check_ram, save_canvas_long_side};
use crate::panorama_utils::session::{
    drop_session, DroppedImage, NormalizedCrop, PanoramaSession,
};
use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, GenericImageView, GrayImage, ImageFormat, Rgb32FImage};
use std::fs;
use std::io::Cursor;
use std::path::Path;
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

const PREVIEW_LONG_SIDE: u32 = 2800;
const FEATURE_LONG_SIDE: u32 = 1600;

#[tauri::command]
pub async fn stitch_panorama(
    paths: Vec<String>,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two images to stitch.".to_string());
    }

    let source_paths: Vec<String> = paths
        .iter()
        .map(|p| parse_virtual_path(p).0.to_string_lossy().into_owned())
        .collect();

    let session_handle = state.panorama_session.clone();
    let lens_db = state.lens_db.lock().unwrap().clone();

    let task = tokio::task::spawn_blocking(move || {
        match run_preview_pipeline(source_paths, app_handle.clone(), lens_db) {
            Ok(session) => {
                let payload = serde_json::json!({
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
                });
                *session_handle.lock().unwrap() = Some(session);
                let _ = app_handle.emit("panorama-complete", payload);
                Ok(())
            }
            Err(e) => {
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
pub async fn reproject_panorama(
    projection: String,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let proj = Projection::parse(&projection)
        .ok_or_else(|| format!("Unknown projection: {}", projection))?;

    let session_handle = state.panorama_session.clone();
    let task = tokio::task::spawn_blocking(move || {
        let mut guard = session_handle.lock().unwrap();
        let session = guard
            .as_mut()
            .ok_or_else(|| "No panorama session. Run stitch first.".to_string())?;

        session.selected_projection = proj;
        let _ = app_handle.emit("panorama-progress", "Reprojecting preview...");
        rebuild_preview(session)?;

        let payload = serde_json::json!({
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
        });
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
    let mut guard = state.panorama_session.lock().unwrap();
    drop_session(&mut guard);
    Ok(())
}

#[tauri::command]
pub async fn save_panorama(
    first_path_str: String,
    crop: Option<NormalizedCrop>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let session = {
        let mut guard = state.panorama_session.lock().unwrap();
        guard
            .take()
            .ok_or_else(|| "No panorama session found to save.".to_string())?
    };

    let crop = crop.unwrap_or(session.crop.clone());
    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let task = tokio::task::spawn_blocking(move || {
        let result = save_full_res(&session, &first_path_str, &crop, &settings, app_handle);
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

fn run_preview_pipeline(
    source_paths: Vec<String>,
    app_handle: AppHandle,
    lens_db: Option<std::sync::Arc<crate::lens_correction::LensDatabase>>,
) -> Result<PanoramaSession, String> {
    // PANO_DEBUG — one file per stitch, next to first source image.
    let dbg = debug_log::PanoDebugLog::start(&source_paths);
    if let Some(ref d) = dbg {
        let _ = app_handle.emit(
            "panorama-progress",
            format!("Debug log → {}", d.path_str()),
        );
    }
    debug_log::log_paths(&source_paths);
    debug_log::write(&format!(
        "lens_db_loaded={}",
        lens_db.is_some()
    ));

    let result = run_preview_pipeline_inner(source_paths, app_handle, lens_db);
    match &result {
        Ok(session) => {
            debug_log::write(&format!(
                "session ok: kept={} dropped={} preview={}x{} proj={} png_b64_len={}",
                session.kept_indices.len(),
                session.dropped.len(),
                session.preview_width,
                session.preview_height,
                session.selected_projection.as_str(),
                session.preview_png_base64.len()
            ));
            if let Some(ref d) = dbg {
                d.finish("ok");
            }
        }
        Err(e) => {
            debug_log::write(&format!("ERROR: {}", e));
            if let Some(ref d) = dbg {
                d.finish("error");
            } else {
                debug_log::clear_current();
            }
        }
    }
    result
}

fn run_preview_pipeline_inner(
    source_paths: Vec<String>,
    app_handle: AppHandle,
    lens_db: Option<std::sync::Arc<crate::lens_correction::LensDatabase>>,
) -> Result<PanoramaSession, String> {
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let _ = app_handle.emit("panorama-progress", "[1/6] Loading images...");
    debug_log::section("LOAD");

    let mut preview_images: Vec<Rgb32FImage> = Vec::new();
    let mut low_detail_masks: Vec<GrayImage> = Vec::new();
    let mut poses: Vec<CameraPose> = Vec::new();
    let mut feature_grays: Vec<GrayImage> = Vec::new();
    let mut full_dims: Vec<(u32, u32)> = Vec::new();

    for (i, path) in source_paths.iter().enumerate() {
        let name = Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let _ = app_handle.emit(
            "panorama-progress",
            format!("[1/6] Loading {}/{}: {}", i + 1, source_paths.len(), name),
        );

        let (img, pose, gray_feat, low_mask, dims) =
            load_linear_frame(path, &settings, lens_db.as_deref())?;
        log::info!(
            "Panorama loaded {}: {}x{}, fov_focal_px={:.1}",
            name,
            dims.0,
            dims.1,
            pose.focal_px
        );
        debug_log::write(&format!(
            "loaded[{}] {} full={}x{} focal_px={:.2} lens=k1={:.5} k2={:.5} k3={:.5} model={} rot={:?}",
            i,
            name,
            dims.0,
            dims.1,
            pose.focal_px,
            pose.lens.k1,
            pose.lens.k2,
            pose.lens.k3,
            pose.lens.model,
            pose.rotation_vector
        ));
        debug_log::log_rgb32f_stats(&format!("linear_full[{}]", i), &img);
        debug_log::log_gray_stats(&format!("feature_gray_full[{}]", i), &gray_feat);

        let (pw, ph, _) =
            crate::panorama_utils::processing::calculate_downscale_dimensions_capped(
                dims.0,
                dims.1,
                PREVIEW_LONG_SIDE,
            );
        let preview = downscale_f32_image(&DynamicImage::ImageRgb32F(img), pw, ph);
        let preview_rgb = match preview {
            DynamicImage::ImageRgb32F(v) => v,
            other => other.to_rgb32f(),
        };
        debug_log::log_rgb32f_stats(&format!("preview_linear[{}]", i), &preview_rgb);

        let (fw, fh, _) =
            crate::panorama_utils::processing::calculate_downscale_dimensions_capped(
                dims.0,
                dims.1,
                FEATURE_LONG_SIDE,
            );
        let gray_small = image::imageops::resize(
            &gray_feat,
            fw,
            fh,
            image::imageops::FilterType::Triangle,
        );

        full_dims.push(dims);
        poses.push(pose.with_scaled_size(fw, fh));
        preview_images.push(preview_rgb);
        low_detail_masks.push(image::imageops::resize(
            &low_mask,
            preview_images.last().unwrap().width(),
            preview_images.last().unwrap().height(),
            image::imageops::FilterType::Nearest,
        ));
        feature_grays.push(gray_small);
        debug_log::write(&format!(
            "scales[{}]: preview={}x{} feature={}x{} pose_focal={:.2}",
            i,
            preview_images[i].width(),
            preview_images[i].height(),
            feature_grays[i].width(),
            feature_grays[i].height(),
            poses[i].focal_px
        ));
    }

    let max_w = full_dims.iter().map(|d| d.0).max().unwrap_or(1);
    let max_h = full_dims.iter().map(|d| d.1).max().unwrap_or(1);
    let est_canvas_w = (max_w as f64 * source_paths.len() as f64 * 0.6).ceil() as u32;
    let est_canvas_h = max_h;
    let ram = check_ram(
        max_w,
        max_h,
        source_paths.len(),
        est_canvas_w.max(PREVIEW_LONG_SIDE),
        est_canvas_h.max(PREVIEW_LONG_SIDE / 2),
    );
    debug_log::write(&format!(
        "ram can_proceed={} est_canvas={}x{} msg={:?}",
        ram.can_proceed, est_canvas_w, est_canvas_h, ram.message
    ));
    if !ram.can_proceed {
        return Err(format!(
            "[ram] {}",
            ram.message.unwrap_or_else(|| "Insufficient RAM".into())
        ));
    }

    let _ = app_handle.emit("panorama-progress", "[2/6] Detecting features...");
    debug_log::section("FEATURES");
    let pairs = generate_steered_brief_pairs();
    let features: Vec<_> = feature_grays
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let feats = detect_and_compute(g, &pairs);
            log::info!(
                "Panorama ORB {}: {} features on {}x{}",
                Path::new(&source_paths[i])
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                feats.len(),
                g.width(),
                g.height()
            );
            debug_log::write(&format!(
                "orb[{}] count={} on {}x{}",
                i,
                feats.len(),
                g.width(),
                g.height()
            ));
            feats
        })
        .collect();

    let weak_feat: Vec<_> = features
        .iter()
        .enumerate()
        .filter(|(_, f)| f.len() < 20)
        .map(|(i, f)| {
            format!(
                "{} ({} feats)",
                Path::new(&source_paths[i])
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                f.len()
            )
        })
        .collect();
    if !features.is_empty() && weak_feat.len() == features.len() {
        return Err(format!(
            "[features] Almost no keypoints found in any image: {}. Images may be too soft, blurry, or low-contrast.",
            weak_feat.join(", ")
        ));
    }

    let _ = app_handle.emit("panorama-progress", "[3/6] Matching images...");
    debug_log::section("MATCH GRAPH");
    let graph = build_match_graph(&features, &poses, &source_paths, Some(&feature_grays));
    debug_log::write(&format!(
        "edges={} kept={:?} dropped={:?} adjacent_only_gba={}",
        graph.edges.len(),
        graph.kept,
        graph.dropped,
        graph.adjacent_only_gba
    ));
    for line in graph.diagnostics.lines() {
        debug_log::write(line);
    }
    if graph.kept.len() < 2 {
        return Err(format!(
            "[match] Could not connect at least two images.\n\nDiagnostics:\n{}",
            graph.diagnostics
        ));
    }

    let dropped: Vec<DroppedImage> = graph
        .dropped
        .iter()
        .map(|(idx, reason)| DroppedImage {
            filename: Path::new(&source_paths[*idx])
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| source_paths[*idx].clone()),
            reason: reason.clone(),
        })
        .collect();

    let _ = app_handle.emit("panorama-progress", "[4/6] Optimizing camera poses...");
    debug_log::section("POSE OPTIMIZE");
    let gba_edges: Vec<_> = if graph.adjacent_only_gba {
        graph
            .edges
            .iter()
            .filter(|e| e.i.abs_diff(e.j) == 1)
            .cloned()
            .collect()
    } else {
        graph.edges.clone()
    };
    log::info!(
        "Panorama GBA using {} edges (adjacent_only={})",
        gba_edges.len(),
        graph.adjacent_only_gba
    );
    debug_log::write(&format!(
        "gba_edges={} adjacent_only={}",
        gba_edges.len(),
        graph.adjacent_only_gba
    ));
    if graph.adjacent_only_gba {
        // 1×N: keep yaw-chain init; free SO(3) BA often trades FOV error into pitch,
        // then pitch-share undoes it and leaves windowblind ridge steps.
        for &idx in &graph.kept {
            if let Some(r) = graph.initial_rotations.get(&idx) {
                poses[idx].set_rotation(*r);
            }
        }
        let mean_focal =
            graph.kept.iter().map(|&i| poses[i].focal_px).sum::<f64>() / graph.kept.len() as f64;
        for &idx in &graph.kept {
            poses[idx].focal_px = mean_focal;
        }
        debug_log::write(&format!("1xN path: yaw-chain init, mean_focal={:.2}", mean_focal));
    } else {
        bundle_adjust(
            &mut poses,
            &features,
            &gba_edges,
            &graph.kept,
            &graph.initial_rotations,
        );
        debug_log::write("multi-row path: free SO(3) bundle_adjust done");
    }
    // OpenCV-style waveCorrect: horizontal for 1×N, vertical if pitch span dominates.
    let yaw_span = {
        let mut yaws: Vec<f64> = graph
            .kept
            .iter()
            .map(|&i| yaw_pitch_from_rotation(&poses[i].rotation()).0)
            .collect();
        yaws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        yaws.last().copied().unwrap_or(0.0) - yaws.first().copied().unwrap_or(0.0)
    };
    let pitch_span = {
        let mut pitches: Vec<f64> = graph
            .kept
            .iter()
            .map(|&i| yaw_pitch_from_rotation(&poses[i].rotation()).1)
            .collect();
        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        pitches.last().copied().unwrap_or(0.0) - pitches.first().copied().unwrap_or(0.0)
    };
    let horizontal_wave = yaw_span.abs() >= pitch_span.abs();
    debug_log::write(&format!(
        "waveCorrect horizontal={} yaw_span={:.3}° pitch_span={:.3}°",
        horizontal_wave,
        yaw_span.to_degrees(),
        pitch_span.to_degrees()
    ));
    wave_correct(&mut poses, &graph.kept, horizontal_wave);
    let single_row = is_likely_single_row(&poses, &graph.kept);
    debug_log::write(&format!("is_likely_single_row={}", single_row));
    enforce_single_row(&mut poses, &graph.kept);

    if graph.adjacent_only_gba {
        // 1×N: yaw-chain from pairwise rotations is the trustworthy global pose.
        // Pixel-transfer yaw polish collapses the span under parallax (v5/v6 regressions).
        // Re-apply chain yaws after leveling so waveCorrect cannot shrink the pan.
        if let Some(&anchor) = graph.kept.iter().next() {
            let (_, shared_pitch) = yaw_pitch_from_rotation(&poses[anchor].rotation());
            let mean_focal =
                graph.kept.iter().map(|&i| poses[i].focal_px).sum::<f64>() / graph.kept.len() as f64;
            for &idx in &graph.kept {
                let yaw = graph
                    .initial_rotations
                    .get(&idx)
                    .map(|r| yaw_pitch_from_rotation(r).0)
                    .unwrap_or(0.0);
                poses[idx].set_rotation(crate::panorama_utils::camera::rotation_from_yaw_pitch(
                    yaw,
                    shared_pitch,
                ));
                poses[idx].focal_px = mean_focal;
            }
        }
        debug_log::write("1xN: kept yaw-chain (skipped transfer yaw polish)");
        refine_single_row_pitches(&mut poses, &features, &gba_edges, &graph.kept);
        debug_log::write("1xN: pitch polish done");
    } else if single_row {
        refine_single_row_yaws(&mut poses, &features, &gba_edges, &graph.kept);
        wave_correct(&mut poses, &graph.kept, true);
        enforce_single_row(&mut poses, &graph.kept);
        refine_single_row_yaws(&mut poses, &features, &gba_edges, &graph.kept);
        debug_log::write("single-row yaw polish (2 passes) done");
    }
    log_alignment_residuals(&poses, &features, &gba_edges, &graph.kept);

    // Local CP mesh — only when global residuals are already in a usable band.
    // Large meshes on a bad global pose amplify tearing (v6).
    debug_log::section("LOCAL MESH");
    let mut local_meshes: Vec<crate::panorama_utils::local_warp::ImageMesh> = poses
        .iter()
        .map(|p| crate::panorama_utils::local_warp::ImageMesh::identity(p.width, p.height))
        .collect();
    let transfer_ok = {
        // Quick adjacent transfer median; skip mesh if still catastrophic.
        use std::collections::HashSet;
        let kept_set: HashSet<usize> = graph.kept.iter().copied().collect();
        let mut errs = Vec::new();
        for e in &gba_edges {
            if !kept_set.contains(&e.i) || !kept_set.contains(&e.j) {
                continue;
            }
            for &(ia, ib) in e.inliers.iter().take(40) {
                if ia >= features[e.i].len() || ib >= features[e.j].len() {
                    continue;
                }
                let ka = features[e.i][ia].keypoint;
                let kb = features[e.j][ib].keypoint;
                let wa = poses[e.i].world_bearing_from_pixel(ka.x as f64, ka.y as f64);
                if let Some((x, y)) = poses[e.j].pixel_from_world_bearing(wa) {
                    let dx = x - kb.x as f64;
                    let dy = y - kb.y as f64;
                    errs.push((dx * dx + dy * dy).sqrt());
                }
            }
        }
        if errs.is_empty() {
            true
        } else {
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = errs[errs.len() / 2];
            debug_log::write(&format!("mesh_gate transfer_median={:.2}px", med));
            med < 120.0
        }
    };
    if transfer_ok {
        local_meshes = build_local_meshes(&poses, &features, &gba_edges, &graph.kept);
    } else {
        debug_log::write("local_mesh skipped (transfer residuals too large)");
    }

    for &idx in &graph.kept {
        let (yaw, pitch) = yaw_pitch_from_rotation(&poses[idx].rotation());
        debug_log::write(&format!(
            "pose_feature_res[{}] yaw={:.3}° pitch={:.3}° focal={:.2} size={}x{}",
            idx,
            yaw.to_degrees(),
            pitch.to_degrees(),
            poses[idx].focal_px,
            poses[idx].width,
            poses[idx].height
        ));
    }

    for &idx in &graph.kept {
        let (fw, _, _) = crate::panorama_utils::processing::calculate_downscale_dimensions_capped(
            full_dims[idx].0,
            full_dims[idx].1,
            FEATURE_LONG_SIDE,
        );
        let scale = preview_images[idx].width() as f64 / fw as f64;
        poses[idx].focal_px *= scale;
        poses[idx].width = preview_images[idx].width();
        poses[idx].height = preview_images[idx].height();
        local_meshes[idx] = local_meshes[idx].scaled(poses[idx].width, poses[idx].height);
        debug_log::write(&format!(
            "pose_preview_res[{}] scale={:.4} focal={:.2} size={}x{}",
            idx,
            scale,
            poses[idx].focal_px,
            poses[idx].width,
            poses[idx].height
        ));
    }

    let kept_poses: Vec<CameraPose> = graph.kept.iter().map(|&i| poses[i]).collect();
    let recommended = recommend_projection(&kept_poses);
    debug_log::write(&format!("recommended_projection={}", recommended.as_str()));

    let filenames: Vec<String> = graph
        .kept
        .iter()
        .map(|&i| {
            Path::new(&source_paths[i])
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| source_paths[i].clone())
        })
        .collect();

    let mut session = PanoramaSession {
        source_paths,
        kept_indices: graph.kept,
        poses,
        dropped,
        preview_png_base64: String::new(),
        overlay_png_base64: String::new(),
        winner_map_png_base64: String::new(),
        filenames,
        recommended_projection: recommended,
        selected_projection: recommended,
        crop: NormalizedCrop::default(),
        preview_width: 0,
        preview_height: 0,
        temp_dir: None,
        preview_images,
        low_detail_masks,
        local_meshes,
    };

    // Preview uses the same exposure match + distance blend as Save; Save only re-renders at full res.
    let _ = app_handle.emit(
        "panorama-progress",
        "[5/6] Matching exposure & blending preview...",
    );
    debug_log::section("PREVIEW BLEND");
    rebuild_preview(&mut session).map_err(|e| format!("[preview] {}", e))?;
    let _ = app_handle.emit("panorama-progress", "[6/6] Preview ready");
    Ok(session)
}

fn rebuild_preview(session: &mut PanoramaSession) -> Result<(), String> {
    let kept_poses: Vec<CameraPose> = session
        .kept_indices
        .iter()
        .map(|&i| session.poses[i])
        .collect();
    let canvas =
        ProjectionCanvas::from_poses(&kept_poses, session.selected_projection, PREVIEW_LONG_SIDE);
    debug_log::write(&format!(
        "canvas {}x{} proj={} hfov={:.3}° vfov={:.3}°",
        canvas.width,
        canvas.height,
        canvas.projection.as_str(),
        canvas.hfov.to_degrees(),
        canvas.vfov.to_degrees()
    ));

    let images: Vec<&Rgb32FImage> = session
        .kept_indices
        .iter()
        .map(|&i| &session.preview_images[i])
        .collect();
    let masks: Vec<&GrayImage> = session
        .kept_indices
        .iter()
        .map(|&i| &session.low_detail_masks[i])
        .collect();

    for (li, &idx) in session.kept_indices.iter().enumerate() {
        debug_log::log_rgb32f_stats(&format!("blend_input[{}->local{}]", idx, li), images[li]);
    }

    let blend = blend_panorama(
        &images,
        &masks,
        &session.poses,
        &session.kept_indices,
        &canvas,
        Some(&session.local_meshes),
        None,
    );

    debug_log::log_rgb32f_stats("blend_linear_out", &blend.image);
    debug_log::log_gray_stats("blend_mask", &blend.mask);
    let win_valid = blend.winners.iter().filter(|&&w| w != u16::MAX).count();
    debug_log::write(&format!(
        "winners: valid={}/{} unique={:?}",
        win_valid,
        blend.winners.len(),
        {
            let mut u: Vec<u16> = blend
                .winners
                .iter()
                .copied()
                .filter(|&w| w != u16::MAX)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            u.sort_unstable();
            u
        }
    ));

    session.crop = max_inscribed_aabb(&blend.mask);
    session.preview_width = canvas.width;
    session.preview_height = canvas.height;
    debug_log::write(&format!(
        "crop x={:.4} y={:.4} w={:.4} h={:.4}",
        session.crop.x, session.crop.y, session.crop.width, session.crop.height
    ));

    let overlay = generate_overlay(&blend.winners, canvas.width, canvas.height)?;
    session.overlay_png_base64 = general_purpose::STANDARD.encode(&overlay.boundary_png);
    session.winner_map_png_base64 = general_purpose::STANDARD.encode(&overlay.winner_map_png);

    let display = apply_linear_to_srgb(DynamicImage::ImageRgb32F(blend.image));
    debug_log::log_dynamic_stats("preview_after_linear_to_srgb", &display);
    session.preview_png_base64 = encode_preview_png(&display)?;
    debug_log::write(&format!(
        "preview_png_base64_len={}",
        session.preview_png_base64.len()
    ));
    Ok(())
}

fn emit_save_progress(app: &AppHandle, percent: u32, message: &str) {
    let _ = app.emit(
        "panorama-save-progress",
        serde_json::json!({
            "percent": percent.min(100),
            "message": message,
        }),
    );
}

fn save_full_res(
    session: &PanoramaSession,
    first_path_str: &str,
    crop: &NormalizedCrop,
    settings: &crate::app_settings::AppSettings,
    app_handle: AppHandle,
) -> Result<String, String> {
    let n = session.kept_indices.len().max(1);
    emit_save_progress(&app_handle, 1, "Preparing full-resolution save...");

    let mut full_images: Vec<Rgb32FImage> = Vec::new();
    let mut full_masks: Vec<GrayImage> = Vec::new();
    let mut full_poses = session.poses.clone();

    for (step, &idx) in session.kept_indices.iter().enumerate() {
        let path = &session.source_paths[idx];
        let name = Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        // Reload phase: 2% .. 38%
        let pct = 2 + ((step as u32 * 36) / n as u32);
        emit_save_progress(
            &app_handle,
            pct,
            &format!("Reloading {}/{}: {}", step + 1, n, name),
        );
        // Pose/lens already come from the preview session; no need to re-resolve lensfun here.
        let (img, _pose, gray, low_mask, dims) = load_linear_frame(path, settings, None)?;
        let scale = dims.0 as f64 / session.poses[idx].width as f64;
        full_poses[idx].focal_px = session.poses[idx].focal_px * scale;
        full_poses[idx].width = dims.0;
        full_poses[idx].height = dims.1;
        full_poses[idx].rotation_vector = session.poses[idx].rotation_vector;
        full_poses[idx].lens = session.poses[idx].lens;
        full_images.push(img);
        full_masks.push(low_mask);
        let _ = gray;
    }

    emit_save_progress(&app_handle, 40, "Building projection canvas...");

    let kept_poses: Vec<CameraPose> = session
        .kept_indices
        .iter()
        .map(|&i| full_poses[i])
        .collect();
    let long_side = save_canvas_long_side(&kept_poses, session.selected_projection);
    let canvas =
        ProjectionCanvas::from_poses(&kept_poses, session.selected_projection, long_side.max(2000));
    debug_log::write(&format!(
        "save_canvas {}x{} long_side={} proj={} hfov={:.3}° vfov={:.3}°",
        canvas.width,
        canvas.height,
        long_side,
        canvas.projection.as_str(),
        canvas.hfov.to_degrees(),
        canvas.vfov.to_degrees()
    ));

    let ram = check_ram(
        full_poses[session.kept_indices[0]].width,
        full_poses[session.kept_indices[0]].height,
        session.kept_indices.len(),
        canvas.width,
        canvas.height,
    );
    debug_log::write(&format!(
        "save_ram can_proceed={} est_mb={} budget_mb={} msg={:?}",
        ram.can_proceed,
        ram.estimated_bytes / (1024 * 1024),
        ram.available_bytes / (1024 * 1024),
        ram.message
    ));
    if !ram.can_proceed {
        return Err(ram.message.unwrap_or_else(|| "Insufficient RAM".into()));
    }

    let images: Vec<&Rgb32FImage> = full_images.iter().collect();
    let masks: Vec<&GrayImage> = full_masks.iter().collect();
    let local_indices: Vec<usize> = (0..session.kept_indices.len()).collect();
    let poses_for_blend: Vec<CameraPose> = session
        .kept_indices
        .iter()
        .map(|&i| full_poses[i])
        .collect();
    let meshes_for_blend: Vec<_> = session
        .kept_indices
        .iter()
        .map(|&i| {
            let m = &session.local_meshes[i];
            m.scaled(full_poses[i].width, full_poses[i].height)
        })
        .collect();

    // Blend phase: 42% .. 88%
    let mut last_emitted = 0u32;
    let mut blend_progress = |frac: f32, msg: &str| {
        let pct = 42 + (frac.clamp(0.0, 1.0) * 46.0) as u32;
        if pct >= last_emitted + 1 || frac >= 1.0 {
            last_emitted = pct;
            emit_save_progress(&app_handle, pct, msg);
        }
    };

    let blend = blend_panorama(
        &images,
        &masks,
        &poses_for_blend,
        &local_indices,
        &canvas,
        Some(&meshes_for_blend),
        Some(&mut blend_progress),
    );

    emit_save_progress(&app_handle, 90, "Cropping & converting color...");

    let x0 = (crop.x * canvas.width as f64).round().max(0.0) as u32;
    let y0 = (crop.y * canvas.height as f64).round().max(0.0) as u32;
    let cw = (crop.width * canvas.width as f64)
        .round()
        .max(1.0)
        .min((canvas.width - x0) as f64) as u32;
    let ch = (crop.height * canvas.height as f64)
        .round()
        .max(1.0)
        .min((canvas.height - y0) as f64) as u32;

    let cropped = image::imageops::crop_imm(&blend.image, x0, y0, cw, ch).to_image();
    let display = apply_linear_to_srgb(DynamicImage::ImageRgb32F(cropped));
    let rgb16 = display.to_rgb16();

    emit_save_progress(&app_handle, 95, "Writing TIFF...");

    let (first_path, _) = parse_virtual_path(first_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("panorama");
    let output_path = parent_dir.join(format!("{}_Pano.tiff", stem));

    rgb16
        .save_with_format(&output_path, ImageFormat::Tiff)
        .map_err(|e| format!("Failed to save panorama: {}", e))?;

    let (real_path, _) = parse_virtual_path(first_path_str);
    let _ = crate::exif_processing::write_rrexif_sidecar(&real_path.to_string_lossy(), &output_path);

    emit_save_progress(&app_handle, 100, "Save complete");

    Ok(output_path.to_string_lossy().to_string())
}

fn load_linear_frame(
    path: &str,
    settings: &crate::app_settings::AppSettings,
    lens_db: Option<&crate::lens_correction::LensDatabase>,
) -> Result<(Rgb32FImage, CameraPose, GrayImage, GrayImage, (u32, u32)), String> {
    let file_bytes = fs::read(path).map_err(|e| {
        format!(
            "[load] Failed to read image {}: {}",
            Path::new(path)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string()),
            e
        )
    })?;
    let mut dynamic = crate::image_loader::load_base_image_from_bytes(
        &file_bytes,
        path,
        false,
        settings,
        None,
    )
    .map_err(|e| {
        format!(
            "[load] Failed to decode {}: {}",
            Path::new(path)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string()),
            e
        )
    })?;

    if !is_raw_file(path) {
        dynamic = apply_srgb_to_linear(dynamic);
    }

    let dims = dynamic.dimensions();
    let exif = crate::exif_processing::read_exif_data_from_bytes(path, &file_bytes);
    let pose = pose_from_exif_with_lens(&exif, dims.0, dims.1, lens_db);
    debug_log::write(&format!(
        "exif_focal path={} FocalLength={:?} FocalLengthIn35mmFilm={:?} ScaleFactor35efl={:?} => focal_px={:.2}",
        Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        exif.get("FocalLength"),
        exif.get("FocalLengthIn35mmFilm"),
        exif.get("ScaleFactor35efl")
            .or_else(|| exif.get("ScaleFactor35Efl")),
        pose.focal_px
    ));

    let display = apply_linear_to_srgb(dynamic.clone());
    let gray_raw = image::imageops::colorops::grayscale(&display.to_rgb8());
    let gray = crate::panorama_utils::processing::enhance_for_matching(&gray_raw);
    let low = generate_low_detail_mask(&gray);
    Ok((dynamic.to_rgb32f(), pose, gray, low, dims))
}

fn encode_preview_png(image: &DynamicImage) -> Result<String, String> {
    let mut buf = Cursor::new(Vec::new());
    image
        .to_rgb8()
        .write_to(&mut buf, ImageFormat::Png)
        .map_err(|e| format!("Failed to encode panorama preview: {}", e))?;
    Ok(format!(
        "data:image/png;base64,{}",
        general_purpose::STANDARD.encode(buf.get_ref())
    ))
}
