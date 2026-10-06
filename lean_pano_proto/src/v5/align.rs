use super::jpeg::{self, QuarterJpeg};
use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};
use nalgebra::Matrix3;
use rapidraw_lib::panorama_stitching::{self, PanoramaLensSetup};
use rapidraw_lib::panorama_utils::v86::ba::bundle_rotations;
use rapidraw_lib::panorama_utils::v86::const_::{focal_px_from_35eq, QUIET_TEX};
use rapidraw_lib::panorama_utils::v86::photo;
use rapidraw_lib::panorama_utils::v86::geom::{self, Projection};
use rapidraw_lib::panorama_utils::v86::layout;
use rapidraw_lib::panorama_utils::v86::lens::LensModel;
use rapidraw_lib::panorama_utils::v86::level::level_poses;
use rapidraw_lib::panorama_utils::v86::local_align::{self, RecordedShift};
use rapidraw_lib::panorama_utils::v86::match_::match_all;
use rapidraw_lib::panorama_utils::v86::stitch::{self, InputFrame, StitchResult};
use rapidraw_lib::panorama_utils::v86::trace;
use rapidraw_lib::panorama_utils::v86::warp::canvas_geom;
use rayon::prelude::*;
use std::path::Path;

const POINT_SIDE: u32 = 800;
const FEATURE_CAP: usize = 2000;
const PREVIEW_LONG_SIDE: u32 = 3600;

pub struct Solve {
    pub rots: Vec<Matrix3<f64>>,
    pub indices: Vec<usize>,
    pub sources: Vec<InputFrame>,
    pub jpeg_w: u32,
    pub jpeg_h: u32,
    pub focal35: f64,
    pub lens: LensModel,
    pub crop_factor: f64,
    pub calib: f64,
}

pub struct Preview {
    pub frames: Vec<InputFrame>,
    pub indices: Vec<usize>,
    pub rots: Vec<Matrix3<f64>>,
    pub result: StitchResult,
    pub shifts: Vec<Option<RecordedShift>>,
    pub lens: LensModel,
    pub focal35: f64,
    pub focal: f64,
    pub jpeg_w: u32,
    pub jpeg_h: u32,
    pub crop_factor: f64,
    pub calib: f64,
}

pub fn solve_paths(paths: &[String], db_dir: &Path) -> Result<Solve, String> {
    let exif = panorama_stitching::exif_of(&paths[0])?;
    let setup: PanoramaLensSetup = panorama_stitching::panorama_lens_setup(db_dir, &exif);
    let decoded: Result<Vec<QuarterJpeg>, String> = paths.par_iter().map(|path| jpeg::decode_quarter(path)).collect();
    let decoded = decoded?;
    let mut all_sources = Vec::with_capacity(decoded.len());
    for (path, image) in paths.iter().zip(decoded.iter()) {
        trace::line(&format!(
            "jpeg {} quarter {}x{} full {}x{}",
            file_name(path),
            image.width,
            image.height,
            image.full_w,
            image.full_h
        ));
        all_sources.push(InputFrame {
            name: file_name(path),
            width: image.width,
            height: image.height,
            rgb: srgb8_to_linear(&image.rgb),
        });
    }
    let jpeg_w = decoded[0].full_w;
    let jpeg_h = decoded[0].full_h;
    let (point_w, point_h) = fit_long_side(all_sources[0].width, all_sources[0].height, POINT_SIDE);
    let point_frames: Vec<InputFrame> = all_sources
        .iter()
        .map(|frame| InputFrame {
            name: frame.name.clone(),
            width: point_w,
            height: point_h,
            rgb: shrink(&frame.rgb, frame.width, frame.height, point_w, point_h),
        })
        .collect();
    let focal_points = focal_px_from_35eq(point_w, point_h, setup.focal35);
    let mut feats: Vec<_> = point_frames.par_iter().map(stitch::find_points).collect();
    for found in &mut feats {
        found.truncate(FEATURE_CAP);
    }
    trace::gate()?;
    let pairs = match_all(&feats);
    for pair in &pairs {
        if !pair.pa.is_empty() {
            trace::line(&format!("match {}<->{}: {} good", point_frames[pair.a].name, point_frames[pair.b].name, pair.pa.len()));
        }
    }
    let (rots_all, kept) = rotations(&pairs, point_frames.len(), point_w, point_h, focal_points, &setup.lens)?;
    let mut order = kept;
    order.sort_unstable();
    let indices = order.clone();
    let rots: Vec<_> = order.iter().map(|&i| rots_all[i]).collect();
    let sources: Vec<_> = order.iter().map(|&i| InputFrame {
        name: all_sources[i].name.clone(),
        width: all_sources[i].width,
        height: all_sources[i].height,
        rgb: std::mem::take(&mut all_sources[i].rgb),
    }).collect();
    trace::line(&format!("aligned kept={} focal35={:.1} jpeg={jpeg_w}x{jpeg_h} points={point_w}x{point_h}", rots.len(), setup.focal35));
    Ok(Solve { rots, indices, sources, jpeg_w, jpeg_h, focal35: setup.focal35, lens: setup.lens, crop_factor: setup.crop_factor, calib: setup.calib })
}

pub fn preview_at(solve: Solve, projection: Option<Projection>) -> Result<Preview, String> {
    let src_w = solve.sources[0].width;
    let src_h = solve.sources[0].height;
    let focal_src = focal_px_from_35eq(src_w, src_h, solve.focal35);
    let recommended = choose_projection(&solve.rots, focal_src, src_w, src_h, &solve.lens);
    let asked = projection.unwrap_or(recommended);
    let (geom, used_guess) = canvas_geom(&solve.rots, focal_src, src_w, src_h, &solve.lens, asked);
    let long = geom.width.max(geom.height).max(1);
    let scale = if long > PREVIEW_LONG_SIDE { PREVIEW_LONG_SIDE as f64 / long as f64 } else { 1.0 };
    let dw = ((src_w as f64) * scale).round().max(1.0) as u32;
    let dh = ((src_h as f64) * scale).round().max(1.0) as u32;
    let focal = focal_px_from_35eq(dw, dh, solve.focal35);
    let mut frames: Vec<InputFrame> = solve
        .sources
        .iter()
        .map(|frame| InputFrame {
            name: frame.name.clone(),
            width: dw,
            height: dh,
            rgb: shrink(&frame.rgb, frame.width, frame.height, dw, dh),
        })
        .collect();
    drop(solve.sources);
    scale_quiet_sky(&mut frames);
    let indices = solve.indices;
    let kept: Vec<usize> = (0..frames.len()).collect();
    local_align::begin_shift_log();
    let result = stitch::compose(&frames, &solve.rots, &kept, solve.lens, focal, solve.focal35, used_guess, false, &|message| {
        trace::line(message);
    })?;
    let shifts = local_align::end_shift_log();
    trace::line(&format!("preview canvas {} {}x{} shifts={}", result.used.as_str(), result.width, result.height, shifts.len()));
    Ok(Preview {
        frames,
        indices,
        rots: solve.rots,
        result,
        shifts,
        lens: solve.lens,
        focal35: solve.focal35,
        focal,
        jpeg_w: solve.jpeg_w,
        jpeg_h: solve.jpeg_h,
        crop_factor: solve.crop_factor,
        calib: solve.calib,
    })
}

#[allow(dead_code)]
pub fn redraw(preview: &Preview, projection: Projection) -> Result<StitchResult, String> {
    let kept: Vec<usize> = (0..preview.frames.len()).collect();
    stitch::rerender(
        &preview.frames,
        &preview.rots,
        &kept,
        &preview.lens,
        preview.focal,
        preview.focal35,
        &preview.result.photo,
        projection,
        preview.result.recommended,
        false,
        &|message| trace::line(message),
    )
}

pub fn choose_projection(rots: &[Matrix3<f64>], focal: f64, w: u32, h: u32, lens: &LensModel) -> Projection {
    let layout = layout::detect_layout(rots, focal, w, h);
    let recommended = geom::recommend_projection(&layout.yaws, &layout.pitches, layout.frame_hfov, layout.frame_vfov);
    let (_, used) = canvas_geom(rots, focal, w, h, lens, recommended);
    used
}

fn rotations(
    pairs: &[rapidraw_lib::panorama_utils::v86::match_::PairMatches],
    n_frames: usize,
    w: u32,
    h: u32,
    f: f64,
    lens: &LensModel,
) -> Result<(Vec<Matrix3<f64>>, Vec<usize>), String> {
    let n = n_frames.max(pairs.iter().map(|p| p.a.max(p.b)).max().unwrap_or(0) + 1);
    let mut geom_pairs = Vec::new();
    let mut edges = Vec::new();
    let mut strong_pairs = Vec::new();
    for p in pairs {
        if p.pa.len() < 6 {
            continue;
        }
        let ba = geom::bearings(&p.pa, f, w, h, lens);
        let bb = geom::bearings(&p.pb, f, w, h, lens);
        let Some((rot, inl, spread, mask)) = geom::ransac_rotation(&ba, &bb, f, &p.pa, w, h, geom::default_ransac_iters()) else {
            continue;
        };
        let mut pa = Vec::new();
        let mut pb = Vec::new();
        for (i, on) in mask.iter().enumerate() {
            if *on {
                pa.push(p.pa[i]);
                pb.push(p.pb[i]);
            }
        }
        if geom::strong(inl, spread) {
            edges.push((p.a, p.b, inl));
            strong_pairs.push((p.a, p.b, inl, spread));
        }
        geom_pairs.push((p.a, p.b, rot, pa, pb, inl, spread));
    }
    let names: Vec<String> = (0..n).map(|i| i.to_string()).collect();
    let kept = geom::largest_component(&names, &strong_pairs);
    if kept.len() < 2 {
        return Err("Could not align the photos. They may not overlap enough.".into());
    }
    let tree_edges: Vec<(usize, usize, usize)> = edges.into_iter().filter(|(a, b, _)| kept.contains(a) && kept.contains(b)).collect();
    let tree = geom::spanning_tree(&kept, &tree_edges);
    let mut parent = vec![None; n];
    for i in 0..tree.parent.len().min(n) {
        parent[i] = tree.parent[i];
    }
    let rots_all = bundle_rotations(&geom_pairs, tree.root, &parent, &tree.order, w, h, f, lens);
    let leveled = {
        let sub: Vec<Matrix3<f64>> = kept.iter().map(|&i| rots_all.get(i).copied().unwrap_or(Matrix3::identity())).collect();
        level_poses(&sub)
    };
    let mut rots = vec![Matrix3::identity(); n];
    for (k, &i) in kept.iter().enumerate() {
        rots[i] = leveled[k];
    }
    Ok((rots, kept))
}

pub fn shrink(rgb: &[f32], w: u32, h: u32, dw: u32, dh: u32) -> Vec<f32> {
    if dw == w && dh == h {
        return rgb.to_vec();
    }
    let image = ImageBuffer::<Rgb<f32>, _>::from_raw(w, h, rgb.to_vec()).unwrap_or_else(|| ImageBuffer::new(w, h));
    image::imageops::resize(&image, dw, dh, FilterType::Triangle).into_raw()
}

pub fn fit_long_side(w: u32, h: u32, long_side: u32) -> (u32, u32) {
    let long = w.max(h).max(1);
    if long <= long_side {
        return (w.max(1), h.max(1));
    }
    let scale = long_side as f64 / long as f64;
    (((w as f64) * scale).round().max(1.0) as u32, ((h as f64) * scale).round().max(1.0) as u32)
}

fn scale_quiet_sky(frames: &mut [InputFrame]) {
    for frame in frames {
        let (scalar, median) = quiet_sky_scalar(&frame.rgb, frame.width, frame.height);
        match median {
            Some(value) => trace::line(&format!("sky scale {} {scalar:.3} median {value:.4}", frame.name)),
            None => trace::line(&format!("sky scale {} 1.000 median none", frame.name)),
        }
        if (scalar - 1.0).abs() > 1e-4 {
            for pixel in &mut frame.rgb {
                *pixel *= scalar;
            }
        }
    }
}

fn quiet_sky_scalar(rgb: &[f32], w: u32, h: u32) -> (f32, Option<f32>) {
    let w = w as usize;
    let h = h as usize;
    if w < 2 || h < 2 {
        return (1.0, None);
    }
    let y_stop = (h / 4).max(2).min(h);
    let mut samples = Vec::new();
    let mut y = 1usize;
    while y < y_stop {
        let mut x = 1usize;
        while x < w {
            let i = y * w + x;
            let l = photo::luma(&rgb[i * 3..i * 3 + 3]);
            let left = photo::luma(&rgb[(i - 1) * 3..(i - 1) * 3 + 3]);
            let up = photo::luma(&rgb[(i - w) * 3..(i - w) * 3 + 3]);
            let tex = ((l - left).abs() + (l - up).abs()).min(0.5) * 2.0;
            if tex < QUIET_TEX {
                samples.push(l);
            }
            x += 32;
        }
        y += 32;
    }
    if samples.is_empty() {
        return (1.0, None);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = samples[samples.len() / 2];
    if median >= 0.08 && median <= 0.30 {
        return (1.0, Some(median));
    }
    if median <= 1e-6 {
        return (1.0, Some(median));
    }
    let aim = 0.15_f32.clamp(0.004, 0.6);
    (aim / median, Some(median))
}

fn srgb8_to_linear(rgb: &[u8]) -> Vec<f32> {
    rgb.iter()
        .map(|&v| {
            let c = v as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        })
        .collect()
}

fn file_name(path: &str) -> String {
    Path::new(path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string())
}
