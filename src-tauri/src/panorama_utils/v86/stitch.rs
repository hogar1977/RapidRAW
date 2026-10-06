use super::ba::bundle_rotations;
use super::blend::{blend_levels, blur_mask, pyr_blend};
use super::clahe;
use super::const_::{focal_px_from_35eq, DISPLAY_LONG_SIDE, POINT_LONG_SIDE};
use rayon::prelude::*;
use super::crop::inscribed_crop;
use super::geom::{self, Projection};
use super::layout::{self, Layout};
use super::lens::LensModel;
use super::level::level_poses;
use super::local_align::align_incoming;
use super::match_::match_all;
use super::photo::{self, Photo};
use super::seam::graphcut_mask;
use super::sift::{self, Feature};
use super::trace;
use super::warp::{self, CanvasGeom};
use image::{DynamicImage, Rgb32FImage};
use nalgebra::Matrix3;

pub struct InputFrame {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<f32>,
}

pub struct StitchResult {
    pub rgb: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub winners: Vec<u16>,
    pub crop_x: f64,
    pub crop_y: f64,
    pub crop_w: f64,
    pub crop_h: f64,
    pub recommended: Projection,
    pub used: Projection,
    pub dropped: Vec<(String, String)>,
    pub kept: Vec<usize>,
    pub rotations: Vec<Matrix3<f64>>,
    pub focal_px: f64,
    pub focal35: f64,
    pub lens: LensModel,
    pub photo: Photo,
    pub names: Vec<String>,
}

// Builds one panorama from the photos, estimating the lens only when asked.
pub fn stitch(
    frames: &[InputFrame],
    mut lens: LensModel,
    mut focal35: f64,
    estimate: bool,
    crop_options: &[f64],
    calib_crop: f64,
    half: bool,
    progress: &dyn Fn(&str),
) -> Result<StitchResult, String> {
    if frames.len() < 2 {
        return Err("Please select at least two images to stitch.".into());
    }
    let w = frames[0].width;
    let h = frames[0].height;
    trace::gate()?;
    progress("Finding matching points...");
    let feats = detect_all(frames, progress)?;
    trace::gate()?;
    let pairs = match_all(&feats);
    for p in &pairs {
        if !p.pa.is_empty() {
            let an = frames.get(p.a).map(|f| f.name.as_str()).unwrap_or("?");
            let bn = frames.get(p.b).map(|f| f.name.as_str()).unwrap_or("?");
            trace::line(&format!("match {an}<->{bn}: {} good", p.pa.len()));
        }
    }
    trace::gate()?;
    if estimate {
        progress("Estimating lens and focal length...");
        let refs: Vec<(&[f64; 2], &[f64; 2])> = Vec::new();
        let _ = refs;
        let packed: Vec<(Vec<[f64; 2]>, Vec<[f64; 2]>)> = pairs.iter().map(|p| (p.pa.clone(), p.pb.clone())).collect();
        let views: Vec<(&[[f64; 2]], &[[f64; 2]])> = packed.iter().map(|(a, b)| (a.as_slice(), b.as_slice())).collect();
        let id = LensModel::identity();
        let (_, f0) = geom::estimate_focal(&views, w, h, &id);
        let mut best_crop = crop_options.first().copied().unwrap_or(1.0);
        let mut best_inl = 0usize;
        if !crop_options.is_empty() {
            for &crop in crop_options {
                let mut trial = lens;
                trial.k = if crop.abs() < 1e-6 { 1.0 } else { calib_crop / crop };
                let (fpx, _) = geom::estimate_focal(&views, w, h, &trial);
                let inl = score_inliers(&pairs, &feats, fpx, w, h, &trial);
                if inl > best_inl {
                    best_inl = inl;
                    best_crop = crop;
                }
            }
            lens.k = if best_crop.abs() < 1e-6 { 1.0 } else { calib_crop / best_crop };
        }
        let (fpx, f35) = geom::estimate_focal(&views, w, h, &lens);
        let _ = f0;
        focal35 = f35;
        let _ = fpx;
    }
    let focal_px = focal_px_from_35eq(w, h, focal35);
    progress("Aligning photos...");
    let solved = solve_rotations(&pairs, w, h, focal_px, &lens)?;
    trace::line(&format!("aligned kept={} dropped={}", solved.kept.len(), solved.dropped.len()));
    trace::gate()?;
    let layout = layout::detect_layout(&solved.rots, focal_px, w, h);
    let recommended = geom::recommend_projection(&layout.yaws, &layout.pitches, layout.frame_hfov, layout.frame_vfov);
    let preview = shrink_frames(frames, DISPLAY_LONG_SIDE);
    let scale = preview[0].width as f64 / frames[0].width.max(1) as f64;
    let mut drawn = render_with(
        &preview,
        &solved,
        layout,
        lens,
        focal_px * scale,
        focal35,
        recommended,
        recommended,
        half,
        progress,
    )?;
    drawn.focal_px = focal_px;
    drawn.focal35 = focal35;
    Ok(drawn)
}

// Shrinks each photo so a later warp can use a smaller canvas.
pub fn shrink_frames(frames: &[InputFrame], long_side: u32) -> Vec<InputFrame> {
    frames
        .par_iter()
        .map(|frame| {
            let (dw, dh) = fit_long_side(frame.width, frame.height, long_side);
            if dw == frame.width && dh == frame.height {
                return InputFrame {
                    name: frame.name.clone(),
                    width: frame.width,
                    height: frame.height,
                    rgb: frame.rgb.clone(),
                };
            }
            let small = if let Some(view) = image::ImageBuffer::<image::Rgb<f32>, &[f32]>::from_raw(frame.width, frame.height, frame.rgb.as_slice()) {
                image::imageops::resize(&view, dw, dh, image::imageops::FilterType::Triangle)
            } else {
                let owned = Rgb32FImage::from_raw(frame.width, frame.height, frame.rgb.clone()).unwrap_or_else(|| Rgb32FImage::new(frame.width, frame.height));
                image::imageops::resize(&owned, dw, dh, image::imageops::FilterType::Triangle)
            };
            InputFrame { name: frame.name.clone(), width: dw, height: dh, rgb: small.into_raw() }
        })
        .collect()
}

// Draws the full-size panorama from rotations that were already solved.
pub fn compose(
    frames: &[InputFrame],
    rotations: &[Matrix3<f64>],
    kept: &[usize],
    lens: LensModel,
    focal_px: f64,
    focal35: f64,
    projection: Projection,
    half: bool,
    progress: &dyn Fn(&str),
) -> Result<StitchResult, String> {
    let w = frames[0].width;
    let h = frames[0].height;
    let layout = layout::detect_layout(rotations, focal_px, w, h);
    let recommended = geom::recommend_projection(&layout.yaws, &layout.pitches, layout.frame_hfov, layout.frame_vfov);
    let solved = Solved {
        rots: rotations.to_vec(),
        kept: kept.to_vec(),
        dropped: Vec::new(),
        order_fallback: kept.to_vec(),
    };
    render_with(frames, &solved, layout, lens, focal_px, focal35, recommended, projection, half, progress)
}

fn fit_long_side(w: u32, h: u32, long_side: u32) -> (u32, u32) {
    let long = w.max(h).max(1);
    if long <= long_side || long_side == 0 {
        return (w.max(1), h.max(1));
    }
    let scale = long_side as f64 / long as f64;
    (((w as f64) * scale).round().max(1.0) as u32, ((h as f64) * scale).round().max(1.0) as u32)
}

// Draws the panorama again for a different projection without matching the photos over.
pub fn rerender(
    frames: &[InputFrame],
    rotations: &[Matrix3<f64>],
    kept: &[usize],
    lens: &LensModel,
    focal_px: f64,
    focal35: f64,
    photo: &Photo,
    projection: Projection,
    recommended: Projection,
    half: bool,
    progress: &dyn Fn(&str),
) -> Result<StitchResult, String> {
    let w = frames[0].width;
    let h = frames[0].height;
    let layout = layout::detect_layout(rotations, focal_px, w, h);
    let solved = Solved {
        rots: rotations.to_vec(),
        kept: kept.to_vec(),
        dropped: Vec::new(),
        order_fallback: kept.to_vec(),
    };
    let mut out = render_known(frames, &solved, &layout, lens, photo, focal_px, projection, half, progress)?;
    out.focal35 = focal35;
    out.recommended = recommended;
    out.lens = *lens;
    out.photo = Photo { gains: photo.gains.clone(), coef: photo.coef.clone(), pedestal: photo.pedestal };
    out.names = frames.iter().map(|f| f.name.clone()).collect();
    out.dropped = Vec::new();
    Ok(out)
}

struct Solved {
    rots: Vec<Matrix3<f64>>,
    kept: Vec<usize>,
    dropped: Vec<(String, String)>,
    order_fallback: Vec<usize>,
}

fn solve_rotations(pairs: &[super::match_::PairMatches], w: u32, h: u32, f: f64, lens: &LensModel) -> Result<Solved, String> {
    let n = pairs.iter().map(|p| p.a.max(p.b)).max().unwrap_or(0) + 1;
    let mut geom_pairs = Vec::new();
    let mut edges = Vec::new();
    let mut strong_pairs = Vec::new();
    for p in pairs {
        trace::gate()?;
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
    let mut dropped = Vec::new();
    for i in 0..n {
        if !kept.contains(&i) {
            dropped.push((i.to_string(), "Not connected to the main group".into()));
        }
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
    Ok(Solved { rots, kept: kept.clone(), dropped, order_fallback: tree.order })
}

fn render_with(
    frames: &[InputFrame],
    solved: &Solved,
    layout: Layout,
    lens: LensModel,
    focal_px: f64,
    focal35: f64,
    recommended: Projection,
    projection: Projection,
    half: bool,
    progress: &dyn Fn(&str),
) -> Result<StitchResult, String> {
    let order = layout::composite_order(&layout, &solved.kept, &solved.order_fallback);
    let kept_frames: Vec<usize> = order.clone();
    let rots: Vec<Matrix3<f64>> = kept_frames.iter().map(|&i| solved.rots[i]).collect();
    progress("Matching brightness...");
    let (geom, used) = warp::canvas_geom(&rots, focal_px, frames[kept_frames[0]].width, frames[kept_frames[0]].height, &lens, projection);
    trace::line(&format!("canvas {} {}x{}", used.as_str(), geom.width, geom.height));
    let (photo, mut placed) = fit_photo(frames, &kept_frames, &rots, focal_px, &lens, &geom)?;
    progress("Blending photos...");
    let (rgb, winners, width, height) = composite(&mut placed, &photo, &geom, half)?;
    let cover: Vec<bool> = winners.iter().map(|id| *id != u16::MAX).collect();
    let crop = inscribed_crop(&cover, width as usize, height as usize);
    let mut dropped = Vec::new();
    for (idx, reason) in &solved.dropped {
        if let Ok(i) = idx.parse::<usize>() {
            if i < frames.len() {
                dropped.push((frames[i].name.clone(), reason.clone()));
            }
        }
    }
    let mut full_rots = solved.rots.clone();
    for (k, &i) in kept_frames.iter().enumerate() {
        full_rots[i] = rots[k];
    }
    Ok(StitchResult {
        rgb,
        width,
        height,
        winners,
        crop_x: crop.x0 as f64 / width.max(1) as f64,
        crop_y: crop.y0 as f64 / height.max(1) as f64,
        crop_w: (crop.x1 - crop.x0) as f64 / width.max(1) as f64,
        crop_h: (crop.y1 - crop.y0) as f64 / height.max(1) as f64,
        recommended,
        used,
        dropped,
        kept: kept_frames,
        rotations: full_rots,
        focal_px,
        focal35,
        lens,
        photo,
        names: frames.iter().map(|f| f.name.clone()).collect(),
    })
}

fn render_known(
    frames: &[InputFrame],
    solved: &Solved,
    layout: &Layout,
    lens: &LensModel,
    photo: &Photo,
    focal_px: f64,
    projection: Projection,
    half: bool,
    progress: &dyn Fn(&str),
) -> Result<StitchResult, String> {
    progress("Blending photos...");
    let order = layout::composite_order(layout, &solved.kept, &solved.order_fallback);
    let rots: Vec<Matrix3<f64>> = order.iter().map(|&i| solved.rots[i]).collect();
    let (geom, used) = warp::canvas_geom(&rots, focal_px, frames[order[0]].width, frames[order[0]].height, lens, projection);
    trace::line(&format!("canvas {} {}x{}", used.as_str(), geom.width, geom.height));
    let mut placed = place_frames(frames, &order, &rots, focal_px, lens, &geom)?;
    let (rgb, winners, width, height) = composite(&mut placed, photo, &geom, half)?;
    let cover: Vec<bool> = winners.iter().map(|id| *id != u16::MAX).collect();
    let crop = inscribed_crop(&cover, width as usize, height as usize);
    Ok(StitchResult {
        rgb,
        width,
        height,
        winners,
        crop_x: crop.x0 as f64 / width.max(1) as f64,
        crop_y: crop.y0 as f64 / height.max(1) as f64,
        crop_w: (crop.x1 - crop.x0) as f64 / width.max(1) as f64,
        crop_h: (crop.y1 - crop.y0) as f64 / height.max(1) as f64,
        recommended: projection,
        used,
        dropped: Vec::new(),
        kept: order,
        rotations: solved.rots.clone(),
        focal_px,
        focal35: 0.0,
        lens: *lens,
        photo: Photo { gains: photo.gains.clone(), coef: photo.coef.clone(), pedestal: photo.pedestal },
        names: Vec::new(),
    })
}

fn fit_photo(
    frames: &[InputFrame],
    indices: &[usize],
    rots: &[Matrix3<f64>],
    f: f64,
    lens: &LensModel,
    geom: &CanvasGeom,
) -> Result<(Photo, Vec<warp::Placed>), String> {
    let placed = place_frames(frames, indices, rots, f, lens, geom)?;
    let pairs = overlap_pairs(&placed);
    trace::gate()?;
    let photo = photo::solve_photometry(&placed, &pairs, 0);
    let corner = placed.first().map(|p| photo::illumination_corner(&photo, p)).unwrap_or(1.0);
    trace::line(&format!("brightness gains={} corner={corner:.3}", photo.gains.len()));
    Ok((photo, placed))
}

// Draws each photo onto the canvas once and keeps that drawing for the rest of the stitch.
fn place_frames(
    frames: &[InputFrame],
    indices: &[usize],
    rots: &[Matrix3<f64>],
    f: f64,
    lens: &LensModel,
    geom: &CanvasGeom,
) -> Result<Vec<warp::Placed>, String> {
    let log = trace::current();
    let placed: Vec<warp::Placed> = indices
        .par_iter()
        .enumerate()
        .map(|(k, &i)| {
            let _guard = log.as_ref().map(|log| trace::install(log.clone()));
            if trace::halted() {
                return warp::Placed { x0: 0, y0: 0, w: 0, h: 0, img: Vec::new(), valid: Vec::new(), uv: Vec::new() };
            }
            let frame = &frames[i];
            let mut rgb = frame.rgb.clone();
            warp::devignette(&mut rgb, frame.width, frame.height, lens);
            let warped = warp::warp_one(&rgb, frame.width, frame.height, &rots[k], f, lens, geom);
            trace::line(&format!(
                "warp {}: bbox=({},{},{},{}) {}x{}",
                frame.name,
                warped.x0,
                warped.y0,
                warped.x1(),
                warped.y1(),
                warped.w,
                warped.h
            ));
            warped
        })
        .collect();
    trace::gate()?;
    if placed.iter().any(|p| p.w == 0 || p.h == 0) && trace::halted() {
        return Err("stopped".into());
    }
    Ok(placed)
}

fn overlap_pairs(placed: &[warp::Placed]) -> Vec<(usize, usize)> {
    let areas: Vec<usize> = placed.iter().map(|p| p.valid.iter().filter(|v| **v).count()).collect();
    let mut pairs = Vec::new();
    for a in 0..placed.len() {
        for b in (a + 1)..placed.len() {
            let x0 = placed[a].x0.max(placed[b].x0);
            let y0 = placed[a].y0.max(placed[b].y0);
            let x1 = placed[a].x1().min(placed[b].x1());
            let y1 = placed[a].y1().min(placed[b].y1());
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let mut n = 0usize;
            for y in (y0..y1).step_by(8) {
                for x in (x0..x1).step_by(8) {
                    let ia = ((y - placed[a].y0) as usize) * placed[a].w + (x - placed[a].x0) as usize;
                    let ib = ((y - placed[b].y0) as usize) * placed[b].w + (x - placed[b].x0) as usize;
                    if placed[a].valid[ia] && placed[b].valid[ib] {
                        n += 1;
                    }
                }
            }
            let smaller = areas[a].min(areas[b]).max(1);
            if (n as f64) * 64.0 > 0.03 * smaller as f64 {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

fn composite(
    placed: &mut [warp::Placed],
    photo: &Photo,
    geom: &CanvasGeom,
    half: bool,
) -> Result<(Vec<f32>, Vec<u16>, u32, u32), String> {
    let w = geom.width as usize;
    let h = geom.height as usize;
    let mut acc = vec![0f32; w * h * 3];
    let mut accm = vec![false; w * h];
    let mut winners = vec![u16::MAX; w * h];
    let levels = if half { blend_levels(geom.width, geom.height, 5, 7) } else { blend_levels(geom.width, geom.height, 6, 8) };
    let pad = 8usize * (1usize << levels.min(8));
    for bi in 0..placed.len() {
        photo::apply_photometry(&mut placed[bi], photo, bi);
        placed[bi].uv = Vec::new();
    }
    for bi in 0..placed.len() {
        trace::gate()?;
        if bi == 0 {
            paste(&mut acc, &mut accm, &mut winners, &placed[bi], bi as u16, w);
            placed[bi].img = Vec::new();
            placed[bi].valid = Vec::new();
            continue;
        }
        let x0 = (placed[bi].x0 as usize).saturating_sub(pad);
        let y0 = (placed[bi].y0 as usize).saturating_sub(pad);
        let x1 = (placed[bi].x1() as usize + pad).min(w);
        let y1 = (placed[bi].y1() as usize + pad).min(h);
        if x1 <= x0 || y1 <= y0 {
            placed[bi].img = Vec::new();
            placed[bi].valid = Vec::new();
            super::local_align::commit_shift(x0 as i32, y0 as i32);
            continue;
        }
        let ww = x1 - x0;
        let hh = y1 - y0;
        let tiles = hh.saturating_sub(128) / 64 + 1;
        let tiles = tiles * (ww.saturating_sub(128) / 64 + 1);
        trace::line(&format!("blend {bi}: window={ww}x{hh} tiles={tiles}"));
        let (bw, bv, aw, av) = window(&placed[bi], &acc, &accm, x0, y0, x1, y1, w);
        let lum_a = plane_luma(&aw);
        let lum_b = plane_luma(&bw);
        trace::gate()?;
        let (bw, bv) = align_incoming(&lum_a, &lum_b, &av, bv, bw, ww, hh);
        super::local_align::commit_shift(x0 as i32, y0 as i32);
        trace::gate()?;
        let lum_b = plane_luma(&bw);
        let hard = graphcut_mask(&lum_a, &lum_b, &av, &bv, ww, hh);
        trace::gate()?;
        let mut keep_sum = 0.0f64;
        let mut keep_n = 0usize;
        for i in 0..ww * hh {
            if av[i] && bv[i] {
                keep_sum += hard[i] as f64;
                keep_n += 1;
            }
        }
        let keep_a = if keep_n == 0 { 1.0 } else { keep_sum / keep_n as f64 };
        let soft = blur_mask(&hard, ww, hh);
        trace::line("mask blurred");
        let (sx0, sy0, sx1, sy1) = blend_rect(&av, &bv, ww, hh, pad);
        let cw = sx1 - sx0;
        let ch = sy1 - sy0;
        let blended = if cw == ww && ch == hh {
            pyr_blend(&aw, &bw, &soft, &av, &bv, ww, hh, levels)
        } else {
            pyr_blend(
                &take_rect_rgb(&aw, ww, sx0, sy0, sx1, sy1),
                &take_rect_rgb(&bw, ww, sx0, sy0, sx1, sy1),
                &take_rect_f32(&soft, ww, sx0, sy0, sx1, sy1),
                &take_rect_bool(&av, ww, sx0, sy0, sx1, sy1),
                &take_rect_bool(&bv, ww, sx0, sy0, sx1, sy1),
                cw,
                ch,
                levels,
            )
        };
        for y in 0..hh {
            for x in 0..ww {
                let s = y * ww + x;
                if !(av[s] || bv[s]) {
                    continue;
                }
                let inside = x >= sx0 && x < sx1 && y >= sy0 && y < sy1;
                let (p0, p1, p2) = if inside {
                    let cs = (y - sy0) * cw + (x - sx0);
                    (blended[cs * 3], blended[cs * 3 + 1], blended[cs * 3 + 2])
                } else if bv[s] {
                    (bw[s * 3], bw[s * 3 + 1], bw[s * 3 + 2])
                } else {
                    continue;
                };
                let d = (y0 + y) * w + x0 + x;
                acc[d * 3] = p0;
                acc[d * 3 + 1] = p1;
                acc[d * 3 + 2] = p2;
                accm[d] = true;
                if bv[s] && (!av[s] || hard[s] < 0.5) {
                    winners[d] = bi as u16;
                }
            }
        }
        trace::line("pasted");
        trace::line(&format!("blended {bi} keepA={keep_a:.2}"));
        placed[bi].img = Vec::new();
        placed[bi].valid = Vec::new();
    }
    Ok((acc, winners, geom.width, geom.height))
}

fn paste(acc: &mut [f32], accm: &mut [bool], winners: &mut [u16], p: &warp::Placed, id: u16, canvas_w: usize) {
    for y in 0..p.h {
        for x in 0..p.w {
            if !p.valid[y * p.w + x] {
                continue;
            }
            let dx = (p.x0 + x as i32) as usize;
            let dy = (p.y0 + y as i32) as usize;
            let d = dy * canvas_w + dx;
            let s = y * p.w + x;
            acc[d * 3] = p.img[s * 3];
            acc[d * 3 + 1] = p.img[s * 3 + 1];
            acc[d * 3 + 2] = p.img[s * 3 + 2];
            accm[d] = true;
            winners[d] = id;
        }
    }
}

fn window(p: &warp::Placed, acc: &[f32], accm: &[bool], x0: usize, y0: usize, x1: usize, y1: usize, canvas_w: usize) -> (Vec<f32>, Vec<bool>, Vec<f32>, Vec<bool>) {
    let ww = x1 - x0;
    let hh = y1 - y0;
    let mut b = vec![0f32; ww * hh * 3];
    let mut bv = vec![false; ww * hh];
    let mut a = vec![0f32; ww * hh * 3];
    let mut av = vec![false; ww * hh];
    for y in 0..hh {
        let cy = y0 + y;
        for x in 0..ww {
            let cx = x0 + x;
            let di = y * ww + x;
            let ai = cy * canvas_w + cx;
            a[di * 3] = acc[ai * 3];
            a[di * 3 + 1] = acc[ai * 3 + 1];
            a[di * 3 + 2] = acc[ai * 3 + 2];
            av[di] = accm[ai];
            if cx >= p.x0 as usize && cy >= p.y0 as usize && (cx as i32) < p.x1() && (cy as i32) < p.y1() {
                let sx = cx - p.x0 as usize;
                let sy = cy - p.y0 as usize;
                let si = sy * p.w + sx;
                if p.valid[si] {
                    b[di * 3] = p.img[si * 3];
                    b[di * 3 + 1] = p.img[si * 3 + 1];
                    b[di * 3 + 2] = p.img[si * 3 + 2];
                    bv[di] = true;
                }
            }
        }
    }
    (b, bv, a, av)
}

fn plane_luma(rgb: &[f32]) -> Vec<f32> {
    let n = rgb.len() / 3;
    let mut o = vec![0f32; n];
    for i in 0..n {
        o[i] = photo::luma(&rgb[i * 3..i * 3 + 3]);
    }
    o
}

// Keeps the multiband blend on the overlap plus the same padding the full window already used.
fn blend_rect(av: &[bool], bv: &[bool], w: usize, h: usize, pad: usize) -> (usize, usize, usize, usize) {
    let mut x0 = w;
    let mut y0 = h;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            if av[row + x] && bv[row + x] {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return (0, 0, w, h);
    }
    let sx0 = x0.saturating_sub(pad);
    let sy0 = y0.saturating_sub(pad);
    let sx1 = (x1 + pad).min(w);
    let sy1 = (y1 + pad).min(h);
    if (sx1 - sx0) * (sy1 - sy0) * 10 > w * h * 9 {
        (0, 0, w, h)
    } else {
        (sx0, sy0, sx1, sy1)
    }
}

fn take_rect_rgb(src: &[f32], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<f32> {
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut o = vec![0f32; cw * ch * 3];
    for y in 0..ch {
        let s = ((y0 + y) * w + x0) * 3;
        let d = y * cw * 3;
        o[d..d + cw * 3].copy_from_slice(&src[s..s + cw * 3]);
    }
    o
}

fn take_rect_f32(src: &[f32], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<f32> {
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut o = vec![0f32; cw * ch];
    for y in 0..ch {
        let s = (y0 + y) * w + x0;
        let d = y * cw;
        o[d..d + cw].copy_from_slice(&src[s..s + cw]);
    }
    o
}

fn take_rect_bool(src: &[bool], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<bool> {
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut o = vec![false; cw * ch];
    for y in 0..ch {
        let s = (y0 + y) * w + x0;
        o[y * cw..(y + 1) * cw].copy_from_slice(&src[s..s + cw]);
    }
    o
}

fn detect_all(frames: &[InputFrame], progress: &dyn Fn(&str)) -> Result<Vec<Vec<Feature>>, String> {
    progress("Finding points...");
    let log = trace::current();
    let found: Vec<Vec<Feature>> = frames
        .par_iter()
        .map(|frame| {
            let _guard = log.as_ref().map(|log| trace::install(log.clone()));
            if trace::halted() {
                return Vec::new();
            }
            detect_one(frame)
        })
        .collect();
    trace::gate()?;
    for (i, frame) in frames.iter().enumerate() {
        trace::line(&format!("points {}/{} {}: {}", i + 1, frames.len(), frame.name, found[i].len()));
    }
    Ok(found)
}

pub fn find_points(frame: &InputFrame) -> Vec<Feature> {
    detect_one(frame)
}

fn detect_one(frame: &InputFrame) -> Vec<Feature> {
    let t = std::time::Instant::now();
    let gray = to_gray(&frame.rgb, frame.width, frame.height);
    trace::line(&format!("gray {:.2}s", t.elapsed().as_secs_f64()));
    let (dw, dh) = fit_long_side(frame.width, frame.height, POINT_LONG_SIDE);
    let small = shrink_gray(&gray, frame.width, frame.height, dw, dh);
    trace::line(&format!("search {dw}x{dh}"));
    let t = std::time::Instant::now();
    let mask = texture_mask(&small, dw as usize, dh as usize);
    trace::line(&format!("mask {:.2}s", t.elapsed().as_secs_f64()));
    let t = std::time::Instant::now();
    let eq = clahe::apply(&small, dw as usize, dh as usize, 2.5, 8, 8);
    trace::line(&format!("clahe {:.2}s", t.elapsed().as_secs_f64()));
    let mut found = sift::detect(&eq, &mask, dw as usize, dh as usize);
    let sx = frame.width as f64 / dw.max(1) as f64;
    let sy = frame.height as f64 / dh.max(1) as f64;
    for feature in &mut found {
        feature.pt[0] *= sx;
        feature.pt[1] *= sy;
    }
    found
}

fn shrink_gray(src: &[u8], w: u32, h: u32, dw: u32, dh: u32) -> Vec<u8> {
    if dw == w && dh == h {
        return src.to_vec();
    }
    let img = image::GrayImage::from_raw(w, h, src.to_vec()).unwrap_or_else(|| image::GrayImage::new(w, h));
    image::imageops::resize(&img, dw, dh, image::imageops::FilterType::Triangle).into_raw()
}

fn to_gray(rgb: &[f32], w: u32, h: u32) -> Vec<u8> {
    let buf = rgb.to_vec();
    let img = Rgb32FImage::from_raw(w, h, buf).unwrap_or_else(|| Rgb32FImage::new(w, h));
    let srgb = crate::image_processing::apply_linear_to_srgb(DynamicImage::ImageRgb32F(img));
    let rgb = srgb.to_rgb32f();
    let (wr, wg, wb) = super::const_::gray_weights();
    let mut g = vec![0u8; (w * h) as usize];
    for (i, px) in rgb.pixels().enumerate() {
        let y = wr * px[0] + wg * px[1] + wb * px[2];
        g[i] = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    g
}

fn texture_mask(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut mag = vec![0f32; w * h];
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let at = |yy: i32, xx: i32| gray[edge_reflect(yy, h as i32) as usize * w + edge_reflect(xx, w as i32) as usize] as f32;
    for y in 0..h {
        for x in 0..w {
            let yy = y as i32;
            let xx = x as i32;
            let gx = -at(yy - 1, xx - 1) + at(yy - 1, xx + 1) - 2.0 * at(yy, xx - 1) + 2.0 * at(yy, xx + 1) - at(yy + 1, xx - 1)
                + at(yy + 1, xx + 1);
            let gy = -at(yy - 1, xx - 1) - 2.0 * at(yy - 1, xx) - at(yy - 1, xx + 1) + at(yy + 1, xx - 1) + 2.0 * at(yy + 1, xx)
                + at(yy + 1, xx + 1);
            mag[y * w + x] = gx.abs() + gy.abs();
        }
    }
    let mag = blur_plane(&mag, w, h, 3.0);
    if mag.is_empty() {
        return Vec::new();
    }
    let thr = percentile_linear(&mag, 0.55);
    mag.iter().map(|v| if *v > thr { 255 } else { 0 }).collect()
}

fn edge_reflect(mut p: i32, len: i32) -> i32 {
    if len <= 1 {
        return 0;
    }
    while p < 0 || p >= len {
        if p < 0 {
            p = -p;
        } else {
            p = 2 * len - p - 2;
        }
    }
    p
}

// Same rank a percentile uses, without moving the picture's pixels.
fn percentile_linear(values: &[f32], q: f64) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }
    let mut copy = values.to_vec();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    copy.select_nth_unstable_by(hi, cmp);
    let hi_v = copy[hi];
    let lo_v = if lo == hi {
        hi_v
    } else {
        copy[..hi].select_nth_unstable_by(lo, cmp);
        copy[lo]
    };
    let frac = (pos - lo as f64) as f32;
    lo_v * (1.0 - frac) + hi_v * frac
}

fn blur_plane(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let mut k = (sigma * 8.0 + 1.0).round() as i32;
    k |= 1;
    if k < 1 {
        k = 1;
    }
    let r = k / 2;
    let mut ker = vec![0f32; k as usize];
    let mut sum = 0.0f32;
    for i in 0..k {
        let x = (i - r) as f32;
        let v = (-0.5 * (x / sigma).powi(2)).exp();
        ker[i as usize] = v;
        sum += v;
    }
    for v in &mut ker {
        *v /= sum;
    }
    sift::blur_with_kernel(src, w, h, &ker, sift::BlurEdge::Reflect)
}

fn score_inliers(pairs: &[super::match_::PairMatches], _feats: &[Vec<Feature>], f: f64, w: u32, h: u32, lens: &LensModel) -> usize {
    let mut tot = 0usize;
    for p in pairs {
        if p.pa.len() < 12 {
            continue;
        }
        let ba = geom::bearings(&p.pa, f, w, h, lens);
        let bb = geom::bearings(&p.pb, f, w, h, lens);
        if let Some((_, inl, _, _)) = geom::ransac_rotation(&ba, &bb, f, &p.pa, w, h, super::const_::RANSAC_ITERS_FOCAL) {
            tot += inl;
        }
    }
    tot
}
