use crate::panorama_utils::camera::CameraPose;
use crate::panorama_utils::local_warp::ImageMesh;
use crate::panorama_utils::projection::ProjectionCanvas;
use crate::panorama_utils::stitching::get_interpolated_pixel;
use image::{GrayImage, Rgb, Rgb32FImage};
use rayon::prelude::*;

pub struct BlendResult {
    pub image: Rgb32FImage,
    pub mask: GrayImage,
    pub winners: Vec<u16>,
}

/// Block-gain compensation → DP seams → hard winner-takes-all composite (linear light).
/// Soft multi-band mixing is intentionally avoided: misaligned overlaps must not ghost.
pub fn blend_panorama(
    images: &[&Rgb32FImage],
    _masks: &[&GrayImage],
    poses: &[CameraPose],
    indices: &[usize],
    canvas: &ProjectionCanvas,
    meshes: Option<&[ImageMesh]>,
    mut on_progress: Option<&mut dyn FnMut(f32, &str)>,
) -> BlendResult {
    let w = canvas.width;
    let h = canvas.height;
    let n = indices.len();
    let mut pano = Rgb32FImage::new(w, h);
    let mut mask = GrayImage::new(w, h);
    let mut winners = vec![u16::MAX; (w as usize) * (h as usize)];

    if images.is_empty() || n == 0 {
        return BlendResult {
            image: pano,
            mask,
            winners,
        };
    }

    if let Some(cb) = on_progress.as_mut() {
        cb(0.0, "Matching exposure...");
    }
    let (gains, rgb_scales) = estimate_block_gains(images, poses, indices, canvas, meshes);
    log::info!(
        "Panorama block gains (mean): {:?} rgb={:?}",
        gains
            .iter()
            .map(|g| format!("{:.3}", g.iter().sum::<f64>() / g.len().max(1) as f64))
            .collect::<Vec<_>>(),
        rgb_scales
            .iter()
            .map(|c| format!("[{:.2},{:.2},{:.2}]", c[0], c[1], c[2]))
            .collect::<Vec<_>>()
    );
    if let Some(cb) = on_progress.as_mut() {
        cb(0.12, "Finding seams...");
    }

    // Coarse seam grid: full-res Dijkstra was ~40s/pair and looked "stuck".
    // Step≈3–4px still allows PTGui-like zigzags after upsampling labels.
    let seam_step = ((w.min(h) / 280).max(3)) as usize;
    let sw = ((w as usize + seam_step - 1) / seam_step).max(1);
    let sh = ((h as usize + seam_step - 1) / seam_step).max(1);
    crate::panorama_utils::debug_log::write(&format!(
        "seam_grid {}x{} step={}",
        sw, sh, seam_step
    ));
    let mut cover: Vec<Vec<u8>> = (0..n).map(|_| vec![0u8; sw * sh]).collect();
    let mut sample_luma: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; sw * sh]).collect();

    for sy in 0..sh {
        for sx in 0..sw {
            let x = (sx * seam_step) as u32;
            let y = (sy * seam_step) as u32;
            if x >= w || y >= h {
                continue;
            }
            let idx = sy * sw + sx;
            for (local_i, &cam_idx) in indices.iter().enumerate() {
                let Some((u, v)) =
                    sample_uv_warped(poses, cam_idx, canvas, meshes, x as f64, y as f64)
                else {
                    continue;
                };
                let img = images[local_i];
                let (iw, ih) = img.dimensions();
                if u < 1.0 || v < 1.0 || u >= iw as f64 - 2.0 || v >= ih as f64 - 2.0 {
                    continue;
                }
                if source_edge_weight(u, v, iw, ih) < 0.05 {
                    continue;
                }
                let g = sample_gain(&gains[local_i], u, v, iw, ih) as f32;
                let rs = rgb_scales[local_i];
                let c = get_interpolated_pixel(img, u, v);
                cover[local_i][idx] = 1;
                sample_luma[local_i][idx] = (luminance(Rgb([
                    c[0] * g * rs[0] as f32,
                    c[1] * g * rs[1] as f32,
                    c[2] * g * rs[2] as f32,
                ])) as f32)
                    .max(0.0);
            }
        }
    }

    let seam_labels = compute_seam_labels(n, sw, sh, &cover, &sample_luma);
    if let Some(cb) = on_progress.as_mut() {
        cb(0.22, "Compositing (hard seams)...");
    }

    let chunk = ((h as usize) / 24).max(8);
    let mut y0 = 0u32;
    while y0 < h {
        let y1 = (y0 + chunk as u32).min(h);
        let row_results: Vec<(u32, Vec<f32>, Vec<u8>, Vec<u16>)> = (y0..y1)
            .into_par_iter()
            .map(|y| {
                let mut row_rgb = vec![0.0f32; w as usize * 3];
                let mut row_mask = vec![0u8; w as usize];
                let mut row_win = vec![u16::MAX; w as usize];

                for x in 0..w {
                    let sx = ((x as usize) / seam_step).min(sw - 1);
                    let sy = ((y as usize) / seam_step).min(sh - 1);
                    let preferred = seam_labels[sy * sw + sx];

                    let sample_cam = |local_i: usize| -> Option<(f32, Rgb<f32>)> {
                        let cam_idx = indices[local_i];
                        let (u, v) =
                            sample_uv_warped(poses, cam_idx, canvas, meshes, x as f64, y as f64)?;
                        let img = images[local_i];
                        let (iw, ih) = img.dimensions();
                        if u < 1.0 || v < 1.0 || u >= iw as f64 - 2.0 || v >= ih as f64 - 2.0 {
                            return None;
                        }
                        let edge = source_edge_weight(u, v, iw, ih);
                        if edge <= 1e-6 {
                            return None;
                        }
                        let g = sample_gain(&gains[local_i], u, v, iw, ih) as f32;
                        let rs = rgb_scales[local_i];
                        let mut c = get_interpolated_pixel(img, u, v);
                        c[0] *= g * rs[0] as f32;
                        c[1] *= g * rs[1] as f32;
                        c[2] *= g * rs[2] as f32;
                        Some((edge as f32, c))
                    };

                    // Hard winner-takes-all (no feather): geometry is good; feather looked like ghosting.
                    let mut best: Option<(u16, f32, Rgb<f32>)> = None;
                    if preferred != u16::MAX {
                        let pi = preferred as usize;
                        if pi < n {
                            if let Some((edge, c)) = sample_cam(pi) {
                                best = Some((preferred, edge, c));
                            }
                        }
                    }
                    if best.is_none() {
                        for local_i in 0..n {
                            let Some((edge, c)) = sample_cam(local_i) else {
                                continue;
                            };
                            match &best {
                                Some((_, bs, _)) if *bs >= edge => {}
                                _ => best = Some((local_i as u16, edge, c)),
                            }
                        }
                    }
                    let Some((winner, _, color)) = best else {
                        continue;
                    };
                    row_win[x as usize] = winner;
                    row_mask[x as usize] = 255;
                    let base = x as usize * 3;
                    row_rgb[base] = color[0];
                    row_rgb[base + 1] = color[1];
                    row_rgb[base + 2] = color[2];
                }

                (y, row_rgb, row_mask, row_win)
            })
            .collect();

        for (y, row_rgb, row_mask, row_win) in row_results {
            for x in 0..w {
                let xi = x as usize;
                if row_mask[xi] == 0 {
                    continue;
                }
                let base = xi * 3;
                pano.put_pixel(
                    x,
                    y,
                    Rgb([row_rgb[base], row_rgb[base + 1], row_rgb[base + 2]]),
                );
                mask.put_pixel(x, y, image::Luma([255]));
                winners[(y as usize) * (w as usize) + xi] = row_win[xi];
            }
        }

        let frac = (y1 as f32 / h as f32).clamp(0.0, 1.0);
        if let Some(cb) = on_progress.as_mut() {
            cb(
                0.22 + 0.78 * frac,
                &format!("Compositing {}%", (frac * 100.0) as u32),
            );
        }
        y0 = y1;
    }

    BlendResult {
        image: pano,
        mask,
        winners,
    }
}

/// PTGui-style irregular seams: row-DP path per adjacent pair (prefer high-texture
/// agreement), then simultaneous 1×N labeling so later pairs cannot overwrite zigzags.
fn compute_seam_labels(
    n: usize,
    sw: usize,
    sh: usize,
    cover: &[Vec<u8>],
    luma: &[Vec<f32>],
) -> Vec<u16> {
    let mut labels = vec![u16::MAX; sw * sh];
    if n == 0 || sw == 0 || sh == 0 {
        return labels;
    }

    // paths[pair][y] = seam x between cam pair and pair+1
    let mut paths: Vec<Vec<Option<usize>>> = (0..n.saturating_sub(1))
        .map(|_| vec![None; sh])
        .collect();

    for a in 0..n.saturating_sub(1) {
        let b = a + 1;
        let mut x_min = sw;
        let mut x_max = 0usize;
        let mut y_min = sh;
        let mut y_max = 0usize;
        let mut any = false;
        for y in 0..sh {
            for x in 0..sw {
                let i = y * sw + x;
                if cover[a][i] != 0 && cover[b][i] != 0 {
                    any = true;
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
        }
        if !any || x_min > x_max || y_min > y_max {
            continue;
        }
        x_min = x_min.saturating_sub(1);
        x_max = (x_max + 1).min(sw - 1);
        let bw = x_max - x_min + 1;
        let bh = y_max - y_min + 1;
        if bw < 2 || bh < 2 {
            continue;
        }

        let grad = |cam: usize, y: usize, x: usize| -> f32 {
            let i = y * sw + x;
            let mut g = 0.0f32;
            if x > 0 && x + 1 < sw {
                g += (luma[cam][i - 1] - luma[cam][i + 1]).abs();
            }
            if y > 0 && y + 1 < sh {
                g += (luma[cam][i - sw] - luma[cam][i + sw]).abs();
            }
            g
        };

        // PTGui-like: cut where images agree, prefer high-texture corridors (trees/clouds).
        let cell_cost = |y: usize, x: usize| -> f32 {
            let i = y * sw + x;
            let ca = cover[a][i] != 0;
            let cb = cover[b][i] != 0;
            if ca && cb {
                let d = (luma[a][i] - luma[b][i]).abs();
                let tex = 0.5 * (grad(a, y, x) + grad(b, y, x));
                // Mismatch is expensive; agreeing high-texture is cheapest (hides the cut).
                let mismatch = d * d * 28.0 + d * tex * 10.0;
                let hide = 0.55 * tex / (1.0 + 25.0 * d);
                (0.08 + mismatch - hide).max(0.002)
            } else if ca || cb {
                10.0
            } else {
                1e4
            }
        };

        // Row DP: O(bh*bw*max_dx) — stays responsive on preview canvases.
        let max_dx = ((bw / 2).max(8)).min(64).min(bw.saturating_sub(1).max(1));
        let mut dp = vec![f32::INFINITY; bh * bw];
        let mut parent = vec![0usize; bh * bw];
        for lx in 0..bw {
            let x = x_min + lx;
            let i = y_min * sw + x;
            if cover[a][i] == 0 && cover[b][i] == 0 {
                continue;
            }
            // Only seed from true overlap so the path starts in the corridor.
            if cover[a][i] == 0 || cover[b][i] == 0 {
                continue;
            }
            dp[lx] = cell_cost(y_min, x);
            parent[lx] = lx;
        }
        // If top row has no overlap seeds, seed mid.
        if !dp.iter().any(|c| c.is_finite()) {
            let mid = bw / 2;
            dp[mid] = cell_cost(y_min, x_min + mid);
            parent[mid] = mid;
        }
        for ly in 1..bh {
            let y = y_min + ly;
            for lx in 0..bw {
                let x = x_min + lx;
                let i = y * sw + x;
                if cover[a][i] == 0 || cover[b][i] == 0 {
                    continue;
                }
                let base = cell_cost(y, x);
                let mut best = f32::INFINITY;
                let mut best_px = lx;
                let lo = lx.saturating_sub(max_dx);
                let hi = (lx + max_dx).min(bw - 1);
                for px in lo..=hi {
                    let prev = dp[(ly - 1) * bw + px];
                    if !prev.is_finite() {
                        continue;
                    }
                    let turn = (px as i32 - lx as i32).unsigned_abs() as f32 * 0.004;
                    let c = prev + turn + base;
                    if c < best {
                        best = c;
                        best_px = px;
                    }
                }
                if best.is_finite() {
                    dp[ly * bw + lx] = best;
                    parent[ly * bw + lx] = best_px;
                }
            }
        }

        let mut best_end = None;
        let mut best_c = f32::INFINITY;
        for lx in 0..bw {
            let c = dp[(bh - 1) * bw + lx];
            if c < best_c {
                best_c = c;
                best_end = Some(lx);
            }
        }
        if best_end.is_none() || !best_c.is_finite() {
            // Fallback: vertical mid-overlap (should be rare).
            let mid = (x_min + x_max) / 2;
            for y in y_min..=y_max {
                paths[a][y] = Some(mid);
            }
            crate::panorama_utils::debug_log::write(&format!(
                "seam {}→{} DP failed; fallback x={}",
                a, b, mid
            ));
            continue;
        }
        let mut cx = best_end.unwrap();

        let mut path_x_for_y = vec![None; sh];
        path_x_for_y[y_min + bh - 1] = Some(x_min + cx);
        for ly in (1..bh).rev() {
            cx = parent[ly * bw + cx];
            path_x_for_y[y_min + ly - 1] = Some(x_min + cx);
        }
        let mut last = (x_min + x_max) / 2;
        for y in 0..sh {
            if let Some(x) = path_x_for_y[y] {
                last = x;
            } else if y >= y_min && y <= y_max {
                path_x_for_y[y] = Some(last);
            }
        }
        paths[a] = path_x_for_y;

        let xs: Vec<f32> = paths[a].iter().filter_map(|x| x.map(|v| v as f32)).collect();
        if xs.len() >= 2 {
            let mean = xs.iter().sum::<f32>() / xs.len() as f32;
            let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / xs.len() as f32;
            crate::panorama_utils::debug_log::write(&format!(
                "seam {}→{} x_std={:.1} (grid) span=[{},{}]",
                a,
                b,
                var.sqrt(),
                x_min,
                x_max
            ));
        }
    }

    // Simultaneous labeling: respect all seam paths at once (1×N left→right).
    for y in 0..sh {
        for x in 0..sw {
            let i = y * sw + x;
            let mut covering: Vec<usize> = Vec::new();
            for cam in 0..n {
                if cover[cam][i] != 0 {
                    covering.push(cam);
                }
            }
            if covering.is_empty() {
                continue;
            }
            if covering.len() == 1 {
                labels[i] = covering[0] as u16;
                continue;
            }
            // Ideal camera index from seam positions.
            let mut ideal = 0usize;
            for p in 0..paths.len() {
                if let Some(sx) = paths[p][y] {
                    if x > sx {
                        ideal = p + 1;
                    }
                }
            }
            let chosen = covering
                .iter()
                .copied()
                .min_by_key(|&c| (c as i32 - ideal as i32).unsigned_abs())
                .unwrap_or(covering[0]);
            labels[i] = chosen as u16;
        }
    }
    labels
}

#[cfg(test)]
mod seam_tests {
    use super::compute_seam_labels;

    #[test]
    fn zigzag_seam_snakes_through_agreeing_corridor() {
        let n = 2;
        let sw = 8;
        let sh = 6;
        let mut cover = vec![vec![0u8; sw * sh]; n];
        let mut luma = vec![vec![0.0f32; sw * sh]; n];
        for y in 0..sh {
            for x in 0..sw {
                let i = y * sw + x;
                if x <= 5 {
                    cover[0][i] = 1;
                    luma[0][i] = 0.5;
                }
                if x >= 2 {
                    cover[1][i] = 1;
                    luma[1][i] = 0.5;
                }
            }
            let good = 2 + y / 2;
            for x in 2..=5 {
                let i = y * sw + x;
                if x == good {
                    luma[0][i] = 0.4;
                    luma[1][i] = 0.4;
                } else {
                    luma[0][i] = 0.2;
                    luma[1][i] = 0.9;
                }
            }
        }
        let labels = compute_seam_labels(n, sw, sh, &cover, &luma);
        let mut seam_xs = Vec::new();
        for y in 0..sh {
            let mut sx = None;
            for x in 0..sw - 1 {
                let i = y * sw + x;
                let j = y * sw + x + 1;
                if labels[i] == 0 && labels[j] == 1 {
                    sx = Some(x);
                    break;
                }
            }
            seam_xs.push(sx.expect("expected a→b transition"));
        }
        let uniq: std::collections::HashSet<_> = seam_xs.iter().copied().collect();
        assert!(
            uniq.len() >= 2,
            "expected zigzag (got vertical {:?})",
            seam_xs
        );
    }
}


fn sample_uv(
    poses: &[CameraPose],
    cam_idx: usize,
    canvas: &ProjectionCanvas,
    x: f64,
    y: f64,
) -> Option<(f64, f64)> {
    let ray = canvas.pixel_to_ray(x, y);
    poses[cam_idx].pixel_from_world_bearing(ray)
}

fn sample_uv_warped(
    poses: &[CameraPose],
    cam_idx: usize,
    canvas: &ProjectionCanvas,
    meshes: Option<&[ImageMesh]>,
    x: f64,
    y: f64,
) -> Option<(f64, f64)> {
    let (u, v) = sample_uv(poses, cam_idx, canvas, x, y)?;
    if let Some(ms) = meshes {
        if let Some(m) = ms.get(cam_idx) {
            return Some(m.warp_uv(u, v));
        }
    }
    Some((u, v))
}

fn source_edge_weight(sx: f64, sy: f64, iw: u32, ih: u32) -> f64 {
    let bx = sx.min(iw as f64 - 1.0 - sx).max(0.0);
    let by = sy.min(ih as f64 - 1.0 - sy).max(0.0);
    let border = bx.min(by);
    let short = iw.min(ih) as f64;
    let margin = (short * 0.14).clamp(48.0, 700.0);
    let t = (border / margin).clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    s * s
}

fn luminance(c: Rgb<f32>) -> f64 {
    (0.2126 * c[0] as f64) + (0.7152 * c[1] as f64) + (0.0722 * c[2] as f64)
}

const BLOCK: u32 = 32;

/// Per-image RGB exposure from overlaps — no block maps (those caused visible tiling).
fn estimate_block_gains(
    images: &[&Rgb32FImage],
    poses: &[CameraPose],
    indices: &[usize],
    canvas: &ProjectionCanvas,
    meshes: Option<&[ImageMesh]>,
) -> (Vec<Vec<f64>>, Vec<[f64; 3]>) {
    let n = indices.len();
    let mut maps: Vec<Vec<f64>> = Vec::with_capacity(n);
    for img in images {
        let bw = ((img.width() + BLOCK - 1) / BLOCK).max(1);
        let bh = ((img.height() + BLOCK - 1) / BLOCK).max(1);
        // Uniform per image — block variation was the "mesh" artifact in the sky.
        maps.push(vec![1.0f64; (bw * bh) as usize]);
    }
    let mut rgb = vec![[1.0f64; 3]; n];

    let w = canvas.width;
    let h = canvas.height;
    let step = ((w.min(h) / 280).max(2)) as u32;

    #[derive(Clone)]
    struct OvSamp {
        a: usize,
        b: usize,
        ca: [f64; 3],
        cb: [f64; 3],
    }
    let mut samples: Vec<OvSamp> = Vec::new();
    for a in 0..n.saturating_sub(1) {
        let b = a + 1;
        for y in (0..h).step_by(step as usize) {
            for x in (0..w).step_by(step as usize) {
                let Some((ua, va)) =
                    sample_uv_warped(poses, indices[a], canvas, meshes, x as f64, y as f64)
                else {
                    continue;
                };
                let Some((ub, vb)) =
                    sample_uv_warped(poses, indices[b], canvas, meshes, x as f64, y as f64)
                else {
                    continue;
                };
                let ia = images[a];
                let ib = images[b];
                let (awa, aha) = ia.dimensions();
                let (bwa, bha) = ib.dimensions();
                if ua < 1.0 || va < 1.0 || ua >= awa as f64 - 2.0 || va >= aha as f64 - 2.0 {
                    continue;
                }
                if ub < 1.0 || vb < 1.0 || ub >= bwa as f64 - 2.0 || vb >= bha as f64 - 2.0 {
                    continue;
                }
                if source_edge_weight(ua, va, awa, aha) < 0.06
                    || source_edge_weight(ub, vb, bwa, bha) < 0.06
                {
                    continue;
                }
                let pa = get_interpolated_pixel(ia, ua, va);
                let pb = get_interpolated_pixel(ib, ub, vb);
                let ca = [pa[0] as f64, pa[1] as f64, pa[2] as f64];
                let cb = [pb[0] as f64, pb[1] as f64, pb[2] as f64];
                if ca.iter().sum::<f64>() < 1e-4 || cb.iter().sum::<f64>() < 1e-4 {
                    continue;
                }
                samples.push(OvSamp { a, b, ca, cb });
            }
        }
    }

    // Pass A: chain adjacent median luma ratios (robust brightness match).
    {
        let mut scalar = vec![1.0f64; n];
        for a in 0..n.saturating_sub(1) {
            let b = a + 1;
            let mut ratios: Vec<f64> = Vec::new();
            for s in &samples {
                if s.a != a || s.b != b {
                    continue;
                }
                let la = luminance(Rgb([
                    (s.ca[0] * scalar[a]) as f32,
                    (s.ca[1] * scalar[a]) as f32,
                    (s.ca[2] * scalar[a]) as f32,
                ]));
                let lb = luminance(Rgb([
                    (s.cb[0] * scalar[b]) as f32,
                    (s.cb[1] * scalar[b]) as f32,
                    (s.cb[2] * scalar[b]) as f32,
                ]));
                if la > 1e-5 && lb > 1e-5 {
                    ratios.push(la / lb);
                }
            }
            if ratios.len() < 12 {
                continue;
            }
            ratios.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
            let med = ratios[ratios.len() / 2].clamp(0.5, 2.0);
            scalar[b] = (scalar[b] * med).clamp(0.35, 2.8);
        }
        let mean_s = scalar.iter().sum::<f64>() / n.max(1) as f64;
        if mean_s > 1e-8 {
            for s in scalar.iter_mut() {
                *s /= mean_s;
            }
        }
        for i in 0..n {
            for k in 0..3 {
                rgb[i][k] = scalar[i];
            }
        }
    }

    // Pass B: iterative per-channel refinement on overlaps.
    for _ in 0..16 {
        let mut num = vec![[0.0f64; 3]; n];
        let mut den = vec![[0.0f64; 3]; n];
        for s in &samples {
            for k in 0..3 {
                let ra = s.ca[k] * rgb[s.a][k];
                let rb = s.cb[k] * rgb[s.b][k];
                if ra > 1e-5 && rb > 1e-5 {
                    let target = (ra * rb).sqrt();
                    num[s.a][k] += target / s.ca[k].max(1e-8);
                    den[s.a][k] += 1.0;
                    num[s.b][k] += target / s.cb[k].max(1e-8);
                    den[s.b][k] += 1.0;
                }
            }
        }
        for i in 0..n {
            for k in 0..3 {
                if den[i][k] >= 8.0 {
                    let g = (num[i][k] / den[i][k]).clamp(0.35, 2.8);
                    rgb[i][k] = (0.35 * rgb[i][k] + 0.65 * g).clamp(0.35, 2.8);
                }
            }
        }
    }

    let mut mean = [0.0f64; 3];
    for i in 0..n {
        for k in 0..3 {
            mean[k] += rgb[i][k];
        }
    }
    for k in 0..3 {
        mean[k] /= n.max(1) as f64;
        if mean[k] > 1e-8 {
            for i in 0..n {
                rgb[i][k] /= mean[k];
            }
        }
    }

    // Fold into uniform block maps (still 1.0) + rgb scales.
    for i in 0..n {
        for v in maps[i].iter_mut() {
            *v = 1.0;
        }
    }

    crate::panorama_utils::debug_log::write(&format!(
        "exposure_rgb: {}",
        rgb.iter()
            .map(|c| format!("[{:.3},{:.3},{:.3}]", c[0], c[1], c[2]))
            .collect::<Vec<_>>()
            .join(" ")
    ));
    crate::panorama_utils::debug_log::write(&format!(
        "exposure_scalar: {}",
        rgb.iter()
            .map(|c| format!("{:.3}", 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]))
            .collect::<Vec<_>>()
            .join(", ")
    ));

    (maps, rgb)
}


fn sample_gain(map: &[f64], u: f64, v: f64, iw: u32, ih: u32) -> f64 {
    let bw = ((iw + BLOCK - 1) / BLOCK).max(1);
    let bh = ((ih + BLOCK - 1) / BLOCK).max(1);
    let bx = ((u as u32) / BLOCK).min(bw - 1);
    let by = ((v as u32) / BLOCK).min(bh - 1);
    map[(by * bw + bx) as usize]
}
