//! Content-preserving local mesh warp (APAP / Zhang–Liu / NISwGSP family).
//! After global SO(3)+focal poses, pull automatic CPs into coincidence
//! with a source-space mesh so overlaps can be seamed cleanly.
//!
//! Important: drive the mesh from **flow matches**, not rotation-RANSAC inliers.
//! Rotation inliers already fit a global rotation — they cannot explain parallax
//! that causes visible ridge/horizon steps at seams.

use crate::panorama_utils::camera::CameraPose;
use crate::panorama_utils::match_graph::PairEdge;
use crate::panorama_utils::orb::OrbFeature;
use nalgebra::{DMatrix, DVector};
use std::collections::HashSet;

const GRID_W: usize = 20;
const GRID_H: usize = 14;
/// Extra weight when a CP needs vertical correction (ridges / horizon steps).
const VERTICAL_CP_WEIGHT: f64 = 5.0;
const SMOOTH_WEIGHT: f64 = 2.0;
const IDENTITY_WEIGHT: f64 = 0.25;
/// Allow enough local contortion to absorb ~p90 transfer residuals.
const MAX_DISP_FRAC: f64 = 0.08;
/// Ignore gross outliers (wrong matches) when scattering mesh samples.
const MAX_CP_RESIDUAL_PX: f64 = 90.0;

#[derive(Debug, Clone)]
pub struct ImageMesh {
    pub width: u32,
    pub height: u32,
    pub gw: usize,
    pub gh: usize,
    /// Per-vertex source displacement (du, dv) so sample(u,v) reads (u+du, v+dv).
    pub du: Vec<f64>,
    pub dv: Vec<f64>,
}

impl ImageMesh {
    pub fn identity(width: u32, height: u32) -> Self {
        let n = GRID_W * GRID_H;
        Self {
            width,
            height,
            gw: GRID_W,
            gh: GRID_H,
            du: vec![0.0; n],
            dv: vec![0.0; n],
        }
    }

    pub fn scaled(&self, new_w: u32, new_h: u32) -> Self {
        let sx = new_w as f64 / self.width.max(1) as f64;
        let sy = new_h as f64 / self.height.max(1) as f64;
        Self {
            width: new_w,
            height: new_h,
            gw: self.gw,
            gh: self.gh,
            du: self.du.iter().map(|d| d * sx).collect(),
            dv: self.dv.iter().map(|d| d * sy).collect(),
        }
    }

    #[inline]
    fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.gw + ix
    }

    /// Bilinear sample of displacement at source pixel (u, v).
    pub fn displacement(&self, u: f64, v: f64) -> (f64, f64) {
        if self.du.iter().all(|d| d.abs() < 1e-9) && self.dv.iter().all(|d| d.abs() < 1e-9) {
            return (0.0, 0.0);
        }
        let w = self.width.max(1) as f64;
        let h = self.height.max(1) as f64;
        let fx = (u / w * (self.gw - 1) as f64).clamp(0.0, (self.gw - 1) as f64);
        let fy = (v / h * (self.gh - 1) as f64).clamp(0.0, (self.gh - 1) as f64);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.gw - 1);
        let y1 = (y0 + 1).min(self.gh - 1);
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let sample = |arr: &[f64]| {
            let a = arr[self.idx(x0, y0)];
            let b = arr[self.idx(x1, y0)];
            let c = arr[self.idx(x0, y1)];
            let d = arr[self.idx(x1, y1)];
            a * (1.0 - tx) * (1.0 - ty)
                + b * tx * (1.0 - ty)
                + c * (1.0 - tx) * ty
                + d * tx * ty
        };
        (sample(&self.du), sample(&self.dv))
    }

    pub fn warp_uv(&self, u: f64, v: f64) -> (f64, f64) {
        let (du, dv) = self.displacement(u, v);
        (u + du, v + dv)
    }
}

/// Build one mesh per camera index (identity for unused). Adjacent-pair CPs drive displacements.
pub fn build_local_meshes(
    poses: &[CameraPose],
    features: &[Vec<OrbFeature>],
    edges: &[PairEdge],
    kept: &[usize],
) -> Vec<ImageMesh> {
    let n = poses.len();
    let mut meshes: Vec<ImageMesh> = (0..n)
        .map(|i| ImageMesh::identity(poses[i].width, poses[i].height))
        .collect();
    let kept_set: HashSet<usize> = kept.iter().copied().collect();

    let mut acc: Vec<Vec<(f64, f64, f64)>> = meshes
        .iter()
        .map(|m| vec![(0.0, 0.0, 0.0); m.gw * m.gh])
        .collect();

    let mut pre_errs: Vec<f64> = Vec::new();
    let mut used_cps = 0usize;

    for e in edges {
        if !kept_set.contains(&e.i) || !kept_set.contains(&e.j) {
            continue;
        }
        // Adjacent only — long-range pairs fight the local overlap corridor.
        if e.i.abs_diff(e.j) != 1 {
            continue;
        }
        let pairs = if e.flow_pairs.len() >= 12 {
            e.flow_pairs.as_slice()
        } else {
            e.inliers.as_slice()
        };
        // Spatially subsample dense flow so one textured patch cannot dominate.
        let step = (pairs.len() / 400).max(1);
        for (k, &(ia, ib)) in pairs.iter().enumerate() {
            if k % step != 0 {
                continue;
            }
            if ia >= features[e.i].len() || ib >= features[e.j].len() {
                continue;
            }
            let ka = features[e.i][ia].keypoint;
            let kb = features[e.j][ib].keypoint;
            let ua = ka.x as f64;
            let va = ka.y as f64;
            let ub = kb.x as f64;
            let vb = kb.y as f64;

            let wb = poses[e.j].world_bearing_from_pixel(ub, vb);
            let wa = poses[e.i].world_bearing_from_pixel(ua, va);

            // Meet in the middle: each image absorbs half the transfer residual.
            if let Some((tu, tv)) = poses[e.i].pixel_from_world_bearing(wb) {
                let du = ua - tu;
                let dv = va - tv;
                let err = (du * du + dv * dv).sqrt();
                pre_errs.push(err);
                if err <= MAX_CP_RESIDUAL_PX {
                    let w = 1.0 + VERTICAL_CP_WEIGHT * (dv.abs() / 5.0).min(10.0);
                    scatter_disp(&mut acc[e.i], &meshes[e.i], tu, tv, 0.5 * du, 0.5 * dv, w);
                    used_cps += 1;
                }
            }
            if let Some((tu, tv)) = poses[e.j].pixel_from_world_bearing(wa) {
                let du = ub - tu;
                let dv = vb - tv;
                let err = (du * du + dv * dv).sqrt();
                if err <= MAX_CP_RESIDUAL_PX {
                    let w = 1.0 + VERTICAL_CP_WEIGHT * (dv.abs() / 5.0).min(10.0);
                    scatter_disp(&mut acc[e.j], &meshes[e.j], tu, tv, 0.5 * du, 0.5 * dv, w);
                    used_cps += 1;
                }
            }
        }
    }

    pre_errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pre_med = pre_errs
        .get(pre_errs.len() / 2)
        .copied()
        .unwrap_or(0.0);
    let pre_p90 = pre_errs
        .get(pre_errs.len().saturating_mul(9) / 10)
        .copied()
        .unwrap_or(pre_med);
    crate::panorama_utils::debug_log::write(&format!(
        "local_mesh cps_used={} pre_transfer median={:.2}px p90={:.2}px",
        used_cps, pre_med, pre_p90
    ));

    for &idx in kept {
        solve_mesh(&mut meshes[idx], &acc[idx]);
        let max_d = (poses[idx].width.min(poses[idx].height) as f64) * MAX_DISP_FRAC;
        for k in 0..meshes[idx].du.len() {
            meshes[idx].du[k] = meshes[idx].du[k].clamp(-max_d, max_d);
            meshes[idx].dv[k] = meshes[idx].dv[k].clamp(-max_d, max_d);
        }
        let mean = meshes[idx]
            .du
            .iter()
            .zip(meshes[idx].dv.iter())
            .map(|(a, b)| (a * a + b * b).sqrt())
            .sum::<f64>()
            / meshes[idx].du.len().max(1) as f64;
        let max_v = meshes[idx]
            .du
            .iter()
            .zip(meshes[idx].dv.iter())
            .map(|(a, b)| (a * a + b * b).sqrt())
            .fold(0.0f64, f64::max);
        log::info!(
            "local_mesh[{}] mean_disp={:.2}px max_disp={:.2}px cap={:.1}",
            idx,
            mean,
            max_v,
            max_d
        );
        crate::panorama_utils::debug_log::write(&format!(
            "local_mesh[{}] mean_disp={:.2}px max_disp={:.2}px",
            idx, mean, max_v
        ));
    }

    // Post-mesh residual on adjacent flow CPs (should drop vs pre_transfer).
    let mut post_errs: Vec<f64> = Vec::new();
    for e in edges {
        if !kept_set.contains(&e.i) || !kept_set.contains(&e.j) || e.i.abs_diff(e.j) != 1 {
            continue;
        }
        let pairs = if e.flow_pairs.len() >= 12 {
            e.flow_pairs.as_slice()
        } else {
            e.inliers.as_slice()
        };
        let step = (pairs.len() / 200).max(1);
        for (k, &(ia, ib)) in pairs.iter().enumerate() {
            if k % step != 0 {
                continue;
            }
            if ia >= features[e.i].len() || ib >= features[e.j].len() {
                continue;
            }
            let ka = features[e.i][ia].keypoint;
            let kb = features[e.j][ib].keypoint;
            let wa = poses[e.i].world_bearing_from_pixel(ka.x as f64, ka.y as f64);
            if let Some((tu, tv)) = poses[e.j].pixel_from_world_bearing(wa) {
                let (wu, wv) = meshes[e.j].warp_uv(tu, tv);
                let dx = wu - kb.x as f64;
                let dy = wv - kb.y as f64;
                post_errs.push((dx * dx + dy * dy).sqrt());
            }
        }
    }
    post_errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if !post_errs.is_empty() {
        let med = post_errs[post_errs.len() / 2];
        let p90 = post_errs[post_errs.len().saturating_mul(9) / 10];
        crate::panorama_utils::debug_log::write(&format!(
            "local_mesh post_transfer median={:.2}px p90={:.2}px",
            med, p90
        ));
    }

    meshes
}

fn scatter_disp(
    acc: &mut [(f64, f64, f64)],
    mesh: &ImageMesh,
    u: f64,
    v: f64,
    du: f64,
    dv: f64,
    w: f64,
) {
    let width = mesh.width.max(1) as f64;
    let height = mesh.height.max(1) as f64;
    let fx = (u / width * (mesh.gw - 1) as f64).clamp(0.0, (mesh.gw - 1) as f64);
    let fy = (v / height * (mesh.gh - 1) as f64).clamp(0.0, (mesh.gh - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(mesh.gw - 1);
    let y1 = (y0 + 1).min(mesh.gh - 1);
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;
    let corners = [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x1, y0, tx * (1.0 - ty)),
        (x0, y1, (1.0 - tx) * ty),
        (x1, y1, tx * ty),
    ];
    for (ix, iy, a) in corners {
        let i = iy * mesh.gw + ix;
        let ww = w * a;
        acc[i].0 += ww * du;
        acc[i].1 += ww * dv;
        acc[i].2 += ww;
    }
}

/// Solve (data + smoothness + identity) for vertex displacements.
fn solve_mesh(mesh: &mut ImageMesh, acc: &[(f64, f64, f64)]) {
    let n = mesh.gw * mesh.gh;
    for (comp, out) in [(0usize, &mut mesh.du), (1usize, &mut mesh.dv)] {
        let mut a = DMatrix::<f64>::zeros(n, n);
        let mut b = DVector::<f64>::zeros(n);
        for iy in 0..mesh.gh {
            for ix in 0..mesh.gw {
                let i = iy * mesh.gw + ix;
                let (sdu, sdv, sw) = acc[i];
                let data = if sw > 1e-6 {
                    if comp == 0 {
                        sdu / sw
                    } else {
                        sdv / sw
                    }
                } else {
                    0.0
                };
                // Vertices with real CP evidence get strong data pull; empty stay near 0.
                let data_w = if sw > 1e-6 {
                    sw + IDENTITY_WEIGHT
                } else {
                    IDENTITY_WEIGHT * 4.0
                };
                a[(i, i)] += data_w;
                b[i] += data_w * data;

                let neighbors = [
                    (ix.wrapping_sub(1), iy),
                    (ix + 1, iy),
                    (ix, iy.wrapping_sub(1)),
                    (ix, iy + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx >= mesh.gw || ny >= mesh.gh {
                        continue;
                    }
                    let j = ny * mesh.gw + nx;
                    a[(i, i)] += SMOOTH_WEIGHT;
                    a[(i, j)] -= SMOOTH_WEIGHT;
                }
            }
        }
        match a.lu().solve(&b) {
            Some(sol) => {
                for i in 0..n {
                    out[i] = sol[i];
                }
            }
            None => {
                for i in 0..n {
                    out[i] = if acc[i].2 > 1e-6 {
                        if comp == 0 {
                            acc[i].0 / acc[i].2
                        } else {
                            acc[i].1 / acc[i].2
                        }
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_mesh_no_warp() {
        let m = ImageMesh::identity(100, 80);
        let (u, v) = m.warp_uv(40.0, 30.0);
        assert!((u - 40.0).abs() < 1e-9);
        assert!((v - 30.0).abs() < 1e-9);
    }

    #[test]
    fn scatter_and_solve_moves_toward_cp() {
        let mut mesh = ImageMesh::identity(100, 100);
        let mut acc = vec![(0.0, 0.0, 0.0); mesh.gw * mesh.gh];
        scatter_disp(&mut acc, &mesh, 50.0, 50.0, 5.0, 0.0, 10.0);
        solve_mesh(&mut mesh, &acc);
        let (du, dv) = mesh.displacement(50.0, 50.0);
        assert!(du > 1.0, "du={du}");
        assert!(dv.abs() < 1.0, "dv={dv}");
    }
}
