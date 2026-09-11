use crate::panorama_utils::camera::{
    estimate_relative_rotation_ransac, filter_panorama_flow_matches, relative_yaw_pitch_roll,
    CameraPose,
};
use crate::panorama_utils::orb::{match_orb, OrbFeature};
use image::GrayImage;
use nalgebra::{Rotation3, Vector3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

const MIN_MATCHES: usize = 8;
const MIN_ROT_INLIERS: usize = 6;
const MIN_ROT_INLIERS_FAR: usize = 12;
const MIN_ROT_INLIERS_VERY_FAR: usize = 18;
const MAX_REL_ROLL_ADJ_DEG: f64 = 20.0;
const MAX_REL_ROLL_FAR_DEG: f64 = 14.0;
const MAX_REL_PITCH_ADJ_DEG: f64 = 42.0;
const MAX_REL_PITCH_FAR_DEG: f64 = 50.0;

#[derive(Debug, Clone)]
pub struct PairEdge {
    pub i: usize,
    pub j: usize,
    pub relative_rotation: Rotation3<f64>,
    /// Rotation-RANSAC inliers (global pose).
    pub inliers: Vec<(usize, usize)>,
    /// Flow-filtered matches (local mesh / parallax) — denser than inliers.
    pub flow_pairs: Vec<(usize, usize)>,
    pub weight: usize,
}

#[derive(Debug, Clone)]
pub struct MatchGraphResult {
    pub edges: Vec<PairEdge>,
    pub kept: Vec<usize>,
    pub dropped: Vec<(usize, String)>,
    pub initial_rotations: HashMap<usize, Rotation3<f64>>,
    /// When true, GBA should use only |i-j|==1 edges (ordered 1×N capture).
    pub adjacent_only_gba: bool,
    pub diagnostics: String,
}

struct Dsu {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }
    fn find(&mut self, i: usize) -> usize {
        if self.parent[i] != i {
            self.parent[i] = self.find(self.parent[i]);
        }
        self.parent[i]
    }
    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

fn short_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn hop(i: usize, j: usize) -> usize {
    i.abs_diff(j)
}

fn relative_motion_ok(r: &Rotation3<f64>, hop_dist: usize) -> bool {
    let (_yaw, pitch, roll) = relative_yaw_pitch_roll(r);
    let pitch_deg = pitch.to_degrees().abs();
    let roll_deg = roll.to_degrees().abs();
    if hop_dist <= 1 {
        roll_deg <= MAX_REL_ROLL_ADJ_DEG && pitch_deg <= MAX_REL_PITCH_ADJ_DEG
    } else {
        roll_deg <= MAX_REL_ROLL_FAR_DEG && pitch_deg <= MAX_REL_PITCH_FAR_DEG
    }
}

fn min_inliers_for_hop(hop_dist: usize) -> usize {
    match hop_dist {
        0 | 1 => MIN_ROT_INLIERS,
        2 | 3 => MIN_ROT_INLIERS_FAR,
        _ => MIN_ROT_INLIERS_VERY_FAR,
    }
}

/// Celeste-class sky/cloud rejection: bright, low-gradient, upper-frame prior.
fn is_likely_sky_cp(gray: &GrayImage, x: f64, y: f64) -> bool {
    let w = gray.width() as i32;
    let h = gray.height() as i32;
    let xi = x.round() as i32;
    let yi = y.round() as i32;
    if xi < 2 || yi < 2 || xi >= w - 2 || yi >= h - 2 {
        return false;
    }
    let upper = y < h as f64 * 0.42;
    if !upper {
        return false;
    }
    let mut sum = 0u32;
    let mut sum_abs = 0u32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let p = gray.get_pixel((xi + dx) as u32, (yi + dy) as u32)[0] as u32;
            sum += p;
            if dx != 0 || dy != 0 {
                let c = gray.get_pixel(xi as u32, yi as u32)[0] as i32;
                sum_abs += (p as i32 - c).unsigned_abs();
            }
        }
    }
    let mean = sum as f64 / 9.0;
    let grad = sum_abs as f64 / 8.0;
    mean > 175.0 && grad < 12.0
}

fn reject_sky_matches(
    matches: &[crate::panorama_utils::orb::OrbMatch],
    feats_a: &[OrbFeature],
    feats_b: &[OrbFeature],
    gray_a: Option<&GrayImage>,
    gray_b: Option<&GrayImage>,
) -> Vec<crate::panorama_utils::orb::OrbMatch> {
    let (Some(ga), Some(gb)) = (gray_a, gray_b) else {
        return matches.to_vec();
    };
    matches
        .iter()
        .copied()
        .filter(|m| {
            let ka = feats_a[m.index1].keypoint;
            let kb = feats_b[m.index2].keypoint;
            !(is_likely_sky_cp(ga, ka.x as f64, ka.y as f64)
                && is_likely_sky_cp(gb, kb.x as f64, kb.y as f64))
        })
        .collect()
}

/// Prefer spatially spread inliers (grid bucket greedily by residual order).
fn spread_inliers(
    pts_a: &[(f64, f64)],
    inlier_idx: &[usize],
    img_w: u32,
    img_h: u32,
    max_keep: usize,
) -> Vec<usize> {
    if inlier_idx.len() <= max_keep {
        return inlier_idx.to_vec();
    }
    let gw = 6usize;
    let gh = 4usize;
    let cell_w = (img_w as f64 / gw as f64).max(1.0);
    let cell_h = (img_h as f64 / gh as f64).max(1.0);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); gw * gh];
    for &k in inlier_idx {
        let (x, y) = pts_a[k];
        let cx = ((x / cell_w) as usize).min(gw - 1);
        let cy = ((y / cell_h) as usize).min(gh - 1);
        buckets[cy * gw + cx].push(k);
    }
    let mut out = Vec::new();
    let mut round = 0usize;
    while out.len() < max_keep {
        let mut added = false;
        for b in &mut buckets {
            if round < b.len() {
                out.push(b[round]);
                added = true;
                if out.len() >= max_keep {
                    break;
                }
            }
        }
        if !added {
            break;
        }
        round += 1;
    }
    out
}

pub fn build_match_graph(
    features: &[Vec<OrbFeature>],
    poses: &[CameraPose],
    filenames: &[String],
    feature_grays: Option<&[GrayImage]>,
) -> MatchGraphResult {
    let n = features.len();
    let mut edges = Vec::new();
    let mut diag_lines: Vec<String> = Vec::new();

    for (i, feats) in features.iter().enumerate() {
        diag_lines.push(format!(
            "features[{}={}]={}",
            i,
            short_name(&filenames[i]),
            feats.len()
        ));
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let hop_dist = hop(i, j);
            // No hard hop cap: multi-row grids need vertical neighbors at hop≈row_width.
            // Far pairs simply need more inliers.
            let raw_matches = match_orb(&features[i], &features[j]);
            let gray_a = feature_grays.and_then(|g| g.get(i));
            let gray_b = feature_grays.and_then(|g| g.get(j));
            let matches = reject_sky_matches(
                &raw_matches,
                &features[i],
                &features[j],
                gray_a,
                gray_b,
            );
            let pair_label = format!(
                "{} <-> {} (hop={})",
                short_name(&filenames[i]),
                short_name(&filenames[j]),
                hop_dist
            );
            if matches.len() < MIN_MATCHES {
                diag_lines.push(format!(
                    "pair {}: raw_matches={} after_sky={} (need >={}) -> skipped",
                    pair_label,
                    raw_matches.len(),
                    matches.len(),
                    MIN_MATCHES
                ));
                continue;
            }

            let pts_a_all: Vec<(f64, f64)> = matches
                .iter()
                .map(|m| {
                    let kp = features[i][m.index1].keypoint;
                    (kp.x as f64, kp.y as f64)
                })
                .collect();
            let pts_b_all: Vec<(f64, f64)> = matches
                .iter()
                .map(|m| {
                    let kp = features[j][m.index2].keypoint;
                    (kp.x as f64, kp.y as f64)
                })
                .collect();

            let flow_idx = filter_panorama_flow_matches(&pts_a_all, &pts_b_all);
            let pts_a: Vec<(f64, f64)> = flow_idx.iter().map(|&k| pts_a_all[k]).collect();
            let pts_b: Vec<(f64, f64)> = flow_idx.iter().map(|&k| pts_b_all[k]).collect();
            let filtered_matches: Vec<_> = flow_idx.iter().map(|&k| matches[k]).collect();

            if filtered_matches.len() < MIN_MATCHES {
                diag_lines.push(format!(
                    "pair {}: raw_matches={} after_flow={} -> skipped",
                    pair_label,
                    matches.len(),
                    filtered_matches.len()
                ));
                continue;
            }

            let min_inliers = min_inliers_for_hop(hop_dist);

            match estimate_relative_rotation_ransac(&pts_a, &pts_b, &poses[i], &poses[j]) {
                Some((r_ij, inlier_idx)) if inlier_idx.len() >= min_inliers => {
                    if !relative_motion_ok(&r_ij, hop_dist) {
                        let (yaw, pitch, roll) = relative_yaw_pitch_roll(&r_ij);
                        diag_lines.push(format!(
                            "pair {}: inliers={} but motion yaw={:.1}° pitch={:.1}° roll={:.1}° -> rejected",
                            pair_label,
                            inlier_idx.len(),
                            yaw.to_degrees(),
                            pitch.to_degrees(),
                            roll.to_degrees()
                        ));
                        continue;
                    }
                    let spread_cap = if hop_dist <= 1 { 80 } else { 48 };
                    let inlier_idx = spread_inliers(
                        &pts_a,
                        &inlier_idx,
                        poses[i].width,
                        poses[i].height,
                        spread_cap,
                    );
                    diag_lines.push(format!(
                        "pair {}: raw_matches={} flow={} rotation_inliers={} -> connected",
                        pair_label,
                        raw_matches.len(),
                        filtered_matches.len(),
                        inlier_idx.len()
                    ));
                    let inliers: Vec<(usize, usize)> = inlier_idx
                        .iter()
                        .map(|&k| {
                            let m = filtered_matches[k];
                            (m.index1, m.index2)
                        })
                        .collect();
                    let flow_pairs: Vec<(usize, usize)> = filtered_matches
                        .iter()
                        .map(|m| (m.index1, m.index2))
                        .collect();
                    edges.push(PairEdge {
                        i,
                        j,
                        relative_rotation: r_ij,
                        inliers,
                        flow_pairs,
                        weight: inlier_idx.len(),
                    });
                }
                Some((_, inlier_idx)) => {
                    diag_lines.push(format!(
                        "pair {}: raw_matches={} rotation_inliers={} (need >={}) -> rejected",
                        pair_label,
                        matches.len(),
                        inlier_idx.len(),
                        min_inliers
                    ));
                }
                None => {
                    diag_lines.push(format!(
                        "pair {}: raw_matches={} rotation_ransac=failed -> rejected",
                        pair_label,
                        matches.len()
                    ));
                }
            }
        }
    }

    let mut dsu = Dsu::new(n);
    for e in &edges {
        dsu.union(e.i, e.j);
    }

    let mut component_sizes: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        *component_sizes.entry(dsu.find(i)).or_default() += 1;
    }
    let largest_root = component_sizes
        .iter()
        .max_by_key(|(_, s)| *s)
        .map(|(r, _)| *r);

    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for i in 0..n {
        let root = dsu.find(i);
        let has_edge = edges.iter().any(|e| e.i == i || e.j == i);
        if Some(root) == largest_root
            && has_edge
            && component_sizes.get(&root).copied().unwrap_or(0) >= 2
        {
            kept.push(i);
        } else {
            let reason = if !has_edge {
                "too_few_matches".to_string()
            } else {
                "not_in_largest_component".to_string()
            };
            dropped.push((i, reason));
        }
    }

    diag_lines.push(format!(
        "graph: edges={} kept={} dropped={}",
        edges.len(),
        kept.len(),
        dropped.len()
    ));

    if kept.len() < 2 {
        let diagnostics = diag_lines.join("\n");
        log::info!("Panorama match graph:\n{}", diagnostics);
        return MatchGraphResult {
            edges,
            kept: vec![],
            dropped: filenames
                .iter()
                .enumerate()
                .map(|(i, f)| (i, format!("could_not_connect: {}", short_name(f))))
                .collect(),
            initial_rotations: HashMap::new(),
            adjacent_only_gba: false,
            diagnostics,
        };
    }

    let kept_set: HashSet<usize> = kept.iter().copied().collect();
    let sequential = has_sequential_backbone(&kept, &edges);
    diag_lines.push(format!(
        "sequential_backbone={} (adjacent-only GBA when true)",
        sequential
    ));
    let diagnostics = diag_lines.join("\n");
    log::info!("Panorama match graph:\n{}", diagnostics);

    let initial_rotations = if sequential {
        init_sequential_yaw_chain(&kept, &edges).unwrap_or_else(|| {
            init_mst_rotations(n, &kept, &kept_set, &edges)
        })
    } else {
        init_mst_rotations(n, &kept, &kept_set, &edges)
    };

    MatchGraphResult {
        edges,
        kept,
        dropped,
        initial_rotations,
        adjacent_only_gba: sequential,
        diagnostics,
    }
}

fn has_sequential_backbone(kept: &[usize], edges: &[PairEdge]) -> bool {
    if kept.len() < 3 {
        return false;
    }
    let mut order = kept.to_vec();
    order.sort_unstable();
    let mut need = 0usize;
    let mut have = 0usize;
    for w in order.windows(2) {
        if w[1] != w[0] + 1 {
            return false;
        }
        need += 1;
        let a = w[0];
        let b = w[1];
        let Some(edge) = edges.iter().find(|e| {
            (e.i == a && e.j == b) || (e.i == b && e.j == a)
        }) else {
            continue;
        };
        let r = if edge.i == a {
            edge.relative_rotation
        } else {
            edge.relative_rotation.inverse()
        };
        // Camera motion a→b.
        let delta = r.inverse();
        let (yaw, pitch, roll) = relative_yaw_pitch_roll(&delta);
        // Row-wrap / vertical neighbors have large pitch or tiny yaw — not a 1×N pan.
        if pitch.to_degrees().abs() > 12.0 {
            return false;
        }
        if roll.to_degrees().abs() > 20.0 {
            return false;
        }
        if yaw.to_degrees().abs() < 1.5 {
            return false;
        }
        have += 1;
    }
    need > 0 && have * 3 >= need * 2
}

/// Relative yaw of camera `b` w.r.t. `a` from an edge (radians).
fn adjacent_pair_dyaw(a: usize, b: usize, edges: &[PairEdge]) -> Option<(f64, f64, f64)> {
    let edge = edges.iter().find(|e| {
        (e.i == a && e.j == b) || (e.i == b && e.j == a)
    })?;
    // r maps bearings of e.i → e.j; camera motion a→b uses r.inverse() when edge is a→b.
    let r = if edge.i == a {
        edge.relative_rotation
    } else {
        edge.relative_rotation.inverse()
    };
    let delta = r.inverse();
    let (dyaw, dpitch, droll) = relative_yaw_pitch_roll(&delta);
    Some((dyaw, dpitch, droll))
}

/// Build poses as a pure yaw chain from consecutive edges — ignores bogus long-range links.
/// Forces a single pan direction (majority vote) so cumulative yaws stay strictly monotonic.
fn init_sequential_yaw_chain(
    kept: &[usize],
    edges: &[PairEdge],
) -> Option<HashMap<usize, Rotation3<f64>>> {
    let mut order = kept.to_vec();
    order.sort_unstable();
    let mut raw_dyaws = Vec::with_capacity(order.len().saturating_sub(1));
    for w in order.windows(2) {
        let (dyaw, dpitch, droll) = adjacent_pair_dyaw(w[0], w[1], edges)?;
        if dpitch.to_degrees().abs() > 20.0 || droll.to_degrees().abs() > 25.0 {
            return None;
        }
        if dyaw.abs().to_degrees() < 0.25 {
            return None;
        }
        raw_dyaws.push(dyaw);
    }
    if raw_dyaws.is_empty() {
        return None;
    }

    let pos = raw_dyaws.iter().filter(|&&d| d > 0.0).count();
    let neg = raw_dyaws.iter().filter(|&&d| d < 0.0).count();
    let sign = if pos >= neg { 1.0 } else { -1.0 };
    let min_step = 0.5f64.to_radians();

    let mut rots = HashMap::new();
    let mut yaw_acc = 0.0f64;
    rots.insert(order[0], Rotation3::identity());
    for (i, &raw) in raw_dyaws.iter().enumerate() {
        let mut step = raw.abs() * sign;
        if step.abs() < min_step {
            step = min_step * sign;
        }
        yaw_acc += step;
        let b = order[i + 1];
        let r_b = Rotation3::from_axis_angle(&Vector3::y_axis(), yaw_acc);
        rots.insert(b, r_b);
        log::info!(
            "yaw-chain {}→{} raw={:.2}° forced={:.2}° acc={:.2}°",
            order[i],
            b,
            raw.to_degrees(),
            step.to_degrees(),
            yaw_acc.to_degrees()
        );
        crate::panorama_utils::debug_log::write(&format!(
            "yaw-chain {}→{} raw={:.3}° forced={:.3}° acc={:.3}° sign={}",
            order[i],
            b,
            raw.to_degrees(),
            step.to_degrees(),
            yaw_acc.to_degrees(),
            sign
        ));
    }
    Some(rots)
}

fn init_mst_rotations(
    n: usize,
    kept: &[usize],
    kept_set: &HashSet<usize>,
    edges: &[PairEdge],
) -> HashMap<usize, Rotation3<f64>> {
    let mut mst_edges: Vec<&PairEdge> = edges
        .iter()
        .filter(|e| kept_set.contains(&e.i) && kept_set.contains(&e.j))
        .collect();
    mst_edges.sort_by(|a, b| {
        let ha = hop(a.i, a.j);
        let hb = hop(b.i, b.j);
        ha.cmp(&hb)
            .then_with(|| b.weight.cmp(&a.weight))
    });

    let mut mst_dsu = Dsu::new(n);
    let mut adj: HashMap<usize, Vec<(usize, Rotation3<f64>, bool)>> = HashMap::new();
    let mut used = 0usize;
    for e in mst_edges {
        if mst_dsu.find(e.i) != mst_dsu.find(e.j) {
            mst_dsu.union(e.i, e.j);
            adj.entry(e.i)
                .or_default()
                .push((e.j, e.relative_rotation, true));
            adj.entry(e.j)
                .or_default()
                .push((e.i, e.relative_rotation, false));
            used += 1;
            if used + 1 >= kept.len() {
                break;
            }
        }
    }

    let start = kept[0];
    let mut initial_rotations = HashMap::new();
    let mut q = VecDeque::new();
    q.push_back((start, Rotation3::identity()));
    let mut visited = HashSet::new();
    visited.insert(start);

    while let Some((u, r_u)) = q.pop_front() {
        initial_rotations.insert(u, r_u);
        if let Some(neighbors) = adj.get(&u) {
            for &(v, r_edge, forward) in neighbors {
                if visited.contains(&v) {
                    continue;
                }
                visited.insert(v);
                let r_v = if forward {
                    r_u * r_edge.inverse()
                } else {
                    r_u * r_edge
                };
                q.push_back((v, r_v));
            }
        }
    }
    initial_rotations
}
