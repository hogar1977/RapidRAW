use image::{GrayImage, ImageBuffer, Luma};
use imageproc::corners::{Corner, corners_fast9};
use imageproc::filter::gaussian_blur_f32;
use nalgebra::Point2;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

pub const ORB_DESCRIPTOR_SIZE: usize = 256;
pub type OrbDescriptor = [u8; ORB_DESCRIPTOR_SIZE / 8];

const FAST_THRESHOLD: u8 = 12;
const NMS_RADIUS: f32 = 6.0;
const BRIEF_PATCH: u32 = 31;
const MAX_FEATURES: usize = 4000;
const MATCH_RATIO: f32 = 0.85;

#[derive(Debug, Clone, Copy)]
pub struct OrbKeyPoint {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
}

#[derive(Debug, Clone)]
pub struct OrbFeature {
    pub keypoint: OrbKeyPoint,
    pub descriptor: OrbDescriptor,
}

#[derive(Debug, Clone, Copy)]
pub struct OrbMatch {
    pub index1: usize,
    pub index2: usize,
}

pub fn generate_steered_brief_pairs() -> Vec<(Point2<i32>, Point2<i32>)> {
    let mut rng = StdRng::seed_from_u64(42);
    let half = BRIEF_PATCH as i32 / 2;
    let dist = rand::distr::Uniform::new(-half, half).expect("uniform");
    (0..ORB_DESCRIPTOR_SIZE)
        .map(|_| {
            (
                Point2::new(dist.sample(&mut rng), dist.sample(&mut rng)),
                Point2::new(dist.sample(&mut rng), dist.sample(&mut rng)),
            )
        })
        .collect()
}

pub fn detect_and_compute(
    gray: &GrayImage,
    pairs: &[(Point2<i32>, Point2<i32>)],
) -> Vec<OrbFeature> {
    let blurred = imageproc::filter::gaussian_blur_f32(gray, 1.2);
    let mut corners = corners_fast9(&blurred, FAST_THRESHOLD);
    corners.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<Corner> = Vec::new();
    let r2 = NMS_RADIUS * NMS_RADIUS;
    for c in corners {
        if kept.iter().any(|k| {
            let dx = k.x as f32 - c.x as f32;
            let dy = k.y as f32 - c.y as f32;
            dx * dx + dy * dy < r2
        }) {
            continue;
        }
        kept.push(c);
        if kept.len() >= MAX_FEATURES {
            break;
        }
    }

    let gray_f = ImageBuffer::from_fn(gray.width(), gray.height(), |x, y| {
        Luma([gray.get_pixel(x, y)[0] as f32])
    });
    let gray_f = gaussian_blur_f32(&gray_f, 2.0);

    kept.par_iter()
        .filter_map(|c| {
            let angle = intensity_centroid_angle(&gray_f, c.x, c.y, BRIEF_PATCH)?;
            let desc = steered_brief(&gray_f, c.x as f32, c.y as f32, angle, pairs)?;
            Some(OrbFeature {
                keypoint: OrbKeyPoint {
                    x: c.x as f32,
                    y: c.y as f32,
                    angle,
                },
                descriptor: desc,
            })
        })
        .collect()
}

fn intensity_centroid_angle(
    img: &ImageBuffer<Luma<f32>, Vec<f32>>,
    x: u32,
    y: u32,
    patch: u32,
) -> Option<f32> {
    let half = patch / 2;
    let (w, h) = img.dimensions();
    if x < half || y < half || x + half >= w || y + half >= h {
        return None;
    }
    let mut m01 = 0.0f32;
    let mut m10 = 0.0f32;
    for dy in -(half as i32)..=(half as i32) {
        for dx in -(half as i32)..=(half as i32) {
            let v = img.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0];
            m10 += dx as f32 * v;
            m01 += dy as f32 * v;
        }
    }
    Some(m01.atan2(m10))
}

fn steered_brief(
    img: &ImageBuffer<Luma<f32>, Vec<f32>>,
    x: f32,
    y: f32,
    angle: f32,
    pairs: &[(Point2<i32>, Point2<i32>)],
) -> Option<OrbDescriptor> {
    let (w, h) = img.dimensions();
    let half = BRIEF_PATCH as f32 / 2.0;
    if x < half || y < half || x >= w as f32 - half || y >= h as f32 - half {
        return None;
    }
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let mut descriptor = [0u8; ORB_DESCRIPTOR_SIZE / 8];
    for (i, pair) in pairs.iter().enumerate() {
        let p1x = (pair.0.x as f32 * cos_a - pair.0.y as f32 * sin_a).round() as i32;
        let p1y = (pair.0.x as f32 * sin_a + pair.0.y as f32 * cos_a).round() as i32;
        let p2x = (pair.1.x as f32 * cos_a - pair.1.y as f32 * sin_a).round() as i32;
        let p2y = (pair.1.x as f32 * sin_a + pair.1.y as f32 * cos_a).round() as i32;
        let x1 = (x as i32 + p1x).clamp(0, w as i32 - 1) as u32;
        let y1 = (y as i32 + p1y).clamp(0, h as i32 - 1) as u32;
        let x2 = (x as i32 + p2x).clamp(0, w as i32 - 1) as u32;
        let y2 = (y as i32 + p2y).clamp(0, h as i32 - 1) as u32;
        if img.get_pixel(x1, y1)[0] < img.get_pixel(x2, y2)[0] {
            descriptor[i / 8] |= 1 << (i % 8);
        }
    }
    Some(descriptor)
}

fn hamming(a: &OrbDescriptor, b: &OrbDescriptor) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

fn match_orb_directed(features1: &[OrbFeature], features2: &[OrbFeature]) -> Vec<OrbMatch> {
    if features1.is_empty() || features2.is_empty() {
        return Vec::new();
    }
    features1
        .par_iter()
        .enumerate()
        .filter_map(|(i, f1)| {
            let mut best = u32::MAX;
            let mut second = u32::MAX;
            let mut best_j = 0usize;
            for (j, f2) in features2.iter().enumerate() {
                let d = hamming(&f1.descriptor, &f2.descriptor);
                if d < best {
                    second = best;
                    best = d;
                    best_j = j;
                } else if d < second {
                    second = d;
                }
            }
            if second > 0 && (best as f32 / second as f32) < MATCH_RATIO {
                Some(OrbMatch {
                    index1: i,
                    index2: best_j,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Symmetric Lowe-ratio matches (mutual nearest neighbors).
pub fn match_orb(features1: &[OrbFeature], features2: &[OrbFeature]) -> Vec<OrbMatch> {
    let forward = match_orb_directed(features1, features2);
    if forward.is_empty() {
        return forward;
    }
    let backward = match_orb_directed(features2, features1);
    let mut back_map = vec![None; features2.len()];
    for m in &backward {
        if m.index1 < back_map.len() {
            back_map[m.index1] = Some(m.index2);
        }
    }
    forward
        .into_iter()
        .filter(|m| back_map.get(m.index2).and_then(|x| *x) == Some(m.index1))
        .collect()
}
