use crate::panorama_utils::session::NormalizedCrop;
use image::GrayImage;

pub fn max_inscribed_aabb(mask: &GrayImage) -> NormalizedCrop {
    let w = mask.width() as usize;
    let h = mask.height() as usize;
    if w == 0 || h == 0 {
        return NormalizedCrop::default();
    }

    let mut heights = vec![0usize; w];
    let mut best_area = 0usize;
    let mut best = (0usize, 0usize, w, h);

    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x as u32, y as u32)[0] > 0 {
                heights[x] += 1;
            } else {
                heights[x] = 0;
            }
        }
        if let Some((left, right, height)) = largest_histogram_rect(&heights) {
            let area = (right - left) * height;
            if area > best_area {
                best_area = area;
                let top = y + 1 - height;
                best = (left, top, right - left, height);
            }
        }
    }

    if best_area == 0 {
        return NormalizedCrop::default();
    }

    NormalizedCrop {
        x: best.0 as f64 / w as f64,
        y: best.1 as f64 / h as f64,
        width: best.2 as f64 / w as f64,
        height: best.3 as f64 / h as f64,
    }
}

fn largest_histogram_rect(heights: &[usize]) -> Option<(usize, usize, usize)> {
    let mut stack: Vec<usize> = Vec::new();
    let mut best_area = 0usize;
    let mut best = None;
    let n = heights.len();

    for i in 0..=n {
        let h = if i < n { heights[i] } else { 0 };
        while let Some(&top) = stack.last() {
            if heights[top] <= h {
                break;
            }
            stack.pop();
            let height = heights[top];
            let left = stack.last().map(|v| v + 1).unwrap_or(0);
            let right = i;
            let area = height * (right - left);
            if area > best_area {
                best_area = area;
                best = Some((left, right, height));
            }
        }
        stack.push(i);
    }
    best
}
