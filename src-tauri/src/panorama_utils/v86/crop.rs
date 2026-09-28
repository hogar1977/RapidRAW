use super::const_::CROP_DS;

pub struct CropPx {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

// Finds the largest rectangle that is completely filled by the panorama.
pub fn inscribed_crop(cover: &[bool], w: usize, h: usize) -> CropPx {
    if w == 0 || h == 0 {
        return CropPx { x0: 0, y0: 0, x1: w as u32, y1: h as u32 };
    }
    let sw = (w + CROP_DS - 1) / CROP_DS;
    let sh = (h + CROP_DS - 1) / CROP_DS;
    let mut mask = vec![false; sw * sh];
    for y in 0..sh {
        let sy = (y * CROP_DS).min(h - 1);
        let row = sy * w;
        for x in 0..sw {
            let sx = (x * CROP_DS).min(w - 1);
            mask[y * sw + x] = cover[row + sx];
        }
    }
    let it = (4usize).max(w.min(h) / 300);
    let coarse = (it + CROP_DS - 1) / CROP_DS;
    for _ in 0..coarse {
        mask = erode(&mask, sw, sh);
    }
    let (mut x0, mut y0, mut x1, mut y1) = largest_rect(&mask, sw, sh);
    x0 *= CROP_DS;
    y0 *= CROP_DS;
    x1 *= CROP_DS;
    y1 *= CROP_DS;
    x1 = x1.min(w);
    y1 = y1.min(h);
    let iy = (2usize).max(CROP_DS).max(((y1 - y0) as f64 * 0.004) as usize);
    let ix = (2usize).max(CROP_DS).max(((x1 - x0) as f64 * 0.004) as usize);
    if x1 > x0 + 2 * ix && y1 > y0 + 2 * iy {
        x0 += ix;
        y0 += iy;
        x1 -= ix;
        y1 -= iy;
    }
    if x1 <= x0 || y1 <= y0 {
        return CropPx { x0: 0, y0: 0, x1: w as u32, y1: h as u32 };
    }
    CropPx { x0: x0 as u32, y0: y0 as u32, x1: x1 as u32, y1: y1 as u32 }
}

fn erode(src: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut dst = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut ok = true;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let yy = y as i32 + dy;
                    let xx = x as i32 + dx;
                    if yy < 0 || xx < 0 || yy >= h as i32 || xx >= w as i32 || !src[yy as usize * w + xx as usize] {
                        ok = false;
                    }
                }
            }
            dst[y * w + x] = ok;
        }
    }
    dst
}

fn largest_rect(mask: &[bool], w: usize, h: usize) -> (usize, usize, usize, usize) {
    let mut heights = vec![0i32; w];
    let mut best_area = 0i32;
    let mut best = (0usize, 0usize, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            heights[x] = if mask[y * w + x] { heights[x] + 1 } else { 0 };
        }
        let mut stack: Vec<(usize, i32)> = Vec::new();
        for x in 0..=w {
            let hx = if x < w { heights[x] } else { 0 };
            let mut start = x;
            while stack.last().is_some_and(|(_, sh)| *sh > hx) {
                let (sx, sh) = stack.pop().unwrap();
                let area = sh * (x as i32 - sx as i32);
                if area > best_area {
                    best_area = area;
                    best = (sx, y + 1 - sh as usize, x, y + 1);
                }
                start = sx;
            }
            if stack.last().is_none_or(|(_, sh)| *sh < hx) {
                stack.push((start, hx));
            }
        }
    }
    best
}
