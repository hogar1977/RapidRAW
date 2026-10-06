use super::const_::{CANVAS_PX_CAP, WARP_PAD};
use super::geom::{proj_forward, proj_inverse, Projection};
use super::lens::LensModel;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

pub struct Placed {
    pub x0: i32,
    pub y0: i32,
    pub w: usize,
    pub h: usize,
    pub img: Vec<f32>,
    pub valid: Vec<bool>,
    pub uv: Vec<f32>,
}

impl Placed {
    pub fn x1(&self) -> i32 {
        self.x0 + self.w as i32
    }
    pub fn y1(&self) -> i32 {
        self.y0 + self.h as i32
    }
}

pub struct CanvasGeom {
    pub width: u32,
    pub height: u32,
    pub tmin: f64,
    pub hmin: f64,
    pub projection: Projection,
}

// Draws one photo onto the shared canvas and keeps only the rectangle it covers.
pub fn warp_one(
    rgb: &[f32],
    sw: u32,
    sh: u32,
    rot: &Matrix3<f64>,
    f: f64,
    lens: &LensModel,
    geom: &CanvasGeom,
) -> Placed {
    let ext = frame_extent(sw, sh, rot, f, lens, geom.projection);
    let w = geom.width as i32;
    let h = geom.height as i32;
    let bx0 = ((ext.0 - geom.tmin) * f) as i32 - WARP_PAD;
    let bx1 = ((ext.1 - geom.tmin) * f) as i32 + WARP_PAD + 1;
    let by0 = ((ext.2 - geom.hmin) * f) as i32 - WARP_PAD;
    let by1 = ((ext.3 - geom.hmin) * f) as i32 + WARP_PAD + 1;
    let bx0 = bx0.clamp(0, w);
    let bx1 = bx1.clamp(0, w);
    let by0 = by0.clamp(0, h);
    let by1 = by1.clamp(0, h);
    let bw = (bx1 - bx0).max(0) as usize;
    let bh = (by1 - by0).max(0) as usize;
    let mut img = vec![0f32; bw * bh * 3];
    let mut valid = vec![false; bw * bh];
    let mut uv = vec![0f32; bw * bh * 2];
    let rt = rot.transpose();
    let hd = (sw as f64 * 0.5).hypot(sh as f64 * 0.5);
    for y in 0..bh {
        if y % 32 == 0 && super::trace::halted() {
            break;
        }
        for x in 0..bw {
            let uu = (bx0 + x as i32) as f64 / f + geom.tmin;
            let vv = (by0 + y as i32) as f64 / f + geom.hmin;
            let ray = proj_inverse(uu, vv, geom.projection);
            let cam = rt * ray;
            if cam.z <= 1e-6 {
                continue;
            }
            let mut dx = f * cam.x / cam.z;
            let mut dy = f * cam.y / cam.z;
            if !lens.is_identity() {
                let ru = dx.hypot(dy) / hd;
                let s = lens.forward_scale(ru.min(2.0));
                dx *= s;
                dy *= s;
            }
            let mx = dx + sw as f64 / 2.0;
            let my = dy + sh as f64 / 2.0;
            if mx < 0.0 || my < 0.0 || mx >= sw as f64 || my >= sh as f64 {
                continue;
            }
            let pix = sample(rgb, sw, sh, mx as f32, my as f32);
            let i = y * bw + x;
            img[i * 3] = pix[0];
            img[i * 3 + 1] = pix[1];
            img[i * 3 + 2] = pix[2];
            valid[i] = true;
            uv[i * 2] = ((mx - sw as f64 / 2.0) / hd) as f32;
            uv[i * 2 + 1] = ((my - sh as f64 / 2.0) / hd) as f32;
        }
    }
    Placed { x0: bx0, y0: by0, w: bw, h: bh, img, valid, uv }
}

// Same drawing as warp_one, with a canvas-pixel nudge applied before the source lookup.
pub fn warp_one_nudged(
    rgb: &[f32],
    sw: u32,
    sh: u32,
    rot: &Matrix3<f64>,
    f: f64,
    lens: &LensModel,
    geom: &CanvasGeom,
    nudge: &(impl Fn(f64, f64) -> (f64, f64) + Sync),
) -> Placed {
    let ext = frame_extent(sw, sh, rot, f, lens, geom.projection);
    let w = geom.width as i32;
    let h = geom.height as i32;
    let bx0 = ((ext.0 - geom.tmin) * f) as i32 - WARP_PAD;
    let bx1 = ((ext.1 - geom.tmin) * f) as i32 + WARP_PAD + 1;
    let by0 = ((ext.2 - geom.hmin) * f) as i32 - WARP_PAD;
    let by1 = ((ext.3 - geom.hmin) * f) as i32 + WARP_PAD + 1;
    let bx0 = bx0.clamp(0, w);
    let bx1 = bx1.clamp(0, w);
    let by0 = by0.clamp(0, h);
    let by1 = by1.clamp(0, h);
    let bw = (bx1 - bx0).max(0) as usize;
    let bh = (by1 - by0).max(0) as usize;
    let mut img = vec![0f32; bw * bh * 3];
    let mut valid = vec![false; bw * bh];
    let mut uv = vec![0f32; bw * bh * 2];
    let rt = rot.transpose();
    let hd = (sw as f64 * 0.5).hypot(sh as f64 * 0.5);
    if bw > 0 && bh > 0 {
        img.par_chunks_mut(bw * 3)
            .zip(valid.par_chunks_mut(bw))
            .zip(uv.par_chunks_mut(bw * 2))
            .enumerate()
            .for_each(|(y, ((row, flags), uvs))| {
                if super::trace::halted() {
                    return;
                }
                for x in 0..bw {
                    let cx = (bx0 + x as i32) as f64;
                    let cy = (by0 + y as i32) as f64;
                    let (nx, ny) = nudge(cx, cy);
                    let uu = (cx + nx) / f + geom.tmin;
                    let vv = (cy + ny) / f + geom.hmin;
                    let ray = proj_inverse(uu, vv, geom.projection);
                    let cam = rt * ray;
                    if cam.z <= 1e-6 {
                        continue;
                    }
                    let mut dx = f * cam.x / cam.z;
                    let mut dy = f * cam.y / cam.z;
                    if !lens.is_identity() {
                        let ru = dx.hypot(dy) / hd;
                        let s = lens.forward_scale(ru.min(2.0));
                        dx *= s;
                        dy *= s;
                    }
                    let mx = dx + sw as f64 / 2.0;
                    let my = dy + sh as f64 / 2.0;
                    if mx < 0.0 || my < 0.0 || mx >= sw as f64 || my >= sh as f64 {
                        continue;
                    }
                    let pix = sample(rgb, sw, sh, mx as f32, my as f32);
                    row[x * 3] = pix[0];
                    row[x * 3 + 1] = pix[1];
                    row[x * 3 + 2] = pix[2];
                    flags[x] = true;
                    uvs[x * 2] = ((mx - sw as f64 / 2.0) / hd) as f32;
                    uvs[x * 2 + 1] = ((my - sh as f64 / 2.0) / hd) as f32;
                }
            });
    }
    Placed { x0: bx0, y0: by0, w: bw, h: bh, img, valid, uv }
}

// Finds how big the canvas must be, and switches a too-wide flat view to a curved one.
pub fn canvas_geom(rots: &[Matrix3<f64>], f: f64, sw: u32, sh: u32, lens: &LensModel, mut projection: Projection) -> (CanvasGeom, Projection) {
    let mut bounds = Vec::new();
    for r in rots {
        bounds.push(frame_extent(sw, sh, r, f, lens, projection));
    }
    let mut geom = geom_from_bounds(&bounds, f, projection);
    if (geom.width as u64) * (geom.height as u64) > CANVAS_PX_CAP as u64 && projection == Projection::Perspective {
        projection = Projection::Cylindrical;
        bounds.clear();
        for r in rots {
            bounds.push(frame_extent(sw, sh, r, f, lens, projection));
        }
        geom = geom_from_bounds(&bounds, f, projection);
    }
    (geom, projection)
}

fn geom_from_bounds(bounds: &[(f64, f64, f64, f64)], f: f64, projection: Projection) -> CanvasGeom {
    let mut tmin = 1e9f64;
    let mut tmax = -1e9f64;
    let mut hmin = 1e9f64;
    let mut hmax = -1e9f64;
    for b in bounds {
        tmin = tmin.min(b.0);
        tmax = tmax.max(b.1);
        hmin = hmin.min(b.2);
        hmax = hmax.max(b.3);
    }
    if bounds.is_empty() {
        tmin = 0.0;
        tmax = 0.0;
        hmin = 0.0;
        hmax = 0.0;
    }
    let width = ((tmax - tmin) * f) as u32 + 1;
    let height = ((hmax - hmin) * f) as u32 + 1;
    CanvasGeom { width, height, tmin, hmin, projection }
}

fn frame_extent(sw: u32, sh: u32, rot: &Matrix3<f64>, f: f64, lens: &LensModel, projection: Projection) -> (f64, f64, f64, f64) {
    let mut tmin = 1e9f64;
    let mut tmax = -1e9f64;
    let mut hmin = 1e9f64;
    let mut hmax = -1e9f64;
    for i in 0..17 {
        let t = i as f64 / 16.0;
        let pts = [
            [t * sw as f64, 0.0],
            [sw as f64, t * sh as f64],
            [(1.0 - t) * sw as f64, sh as f64],
            [0.0, (1.0 - t) * sh as f64],
        ];
        for p in pts {
            let b = super::geom::bearings(&[p], f, sw, sh, lens);
            let rw = rot * b[0];
            let (th, hh) = proj_forward(rw.x, rw.y, rw.z, projection);
            tmin = tmin.min(th);
            tmax = tmax.max(th);
            hmin = hmin.min(hh);
            hmax = hmax.max(hh);
        }
    }
    let _ = Vector3::<f64>::zeros();
    (tmin, tmax, hmin, hmax)
}

fn sample(rgb: &[f32], w: u32, h: u32, x: f32, y: f32) -> [f32; 3] {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let at = |yy: i32, xx: i32| {
        let xx = xx.clamp(0, w as i32 - 1) as u32;
        let yy = yy.clamp(0, h as i32 - 1) as u32;
        let i = ((yy * w + xx) * 3) as usize;
        [rgb[i], rgb[i + 1], rgb[i + 2]]
    };
    let a = at(y0, x0);
    let b = at(y0, x0 + 1);
    let c = at(y0 + 1, x0);
    let d = at(y0 + 1, x0 + 1);
    let mut o = [0f32; 3];
    for k in 0..3 {
        let u = a[k] * (1.0 - dx) + b[k] * dx;
        let v = c[k] * (1.0 - dx) + d[k] * dx;
        o[k] = u * (1.0 - dy) + v * dy;
    }
    o
}

// Darkens the corners back to an even brightness using the lens profile.
pub fn devignette(rgb: &mut [f32], w: u32, h: u32, lens: &LensModel) {
    if !lens.has_vig {
        return;
    }
    let hd2 = (w as f64 * 0.5).powi(2) + (h as f64 * 0.5).powi(2);
    let k = lens.k;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 + 0.5 - w as f64 / 2.0;
            let dy = y as f64 + 0.5 - h as f64 / 2.0;
            let r2 = (k * k) * (dx * dx + dy * dy) / hd2;
            let v = 1.0 + lens.vig_k1 * r2 + lens.vig_k2 * r2 * r2 + lens.vig_k3 * r2 * r2 * r2;
            let s = 1.0 / v.max(0.1);
            let i = ((y * w + x) * 3) as usize;
            rgb[i] = (rgb[i] as f64 * s) as f32;
            rgb[i + 1] = (rgb[i + 1] as f64 * s) as f32;
            rgb[i + 2] = (rgb[i + 2] as f64 * s) as f32;
        }
    }
}
