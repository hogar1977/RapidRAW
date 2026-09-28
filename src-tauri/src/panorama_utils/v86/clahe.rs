pub fn apply(src: &[u8], width: usize, height: usize, clip_limit: f64, tiles_x: usize, tiles_y: usize) -> Vec<u8> {
    let (buf, bw, bh) = if width % tiles_x == 0 && height % tiles_y == 0 {
        (src.to_vec(), width, height)
    } else {
        let bw = width + (tiles_x - width % tiles_x);
        let bh = height + (tiles_y - height % tiles_y);
        let mut ext = vec![0u8; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let sx = reflect(x as i32, width);
                let sy = reflect(y as i32, height);
                ext[y * bw + x] = src[sy * width + sx];
            }
        }
        (ext, bw, bh)
    };
    let tile_w = bw / tiles_x;
    let tile_h = bh / tiles_y;
    let tile_area = tile_w * tile_h;
    let hist_size = 256usize;
    let lut_scale = (hist_size - 1) as f32 / tile_area as f32;
    let clip = if clip_limit > 0.0 {
        ((clip_limit * tile_area as f64 / hist_size as f64) as i32).max(1)
    } else {
        0
    };
    let mut luts = vec![0u8; tiles_x * tiles_y * hist_size];
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut hist = [0i32; 256];
            for y in 0..tile_h {
                let row = (ty * tile_h + y) * bw + tx * tile_w;
                for x in 0..tile_w {
                    hist[buf[row + x] as usize] += 1;
                }
            }
            if clip > 0 {
                let mut clipped = 0i32;
                for h in hist.iter_mut() {
                    if *h > clip {
                        clipped += *h - clip;
                        *h = clip;
                    }
                }
                let batch = clipped / hist_size as i32;
                let mut residual = clipped - batch * hist_size as i32;
                for h in hist.iter_mut() {
                    *h += batch;
                }
                if residual != 0 {
                    let step = (hist_size as i32 / residual).max(1) as usize;
                    let mut i = 0usize;
                    while i < hist_size && residual > 0 {
                        hist[i] += 1;
                        residual -= 1;
                        i += step;
                    }
                }
            }
            let mut sum = 0i32;
            let base = (ty * tiles_x + tx) * hist_size;
            for i in 0..hist_size {
                sum += hist[i];
                let v = (sum as f32 * lut_scale).round().clamp(0.0, 255.0) as u8;
                luts[base + i] = v;
            }
        }
    }
    let mut dst = vec![0u8; width * height];
    let inv_tw = 1.0 / tile_w as f32;
    let inv_th = 1.0 / tile_h as f32;
    let lut_step = hist_size;
    for y in 0..height {
        let tyf = y as f32 * inv_th - 0.5;
        let mut ty1 = tyf.floor() as i32;
        let ya = tyf - ty1 as f32;
        let ya1 = 1.0 - ya;
        let mut ty2 = ty1 + 1;
        ty1 = ty1.max(0);
        ty2 = ty2.min(tiles_y as i32 - 1);
        for x in 0..width {
            let txf = x as f32 * inv_tw - 0.5;
            let mut tx1 = txf.floor() as i32;
            let xa = txf - tx1 as f32;
            let xa1 = 1.0 - xa;
            let mut tx2 = tx1 + 1;
            tx1 = tx1.max(0);
            tx2 = tx2.min(tiles_x as i32 - 1);
            let src_val = src[y * width + x] as usize;
            let i1 = (ty1 as usize * tiles_x + tx1 as usize) * lut_step + src_val;
            let i2 = (ty1 as usize * tiles_x + tx2 as usize) * lut_step + src_val;
            let j1 = (ty2 as usize * tiles_x + tx1 as usize) * lut_step + src_val;
            let j2 = (ty2 as usize * tiles_x + tx2 as usize) * lut_step + src_val;
            let res = (luts[i1] as f32 * xa1 + luts[i2] as f32 * xa) * ya1
                + (luts[j1] as f32 * xa1 + luts[j2] as f32 * xa) * ya;
            dst[y * width + x] = res.round().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

fn reflect(mut p: i32, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let n = len as i32;
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    p %= period;
    if p < 0 {
        p += period;
    }
    if p >= n {
        (period - p) as usize
    } else {
        p as usize
    }
}
