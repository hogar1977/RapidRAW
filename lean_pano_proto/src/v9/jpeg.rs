use image::{RgbImage, imageops};
use rapidraw_lib::panorama_stitching;
use std::io::Cursor;

pub struct QuarterJpeg {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub full_w: u32,
    pub full_h: u32,
}

pub fn decode_quarter(path: &str) -> Result<QuarterJpeg, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    let orientation = panorama_stitching::orientation_of_bytes(&bytes);
    let mut packets = Vec::new();
    if let Some(jpeg) = fuji_embedded_jpeg(&bytes) {
        packets.push(jpeg);
    }
    packets.extend(tiff_jpegs(&bytes));
    if packets.is_empty() {
        return Err(format!("No embedded preview in {path}"));
    }
    let mut last_error = String::from("No embedded preview");
    for jpeg in &packets {
        match decode_scaled(jpeg) {
            Ok((rgb, use_w, use_h, raw_w, raw_h)) => {
                let image = RgbImage::from_raw(use_w, use_h, rgb).ok_or_else(|| format!("Could not build the preview for {path}"))?;
                let turned = orient(image, orientation);
                let (width, height) = turned.dimensions();
                let (full_w, full_h) = orient_size(raw_w, raw_h, orientation);
                return Ok(QuarterJpeg { rgb: turned.into_raw(), width, height, full_w, full_h });
            }
            Err(error) => last_error = error,
        }
    }
    let image = image::load_from_memory_with_format(packets[0], image::ImageFormat::Jpeg).map_err(|e| format!("{last_error}; {e}"))?;
    let rgb = image.to_rgb8();
    let (raw_w, raw_h) = rgb.dimensions();
    let turned = orient(rgb, orientation);
    let (width, height) = turned.dimensions();
    let (full_w, full_h) = orient_size(raw_w, raw_h, orientation);
    Ok(QuarterJpeg { rgb: turned.into_raw(), width, height, full_w, full_h })
}

fn decode_scaled(jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32, u32, u32), String> {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(jpeg));
    decoder.read_info().map_err(|e| e.to_string())?;
    let info = decoder.info().ok_or_else(|| "No JPEG size".to_string())?;
    let components = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => 3,
        jpeg_decoder::PixelFormat::L8 => 1,
        _ => return Err("Unsupported JPEG".into()),
    };
    let orig_w = info.width as u32;
    let orig_h = info.height as u32;
    let req_w = info.width;
    let req_h = info.height;
    let (use_w, use_h) = decoder.scale(req_w, req_h).map_err(|e| e.to_string())?;
    let pixels = decoder.decode().map_err(|e| e.to_string())?;
    let use_w = use_w as u32;
    let use_h = use_h as u32;
    if pixels.len() != (use_w as usize) * (use_h as usize) * components {
        return Err(format!("JPEG scale mismatch: {} bytes for {use_w}x{use_h}", pixels.len()));
    }
    Ok((expand_rgb(&pixels, components), use_w, use_h, orig_w, orig_h))
}

fn expand_rgb(pixels: &[u8], components: usize) -> Vec<u8> {
    if components == 3 {
        return pixels.to_vec();
    }
    let mut rgb = Vec::with_capacity(pixels.len() * 3);
    for &v in pixels {
        rgb.extend_from_slice(&[v, v, v]);
    }
    rgb
}

fn orient(image: RgbImage, orientation: u16) -> RgbImage {
    match orientation {
        2 => imageops::flip_horizontal(&image),
        3 => imageops::rotate180(&image),
        4 => imageops::flip_vertical(&image),
        5 => imageops::flip_horizontal(&imageops::rotate90(&image)),
        6 => imageops::rotate90(&image),
        7 => imageops::flip_horizontal(&imageops::rotate270(&image)),
        8 => imageops::rotate270(&image),
        _ => image,
    }
}

fn orient_size(w: u32, h: u32, orientation: u16) -> (u32, u32) {
    match orientation {
        5 | 6 | 7 | 8 => (h, w),
        _ => (w, h),
    }
}

fn fuji_embedded_jpeg(file_bytes: &[u8]) -> Option<&[u8]> {
    const MAGIC: &[u8] = b"FUJIFILMCCD-RAW ";
    if file_bytes.len() < 0x5c || !file_bytes.starts_with(MAGIC) {
        return None;
    }
    let offset = u32::from_be_bytes(file_bytes[0x54..0x58].try_into().ok()?) as usize;
    let length = u32::from_be_bytes(file_bytes[0x58..0x5c].try_into().ok()?) as usize;
    let jpeg = file_bytes.get(offset..offset.checked_add(length)?)?;
    if jpeg.starts_with(&[0xFF, 0xD8]) { Some(jpeg) } else { None }
}

fn tiff_jpegs(buf: &[u8]) -> Vec<&[u8]> {
    let le = match buf.get(..4) {
        Some([0x49, 0x49, 0x2A, 0x00]) => true,
        Some([0x4D, 0x4D, 0x00, 0x2A]) => false,
        _ => return Vec::new(),
    };
    let rd16 = |o: usize| -> Option<u64> {
        let b: [u8; 2] = buf.get(o..o + 2)?.try_into().ok()?;
        Some(if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) } as u64)
    };
    let rd32 = |o: usize| -> Option<u64> {
        let b: [u8; 4] = buf.get(o..o + 4)?.try_into().ok()?;
        Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) } as u64)
    };
    let mut candidates: Vec<(u64, u64)> = Vec::new();
    let Some(first) = rd32(4) else { return Vec::new() };
    let mut queue: Vec<u64> = vec![first];
    let mut seen = std::collections::HashSet::new();
    while let Some(ifd) = queue.pop() {
        if !seen.insert(ifd) || seen.len() > 64 {
            continue;
        }
        let Some(n) = rd16(ifd as usize) else { continue };
        let mut compression = 0u64;
        let mut strip: Option<(u64, u64)> = None;
        let mut old_jpeg: Option<(u64, u64)> = None;
        for i in 0..n {
            let e = ifd as usize + 2 + (i as usize) * 12;
            let (Some(tag), Some(count), Some(val)) = (rd16(e), rd32(e + 4), rd32(e + 8)) else { continue };
            match tag {
                259 => compression = val,
                273 if count == 1 => strip = Some((val, strip.map_or(0, |s| s.1))),
                279 if count == 1 => strip = strip.map(|s| (s.0, val)).or(Some((0, val))),
                513 => old_jpeg = Some((val, old_jpeg.map_or(0, |s| s.1))),
                514 => old_jpeg = old_jpeg.map(|s| (s.0, val)).or(Some((0, val))),
                330 => {
                    if count == 1 {
                        queue.push(val);
                    } else {
                        for j in 0..count.min(8) {
                            if let Some(p) = rd32(val as usize + (j as usize) * 4) {
                                queue.push(p);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if matches!(compression, 6 | 7)
            && let Some(s) = strip
        {
            candidates.push(s);
        }
        if let Some(oj) = old_jpeg {
            candidates.push(oj);
        }
        if let Some(next) = rd32(ifd as usize + 2 + (n as usize) * 12)
            && next != 0
        {
            queue.push(next);
        }
    }
    candidates.sort_by_key(|&(_, len)| std::cmp::Reverse(len));
    let mut found = Vec::new();
    for (off, len) in candidates {
        if let Some(bytes) = buf.get(off as usize..(off + len) as usize)
            && bytes.starts_with(&[0xFF, 0xD8])
        {
            found.push(bytes);
        }
    }
    found
}
