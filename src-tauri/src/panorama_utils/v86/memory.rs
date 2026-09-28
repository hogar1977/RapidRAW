use super::sift::pyramid_bytes;

// Bytes the feature pass needs for one photo at a time, plus every photo's pixels and descriptors.
pub fn feature_peak_bytes(n: u32, width: u32, height: u32) -> u64 {
    let pix = (width as u64).saturating_mul(height as u64);
    let sources = (n as u64).saturating_mul(pix).saturating_mul(12);
    let descriptors = (n as u64).saturating_mul(12_000).saturating_mul(128).saturating_mul(4);
    sources.saturating_add(descriptors).saturating_add(pyramid_bytes(width, height))
}

// Bytes while the pictures are being joined, with only one warped photo in memory.
pub fn stitch_peak_bytes(n: u32, width: u32, height: u32) -> u64 {
    let n = n.max(1);
    let pix = (width as u64).saturating_mul(height as u64);
    let sources = (n as u64).saturating_mul(pix).saturating_mul(12);
    let area = (pix as f64 * (1.0 + 0.5 * (n as f64 - 1.0))).round() as u64;
    let output = area.saturating_mul(12);
    let coverage = area;
    let winners = area;
    let warped = pix.saturating_mul(12);
    let overlap = pix.saturating_mul(12) / 2;
    sources
        .saturating_add(output)
        .saturating_add(coverage)
        .saturating_add(winners)
        .saturating_add(warped)
        .saturating_add(overlap)
}

pub fn peak_bytes(n: u32, width: u32, height: u32) -> u64 {
    feature_peak_bytes(n, width, height).max(stitch_peak_bytes(n, width, height))
}
