use std::path::Path;

pub const ORDER: &[&str] = &["0140", "0200", "0559", "0566", "0578", "4171", "6498", "6645", "8425", "8715", "9237"];

pub fn select(input: &Path, wanted: &[String]) -> Vec<(String, Vec<String>)> {
    let mut sets = discover(input);
    sets.sort_by_key(|set| ORDER.iter().position(|stem| set.0.contains(stem)).unwrap_or(99));
    if !wanted.is_empty() {
        sets.retain(|set| wanted.iter().any(|want| set.0.contains(want)));
    }
    sets
}

fn discover(input: &Path) -> Vec<(String, Vec<String>)> {
    let mut found = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for dir in [input.to_path_buf(), input.join("old1")] {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with("_Pano.pano.log") {
                continue;
            }
            let stem = name.trim_end_matches("_Pano.pano.log").to_string();
            if !seen.insert(stem.clone()) {
                continue;
            }
            let image_dir = if dir.ends_with("old1") { input.to_path_buf() } else { dir.clone() };
            if let Some(paths) = paths_from_log(&entry.path(), &image_dir) {
                found.push((stem, paths));
            }
        }
    }
    // v27: a set of non-raw inputs has no stitch log to read, so fall back to
    // grouping the image files in the input directory by name.
    if found.is_empty() {
        found = discover_by_name(input);
    }
    found
}

fn is_image(name: &str) -> bool {
    let Some((_, ext)) = name.rsplit_once('.') else { return false };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp" | "gif"
    )
}

/// Group `PREFIX_01.EXT`, `PREFIX_02.EXT`, ... into one set per PREFIX. The
/// trailing number is the frame index; everything before it is the set name.
fn discover_by_name(input: &Path) -> Vec<(String, Vec<String>)> {
    let mut groups: std::collections::BTreeMap<String, std::collections::BTreeMap<u32, String>> =
        std::collections::BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(input) else { return Vec::new() };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !is_image(&name) {
            continue;
        }
        let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(&name);
        let Some((prefix, index)) = stem.rsplit_once('_') else { continue };
        let Ok(index) = index.parse::<u32>() else { continue };
        let prefix = prefix.to_string();
        groups
            .entry(prefix)
            .or_default()
            .entry(index)
            .or_insert_with(|| input.join(&name).to_string_lossy().into_owned());
    }
    groups
        .into_iter()
        .filter(|(_, frames)| frames.len() >= 2)
        .map(|(prefix, frames)| (prefix, frames.into_values().collect()))
        .collect()
}

fn paths_from_log(log: &Path, image_dir: &Path) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(log).ok()?;
    let mut indexed = std::collections::BTreeMap::new();
    for line in text.lines() {
        // "loaded" misses a file when that line was not flushed. "points" names every photo in the set.
        let rest = line.split(" points ").nth(1).or_else(|| line.split(" loaded ").nth(1));
        let Some(rest) = rest else { continue };
        let mut parts = rest.split_whitespace();
        let Some(index) = parts.next() else { continue };
        let Some(name) = parts.next() else { continue };
        let Ok(n) = index.split('/').next().unwrap_or("").parse::<usize>() else { continue };
        let name = name.trim_end_matches(':');
        indexed.insert(n, image_dir.join(name).to_string_lossy().into_owned());
    }
    if indexed.len() < 2 {
        return None;
    }
    Some(indexed.into_values().collect())
}
