use rapidraw_lib::panorama_utils::camera::{focal_px_from_fov, fov_rad_from_focal_mm_35eq, parse_focal_mm_35eq};
use rapidraw_lib::panorama_utils::v86::ba::bundle_rotations;
use rapidraw_lib::panorama_utils::v86::clahe;
use rapidraw_lib::panorama_utils::v86::const_::{focal_px_from_35eq, FF_DIAG_MM};
use rapidraw_lib::panorama_utils::v86::geom::{self, kabsch, proj_forward, proj_inverse, recommend_projection, yaw_pitch_deg, Projection};
use rapidraw_lib::panorama_utils::v86::lens::{LensKind, LensModel};
use rapidraw_lib::panorama_utils::v86::memory::{feature_peak_bytes, peak_bytes, stitch_peak_bytes};
use rapidraw_lib::panorama_utils::v86::photo::{illumination_corner, solve_photometry, Photo};
use rapidraw_lib::panorama_utils::v86::rng::NumpyRng;
use rapidraw_lib::panorama_utils::v86::seam::{graphcut_mask, graphcut_refine_nodes};
use rapidraw_lib::panorama_utils::v86::sift;
use rapidraw_lib::panorama_utils::v86::stitch::{self, InputFrame};
use rapidraw_lib::panorama_utils::v86::trace::{self, StitchLog};
use rapidraw_lib::panorama_utils::v86::warp::{self, Placed};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;

#[test]
fn fov_parse_from_exif_map() {
    let mut map = HashMap::new();
    map.insert("FocalLengthIn35mmFilm".to_string(), "35".to_string());
    map.insert("FocalLength".to_string(), "23".to_string());
    assert!((parse_focal_mm_35eq(&map) - 35.0).abs() < 1e-6);
    let fov = fov_rad_from_focal_mm_35eq(35.0);
    assert!(fov > 0.9 && fov < 1.2);
    let fpx = focal_px_from_fov(fov, 4000);
    assert!(fpx > 1000.0);
}

#[test]
fn fov_parse_prefers_distinct_35mm_eq_across_brands() {
    // Fuji-style APS-C
    let mut fuji = HashMap::new();
    fuji.insert("FocalLength".to_string(), "62.5".to_string());
    fuji.insert("FocalLengthIn35mmFilm".to_string(), "94".to_string());
    assert!((parse_focal_mm_35eq(&fuji) - 94.0).abs() < 1e-6);

    // Canon/Nikon-style crop body
    let mut canon = HashMap::new();
    canon.insert("FocalLength".to_string(), "50".to_string());
    canon.insert("FocalLengthIn35mmFilm".to_string(), "80".to_string());
    assert!((parse_focal_mm_35eq(&canon) - 80.0).abs() < 1e-6);

    // Full-frame: tags often match
    let mut ff = HashMap::new();
    ff.insert("FocalLength".to_string(), "85".to_string());
    ff.insert("FocalLengthIn35mmFilm".to_string(), "85".to_string());
    assert!((parse_focal_mm_35eq(&ff) - 85.0).abs() < 1e-6);
}

#[test]
fn fov_parse_scale_factor_when_35mm_missing() {
    let mut map = HashMap::new();
    map.insert("FocalLength".to_string(), "35".to_string());
    map.insert("ScaleFactor35efl".to_string(), "1.5".to_string());
    assert!((parse_focal_mm_35eq(&map) - 52.5).abs() < 1e-6);
}

#[test]
fn fov_parse_identical_short_tags_aps_c_heuristic() {
    let mut map = HashMap::new();
    map.insert("FocalLength".to_string(), "18".to_string());
    map.insert("FocalLengthIn35mmFilm".to_string(), "18".to_string());
    assert!((parse_focal_mm_35eq(&map) - 27.0).abs() < 1e-6);
}

#[test]
fn fov_parse_missing_exif_uses_default_50mm() {
    let map = HashMap::new();
    assert!((parse_focal_mm_35eq(&map) - 50.0).abs() < 1e-6);
}

#[test]
fn fov_parse_absurd_exif_falls_back_to_default() {
    let mut map = HashMap::new();
    map.insert("FocalLengthIn35mmFilm".to_string(), "99999".to_string());
    assert!((parse_focal_mm_35eq(&map) - 50.0).abs() < 1e-6);

    let mut map2 = HashMap::new();
    map2.insert("FocalLength".to_string(), "0.01".to_string());
    assert!((parse_focal_mm_35eq(&map2) - 50.0).abs() < 1e-6);
}


#[test]
fn diagonal_focal_matches_sensor_diagonal() {
    let f = focal_px_from_35eq(4000, 3000, 35.0);
    let expect = 35.0 * (4000.0f64.hypot(3000.0)) / FF_DIAG_MM;
    assert!((f - expect).abs() < 1e-6);
}

#[test]
fn kabsch_recovers_twenty_five_degree_yaw() {
    let yaw = 25.0f64.to_radians();
    let c = yaw.cos();
    let s = yaw.sin();
    let rot = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
    let dirs = [
        Vector3::new(0.2, 0.1, 1.0),
        Vector3::new(-0.3, 0.2, 1.0),
        Vector3::new(0.1, -0.4, 1.0),
        Vector3::new(0.5, 0.2, 1.0),
        Vector3::new(-0.2, -0.3, 1.0),
    ];
    let a: Vec<_> = dirs.iter().map(|v| v.normalize()).collect();
    let b: Vec<_> = a.iter().map(|v| rot * v).collect();
    let got = kabsch(&a, &b).unwrap();
    let (gy, _) = geom::yaw_pitch_deg(&got);
    assert!((gy - 25.0).abs() < 0.05, "yaw {gy}");
}

#[test]
fn lens_newton_inverts_forward_radius() {
    let mut lens = LensModel::identity();
    lens.kind = LensKind::PtLens;
    lens.a = -0.02;
    lens.b = 0.01;
    lens.c = 0.005;
    lens.k = 1.0;
    let ru = 0.45;
    let rd = ru * lens.forward_scale(ru);
    let back = lens.invert_radius(rd);
    assert!((back - ru).abs() < 1e-4, "back {back}");
}

#[test]
fn numpy_choice_seed_zero_matches_samples() {
    let mut rng = NumpyRng::seed0();
    let expected = [[12usize, 10, 15], [19, 1, 0], [11, 10, 17], [13, 10, 12], [15, 13, 4]];
    for row in expected {
        assert_eq!(rng.choice(20, 3), row);
    }
}

#[test]
fn projection_round_trip_and_cutovers() {
    for proj in [Projection::Perspective, Projection::Cylindrical, Projection::Spherical] {
        let (th, hh) = proj_forward(0.2, -0.15, 0.9, proj);
        let ray = proj_inverse(th, hh, proj);
        let (th2, hh2) = proj_forward(ray.x, ray.y, ray.z, proj);
        assert!((th - th2).abs() < 1e-6 && (hh - hh2).abs() < 1e-6);
    }
    assert_eq!(recommend_projection(&[0.0, 10.0], &[0.0], 40.0, 30.0), Projection::Perspective);
    assert_eq!(recommend_projection(&[0.0, 40.0], &[0.0], 50.0, 30.0), Projection::Cylindrical);
    assert_eq!(recommend_projection(&[0.0, 100.0], &[0.0], 70.0, 40.0), Projection::Spherical);
}

#[test]
fn memory_peak_matches_formula() {
    let n = 4u32;
    let w = 100u32;
    let h = 50u32;
    let pix = (w as u64) * (h as u64);
    let sources = (n as u64) * pix * 12;
    let descriptors = (n as u64) * 12_000 * 128 * 4;
    let pyr = rapidraw_lib::panorama_utils::v86::sift::pyramid_bytes(w, h);
    assert_eq!(feature_peak_bytes(n, w, h), sources + descriptors + pyr);
    let area = (pix as f64 * (1.0 + 0.5 * (n as f64 - 1.0))).round() as u64;
    let stitch = sources + area * 12 + area + area + pix * 12 + pix * 12 / 2;
    assert_eq!(stitch_peak_bytes(n, w, h), stitch);
    assert_eq!(peak_bytes(n, w, h), feature_peak_bytes(n, w, h).max(stitch));
}

#[test]
fn bundle_recovers_eight_degree_yaw() {
    let w = 800u32;
    let h = 600u32;
    let f = focal_px_from_35eq(w, h, 35.0);
    let yaw = 8.0f64.to_radians();
    let c = yaw.cos();
    let s = yaw.sin();
    let r1 = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
    let pair_rot = r1.transpose();
    let lens = LensModel::identity();
    let mut pa = Vec::new();
    let mut pb = Vec::new();
    for y in (120..480).step_by(40) {
        for x in (160..640).step_by(40) {
            let p = [x as f64, y as f64];
            let b0 = geom::bearings(&[p], f, w, h, &lens)[0];
            let b1 = pair_rot * b0;
            if b1.z <= 1e-6 {
                continue;
            }
            let qx = f * b1.x / b1.z + w as f64 / 2.0;
            let qy = f * b1.y / b1.z + h as f64 / 2.0;
            if qx > 20.0 && qy > 20.0 && qx < w as f64 - 20.0 && qy < h as f64 - 20.0 {
                pa.push(p);
                pb.push([qx, qy]);
            }
        }
    }
    assert!(pa.len() >= 12);
    let pairs = vec![(0usize, 1usize, pair_rot, pa, pb, 40usize, 4usize)];
    let rots = bundle_rotations(&pairs, 0, &[None, Some(0)], &[0, 1], w, h, f, &lens);
    let (got, pitch) = geom::yaw_pitch_deg(&rots[1]);
    assert!(pitch.abs() < 0.1, "pitch {pitch}");
    assert!((got - 8.0).abs() < 0.1, "yaw {got}");
}

#[test]
fn photometry_gain_ratio_is_about_one_point_two_five() {
    let a = block_frame(64, 64, 1.0);
    let b = block_frame(64, 64, 1.0 / 1.25);
    let photo: Photo = solve_photometry(&[a, b], &[(0, 1)], 0);
    let ratio = photo.gains[1] / photo.gains[0];
    assert!((1.20..=1.30).contains(&ratio), "ratio {ratio} gains {:?}", photo.gains);
}

#[test]
fn flat_sky_lifts_a_dark_corner() {
    let frame = radial_sky_frame(320, 240);
    let photo = solve_photometry(&[frame], &[], 0);
    let corner = illumination_corner(&photo, &radial_sky_frame(320, 240));
    assert!(corner < 0.85, "corner {corner} coef {:?}", photo.coef);
}

#[test]
fn seam_keeps_exclusive_sides_on_a_four_by_four() {
    let w = 4usize;
    let h = 4usize;
    let mut lum_a = vec![0.2f32; w * h];
    let mut lum_b = vec![0.2f32; w * h];
    let mut va = vec![false; w * h];
    let mut vb = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            va[i] = x <= 2;
            vb[i] = x >= 1;
            if x == 1 {
                lum_a[i] = 0.9;
                lum_b[i] = 0.1;
            }
        }
    }
    let mask = graphcut_mask(&lum_a, &lum_b, &va, &vb, w, h);
    for y in 0..h {
        assert!(mask[y * w] > 0.5);
        assert!(mask[y * w + 3] < 0.5);
    }
}

#[test]
fn diagonal_seam_solves_only_the_band() {
    let w = 1600usize;
    let h = 900usize;
    let mut lum_a = vec![0.8f32; w * h];
    let mut lum_b = vec![0.05f32; w * h];
    let mut va = vec![false; w * h];
    let mut vb = vec![false; w * h];
    for y in 0..h {
        let diag = 8.0 + (w as f32 - 16.0) * (y as f32) / (h as f32 - 1.0);
        for x in 0..w {
            let i = y * w + x;
            va[i] = x + 4 < w;
            vb[i] = x >= 4;
            if (x as f32 - diag).abs() < 12.0 {
                lum_a[i] = 0.2;
                lum_b[i] = 0.2;
            }
        }
    }
    let nodes = graphcut_refine_nodes(&lum_a, &lum_b, &va, &vb, w, h);
    let full = (h / 4) * (w / 4);
    assert!(nodes > 0, "refine nodes {nodes}");
    assert!(nodes * 3 < full, "refine nodes {nodes} full grid about {full}");
}

#[test]
fn clahe_matches_reference_tile() {
    let src: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
    let got = clahe::apply(&src, 64, 64, 2.5, 8, 8);
    let expect = include_bytes!("clahe_64.bin");
    assert_eq!(got, expect);
}

#[test]
fn sift_matches_reference_point() {
    let gray = include_bytes!("sift_gray.bin");
    let mask = vec![255u8; gray.len()];
    let feats = sift::detect(gray, &mask, 128, 96);
    let raw = include_bytes!("sift_all.bin");
    let n = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mut refs = Vec::with_capacity(n);
    let mut off = 4usize;
    for _ in 0..n {
        let x = f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()) as f64;
        let y = f32::from_le_bytes(raw[off + 4..off + 8].try_into().unwrap()) as f64;
        off += 8;
        let mut desc = [0f32; 128];
        for i in 0..128 {
            desc[i] = f32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
            off += 4;
        }
        refs.push((x, y, desc));
    }
    let mut best_cos = 0.0f32;
    let mut near = 0usize;
    for f in &feats {
        for (x, y, desc) in &refs {
            if (f.pt[0] - x).hypot(f.pt[1] - y) <= 0.5 {
                near += 1;
                best_cos = best_cos.max(cosine_desc(&f.desc, desc));
            }
        }
    }
    assert!(near > 0, "no point within 0.5 px, found {}", feats.len());
    assert!(best_cos >= 0.99, "cosine {best_cos} near {near}");
}

#[test]
#[ignore]
fn write_rapidraw_frames() {
    rapidraw_lib::panorama_stitching::write_loaded_frames(
        "/home/dalibor/Projects/RapidRAW/temporary_pano",
        "/home/dalibor/tmp/rr_decode",
    )
    .expect("decode");
}

#[test]
#[ignore]
fn lake_sky_column() {
    let dir = "/home/dalibor/tmp/rr_decode";
    let out = Path::new("/home/dalibor/tmp/lake_sky");
    std::fs::create_dir_all(out).unwrap();
    let names = ["DSCF8715", "DSCF8716", "DSCF8717", "DSCF8718", "DSCF8719", "DSCF8720", "DSCF8721"];
    let gains = [0.45_f32, 0.75, 0.85, 1.0, 1.2, 1.5, 1.6];
    let mut frames = Vec::new();
    for (name, gain) in names.iter().zip(gains) {
        let (mut rgb, w, h) = read_prepared(&format!("{dir}/{name}.f32"));
        for p in &mut rgb {
            *p *= gain;
        }
        frames.push(InputFrame { name: format!("{name}.RAF"), width: w, height: h, rgb });
    }
    let mut lens = LensModel::identity();
    lens.kind = LensKind::PtLens;
    lens.a = 0.00267;
    lens.b = -0.00450;
    lens.c = 0.01470;
    lens.k = 1.529 / 1.534;
    lens.vig_k1 = -0.1619;
    lens.vig_k2 = 0.5330;
    lens.vig_k3 = -0.7537;
    lens.has_vig = true;
    let stop = Arc::new(AtomicBool::new(false));
    let log = StitchLog::create(&out.join("run.log"), stop).unwrap();
    let _guard = trace::install(log);
    let drawn = stitch::stitch(&frames, lens, 94.0, false, &[], 1.529, false, &|msg| println!("{msg}")).expect("stitch");
    let full = stitch::compose(
        &frames,
        &drawn.rotations,
        &drawn.kept,
        lens,
        drawn.focal_px,
        94.0,
        Projection::Perspective,
        false,
        &|msg| println!("{msg}"),
    )
    .expect("compose");
    let x0 = (full.crop_x * full.width as f64).round() as usize;
    let y0 = (full.crop_y * full.height as f64).round() as usize;
    let cw = (full.crop_w * full.width as f64).round() as usize;
    let rows = 160usize.min((full.crop_h * full.height as f64).round() as usize);
    let mut strip = Vec::with_capacity(cw * rows);
    for y in y0..y0 + rows {
        for x in x0..x0 + cw {
            let i = (y * full.width as usize + x) * 3;
            let l = 0.2126 * full.rgb[i] + 0.7152 * full.rgb[i + 1] + 0.0722 * full.rgb[i + 2];
            strip.push(l);
        }
    }
    let mut bytes = Vec::with_capacity(8 + strip.len() * 4);
    bytes.extend_from_slice(&(cw as u32).to_le_bytes());
    bytes.extend_from_slice(&(rows as u32).to_le_bytes());
    for v in &strip {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out.join("strip.f32"), bytes).unwrap();
    let text = std::fs::read_to_string(out.join("run.log")).unwrap();
    let mut corners = Vec::new();
    for line in text.lines() {
        if let Some(pos) = line.find("corner=") {
            let num: String = line[pos + 7..].chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
            corners.push(num.parse::<f64>().unwrap());
        }
    }
    println!("corners {corners:?} strip {cw}x{rows}");
    assert!(corners.len() >= 2, "brightness lines missing corner");
    assert!(corners.iter().all(|c| (0.85..0.99).contains(c)), "corners {corners:?}");
    assert!(!text.lines().any(|l| l.contains(" sky ")), "unexpected stage in log");
}

// Stitches seven prepared frames and records whether each join is copied or moved.
#[test]
#[ignore]
fn same_pixels_preview() {
    let dir = "/home/dalibor/tmp/same_decode";
    let names = ["DSCF8715", "DSCF8716", "DSCF8717", "DSCF8718", "DSCF8719", "DSCF8720", "DSCF8721"];
    let gains = [0.45_f32, 0.75, 0.85, 1.0, 1.2, 1.5, 1.6];
    let mut frames = Vec::new();
    for name in names {
        let (rgb, w, h) = read_prepared(&format!("{dir}/{name}.f32"));
        frames.push(InputFrame { name: format!("{name}.RAF"), width: w, height: h, rgb });
    }
    let mut lens = LensModel::identity();
    lens.kind = LensKind::PtLens;
    lens.a = 0.00267;
    lens.b = -0.00450;
    lens.c = 0.01470;
    lens.k = 1.015;
    lens.vig_k1 = -0.1619;
    lens.vig_k2 = 0.5330;
    lens.vig_k3 = -0.7537;
    lens.has_vig = true;
    let stop = Arc::new(AtomicBool::new(false));
    let log = StitchLog::create(Path::new("/home/dalibor/tmp/same_decode/stages_rust.log"), stop).expect("log");
    let _guard = trace::install(log);
    for (frame, gain) in frames.iter_mut().zip(gains) {
        for p in frame.rgb.iter_mut() {
            *p *= gain;
        }
        let mean = frame.rgb.iter().sum::<f32>() / frame.rgb.len() as f32;
        trace::line(&format!("gain {} {mean:.4}", frame.name));
        warp::devignette(&mut frame.rgb, frame.width, frame.height, &lens);
        let mean = frame.rgb.iter().sum::<f32>() / frame.rgb.len() as f32;
        trace::line(&format!("vig {} {mean:.4}", frame.name));
    }
    lens.has_vig = false;
    let drawn = stitch::stitch(&frames, lens, 94.0, false, &[], 1.529, false, &|_| {}).expect("stitch");
    for (i, name) in drawn.names.iter().enumerate() {
        let (yaw, pitch) = yaw_pitch_deg(&drawn.rotations[i]);
        trace::line(&format!("pose {name} yaw={yaw:+.2} pitch={pitch:+.2}"));
    }
    trace::line(&format!("f_px={:.1}", drawn.focal_px));
    let (geom, used) = warp::canvas_geom(&drawn.rotations, drawn.focal_px, frames[0].width, frames[0].height, &lens, Projection::Perspective);
    trace::line(&format!("full canvas {} {}x{}", used.as_str(), geom.width, geom.height));
    let mut hold: Vec<Option<warp::Placed>> = vec![None, None];
    for (i, frame) in frames.iter().enumerate() {
        let placed = warp::warp_one(&frame.rgb, frame.width, frame.height, &drawn.rotations[i], drawn.focal_px, &lens, &geom);
        trace::line(&format!(
            "full warp {} bbox=({},{},{},{}) {}x{}",
            frame.name,
            placed.x0,
            placed.y0,
            placed.x0 + placed.w as i32,
            placed.y0 + placed.h as i32,
            placed.w,
            placed.h
        ));
        if frame.name.starts_with("DSCF8718") {
            hold[0] = Some(placed);
        } else if frame.name.starts_with("DSCF8719") {
            hold[1] = Some(placed);
        }
    }
    if let (Some(a), Some(b)) = (hold[0].as_ref(), hold[1].as_ref()) {
        for (name, cx, cy) in [("mid", 8316, 4200), ("low", 8316, 5000), ("high", 8316, 2500)] {
            let (dx, dy, sad, zero) = overlap_shift(a, b, cx, cy);
            trace::line(&format!("overlap {name} dx={dx} dy={dy} sad={sad:.5} zero={zero:.5}"));
        }
    }
    assert!(drawn.width > 1000, "{}", drawn.width);
}

// Renders the full-size panorama from the prepared frames and writes the linear picture.
#[test]
#[ignore]
fn same_pixels_full() {
    let dir = "/home/dalibor/tmp/same_decode";
    let names = ["DSCF8715", "DSCF8716", "DSCF8717", "DSCF8718", "DSCF8719", "DSCF8720", "DSCF8721"];
    let gains = [0.45_f32, 0.75, 0.85, 1.0, 1.2, 1.5, 1.6];
    let mut frames = Vec::new();
    for name in names {
        let (rgb, w, h) = read_prepared(&format!("{dir}/{name}.f32"));
        frames.push(InputFrame { name: format!("{name}.RAF"), width: w, height: h, rgb });
    }
    let mut lens = LensModel::identity();
    lens.kind = LensKind::PtLens;
    lens.a = 0.00267;
    lens.b = -0.00450;
    lens.c = 0.01470;
    lens.k = 1.015;
    lens.vig_k1 = -0.1619;
    lens.vig_k2 = 0.5330;
    lens.vig_k3 = -0.7537;
    lens.has_vig = true;
    let stop = Arc::new(AtomicBool::new(false));
    let log = StitchLog::create(Path::new("/home/dalibor/tmp/same_decode/full_rust.log"), stop).expect("log");
    let _guard = trace::install(log);
    for (frame, gain) in frames.iter_mut().zip(gains) {
        for p in frame.rgb.iter_mut() {
            *p *= gain;
        }
        warp::devignette(&mut frame.rgb, frame.width, frame.height, &lens);
    }
    lens.has_vig = false;
    let drawn = stitch::stitch(&frames, lens, 94.0, false, &[], 1.529, false, &|_| {}).expect("stitch");
    let rots = drawn.rotations.clone();
    let kept = drawn.kept.clone();
    let focal = drawn.focal_px;
    let rendered = stitch::compose(&frames, &rots, &kept, lens, focal, 94.0, Projection::Perspective, false, &|_| {}).expect("compose");
    let w = rendered.width;
    let h = rendered.height;
    let x0 = (rendered.crop_x * w as f64).round().clamp(0.0, w as f64) as u32;
    let y0 = (rendered.crop_y * h as f64).round().clamp(0.0, h as f64) as u32;
    let x1 = ((rendered.crop_x + rendered.crop_w) * w as f64).round().clamp(x0 as f64 + 1.0, w as f64) as u32;
    let y1 = ((rendered.crop_y + rendered.crop_h) * h as f64).round().clamp(y0 as f64 + 1.0, h as f64) as u32;
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut cropped = vec![0f32; (cw as usize) * (ch as usize) * 3];
    for y in 0..ch {
        let s = (((y0 + y) * w + x0) * 3) as usize;
        let d = ((y * cw) * 3) as usize;
        let n = (cw as usize) * 3;
        cropped[d..d + n].copy_from_slice(&rendered.rgb[s..s + n]);
    }
    write_linear(&format!("{dir}/full_linear.f32"), &cropped, cw, ch);
    trace::line(&format!("full linear {cw}x{ch}"));
}

fn write_linear(path: &str, rgb: &[f32], w: u32, h: u32) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).unwrap_or_else(|e| panic!("create {path}: {e}"));
    file.write_all(&w.to_le_bytes()).unwrap();
    file.write_all(&h.to_le_bytes()).unwrap();
    for chunk in rgb.chunks(1_048_576) {
        let mut buf = Vec::with_capacity(chunk.len() * 4);
        for v in chunk {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        file.write_all(&buf).unwrap();
    }
}

fn overlap_shift(a: &Placed, b: &Placed, cx: i32, cy: i32) -> (i32, i32, f32, f32) {
    let n = 96i32;
    let mut best = (0, 0, f32::MAX);
    let mut zero = f32::MAX;
    for dy in -24..=24 {
        for dx in -24..=24 {
            let mut sad = 0.0f32;
            let mut count = 0i32;
            for y in 0..n {
                for x in 0..n {
                    let Some(pa) = placed_luma(a, cx + x, cy + y) else { continue };
                    let Some(pb) = placed_luma(b, cx + x + dx, cy + y + dy) else { continue };
                    sad += (pa - pb).abs();
                    count += 1;
                }
            }
            if count > n * n / 2 {
                let mean = sad / count as f32;
                if dx == 0 && dy == 0 {
                    zero = mean;
                }
                if mean < best.2 {
                    best = (dx, dy, mean);
                }
            }
        }
    }
    (best.0, best.1, best.2, zero)
}

fn placed_luma(p: &Placed, x: i32, y: i32) -> Option<f32> {
    if x < p.x0 || y < p.y0 || x >= p.x0 + p.w as i32 || y >= p.y0 + p.h as i32 {
        return None;
    }
    let lx = (x - p.x0) as usize;
    let ly = (y - p.y0) as usize;
    let i = ly * p.w + lx;
    if !p.valid[i] {
        return None;
    }
    let px = &p.img[i * 3..i * 3 + 3];
    Some(0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2])
}

fn read_prepared(path: &str) -> (Vec<f32>, u32, u32) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let w = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let h = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let rgb = bytes[8..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (rgb, w, h)
}

fn cosine_desc(a: &[f32; 128], b: &[f32; 128]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..128 {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn radial_sky_frame(w: usize, h: usize) -> Placed {
    let hd = (w as f64 * 0.5).hypot(h as f64 * 0.5);
    let mut img = vec![0f32; w * h * 3];
    let mut uv = vec![0f32; w * h * 2];
    for y in 0..h {
        for x in 0..w {
            let u = (x as f64 + 0.5 - w as f64 / 2.0) / hd;
            let v = (y as f64 + 0.5 - h as f64 / 2.0) / hd;
            let r2 = u * u + v * v;
            let i = y * w + x;
            let l = if y >= h - 20 {
                0.0
            } else if y < h / 2 {
                (0.40 * (-0.55 * r2).exp()) as f32
            } else if (x / 2 + y / 2) % 2 == 0 {
                0.15
            } else {
                0.45
            };
            img[i * 3] = l;
            img[i * 3 + 1] = l;
            img[i * 3 + 2] = l;
            uv[i * 2] = u as f32;
            uv[i * 2 + 1] = v as f32;
        }
    }
    Placed { x0: 0, y0: 0, w, h, img, valid: vec![true; w * h], uv }
}

fn block_frame(w: usize, h: usize, scale: f32) -> Placed {
    let mut img = vec![0f32; w * h * 3];
    let mut uv = vec![0f32; w * h * 2];
    for y in 0..h {
        for x in 0..w {
            let l = if y == 0 && x < 8 && x % 4 == 0 { 0.0 } else { 0.5 * scale };
            let i = y * w + x;
            img[i * 3] = l;
            img[i * 3 + 1] = l;
            img[i * 3 + 2] = l;
            uv[i * 2] = x as f32 / w as f32 - 0.5;
            uv[i * 2 + 1] = y as f32 / h as f32 - 0.5;
        }
    }
    Placed { x0: 0, y0: 0, w, h, img, valid: vec![true; w * h], uv }
}
