use rapidraw_lib::panorama_utils::auto_crop::max_inscribed_aabb;
use rapidraw_lib::panorama_utils::bundle_adjust::refine_two_camera_yaw;
use rapidraw_lib::panorama_utils::camera::{
    camera_roll_rad, filter_panorama_flow_matches, focal_px_from_fov, fov_rad_from_focal_mm_35eq,
    parse_focal_mm_35eq, zero_camera_roll, CameraPose,
};
use rapidraw_lib::panorama_utils::projection::{recommend_projection, Projection, ProjectionCanvas};
use rapidraw_lib::panorama_utils::ram::{estimate_panorama_bytes, max_safe_bytes};
use image::{GrayImage, Luma};
use nalgebra::{Rotation3, Vector3};
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
fn enforce_monotonic_yaws_fixes_crossing() {
    use rapidraw_lib::panorama_utils::bundle_adjust::enforce_monotonic_yaws;
    let mut yaws = vec![
        0.0,
        12.4f64.to_radians(),
        7.1f64.to_radians(),
        1.5f64.to_radians(),
        (-3.5f64).to_radians(),
    ];
    enforce_monotonic_yaws(&mut yaws, -1.0);
    for w in yaws.windows(2) {
        assert!(
            w[1] < w[0],
            "expected strictly decreasing, got {:.3} then {:.3}",
            w[0].to_degrees(),
            w[1].to_degrees()
        );
    }
}

#[test]
fn kabsch_recovers_yaw() {
    use rapidraw_lib::panorama_utils::camera::rotation_from_bearings_kabsch;
    let yaw = 25f64.to_radians();
    let r_true = Rotation3::from_axis_angle(&Vector3::y_axis(), yaw);
    let a = vec![
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.2, 0.1, 1.0).normalize(),
        Vector3::new(-0.15, -0.05, 1.0).normalize(),
        Vector3::new(0.05, 0.2, 1.0).normalize(),
    ];
    let b: Vec<_> = a.iter().map(|v| r_true * v).collect();
    let r = rotation_from_bearings_kabsch(&a, &b).expect("kabsch");
    let recovered = (r * Vector3::new(0.0, 0.0, 1.0)).x.atan2((r * Vector3::new(0.0, 0.0, 1.0)).z);
    assert!((recovered - yaw).abs() < 1e-6);
}

#[test]
fn zero_roll_preserves_look_at() {
    let mut poses = [CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1500.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        }];
    let yaw = 30f64.to_radians();
    let roll = 20f64.to_radians();
    let r = Rotation3::from_axis_angle(&Vector3::y_axis(), yaw)
        * Rotation3::from_axis_angle(&Vector3::z_axis(), roll);
    poses[0].set_rotation(r);
    let before = poses[0].rotation() * Vector3::new(0.0, 0.0, 1.0);
    zero_camera_roll(&mut poses, &[0]);
    let after = poses[0].rotation() * Vector3::new(0.0, 0.0, 1.0);
    assert!(before.angle(&after) < 1e-6);
    assert!(camera_roll_rad(&poses[0].rotation()).abs() < 1e-5);
}

#[test]
fn single_row_shares_pitch() {
    use rapidraw_lib::panorama_utils::camera::{
        enforce_single_row, is_likely_single_row, yaw_pitch_from_rotation,
    };
    let mut poses = [
        CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1500.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        },
        CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1500.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        },
        CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1500.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        },
    ];
    poses[0].set_rotation(
        Rotation3::from_axis_angle(&Vector3::y_axis(), -0.3)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), 0.02),
    );
    poses[1].set_rotation(
        Rotation3::from_axis_angle(&Vector3::y_axis(), 0.0)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), -0.03),
    );
    poses[2].set_rotation(
        Rotation3::from_axis_angle(&Vector3::y_axis(), 0.35)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), 0.04),
    );
    assert!(is_likely_single_row(&poses, &[0, 1, 2]));
    enforce_single_row(&mut poses, &[0, 1, 2]);
    let p0 = yaw_pitch_from_rotation(&poses[0].rotation()).1;
    let p1 = yaw_pitch_from_rotation(&poses[1].rotation()).1;
    let p2 = yaw_pitch_from_rotation(&poses[2].rotation()).1;
    assert!((p0 - p1).abs() < 1e-6);
    assert!((p1 - p2).abs() < 1e-6);
}

#[test]
fn multi_row_keeps_pitch_difference() {
    use rapidraw_lib::panorama_utils::camera::{
        enforce_single_row, is_likely_single_row, yaw_pitch_from_rotation,
    };
    let mut poses = [
        CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1500.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        },
        CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1500.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        },
    ];
    poses[0].set_rotation(Rotation3::from_axis_angle(&Vector3::y_axis(), 0.0));
    poses[1].set_rotation(
        Rotation3::from_axis_angle(&Vector3::x_axis(), 25f64.to_radians())
            * Rotation3::from_axis_angle(&Vector3::y_axis(), 0.2),
    );
    assert!(!is_likely_single_row(&poses, &[0, 1]));
    let before = yaw_pitch_from_rotation(&poses[1].rotation()).1;
    enforce_single_row(&mut poses, &[0, 1]);
    let after = yaw_pitch_from_rotation(&poses[1].rotation()).1;
    assert!((before - after).abs() < 1e-3, "multi-row pitch must be preserved");
}

#[test]
fn flow_filter_keeps_horizontal_pan() {
    let pts_a: Vec<(f64, f64)> = (0..20).map(|i| (100.0 + i as f64, 200.0)).collect();
    let mut pts_b: Vec<(f64, f64)> = pts_a.iter().map(|&(x, y)| (x + 80.0, y + 2.0)).collect();
    pts_b[3] = (50.0, 400.0); // outlier
    pts_b[7] = (40.0, 10.0);
    let kept = filter_panorama_flow_matches(&pts_a, &pts_b);
    assert!(kept.len() >= 16);
    assert!(!kept.contains(&3));
    assert!(!kept.contains(&7));
}

#[test]
fn projection_round_trip_spherical() {
    let poses = [CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1500.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        }];
    let canvas = ProjectionCanvas::from_poses(&poses, Projection::Spherical, 800);
    let ray = canvas.pixel_to_ray(canvas.width as f64 * 0.5, canvas.height as f64 * 0.5);
    let back = canvas.ray_to_pixel(ray).expect("round trip");
    assert!((back.0 - canvas.width as f64 * 0.5).abs() < 2.0);
    assert!((back.1 - canvas.height as f64 * 0.5).abs() < 2.0);
}

#[test]
fn projection_round_trip_cylindrical() {
    let poses = [CameraPose {
            rotation_vector: [0.0, 0.3, 0.0],
            focal_px: 1800.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        }];
    let canvas = ProjectionCanvas::from_poses(&poses, Projection::Cylindrical, 800);
    let ray = canvas.pixel_to_ray(100.0, 200.0);
    assert!((ray.norm() - 1.0).abs() < 1e-6);
}

#[test]
fn recommend_projection_heuristic() {
    let narrow = refine_two_camera_yaw(20f64.to_radians(), 2000.0, 2000, 1500);
    assert_eq!(recommend_projection(&narrow), Projection::Perspective);

    let wide = refine_two_camera_yaw(90f64.to_radians(), 2000.0, 2000, 1500);
    assert_eq!(recommend_projection(&wide), Projection::Cylindrical);

    let mut multi_row = [
        CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1800.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        },
        CameraPose {
            rotation_vector: [0.0, 0.0, 0.0],
            focal_px: 1800.0,
            width: 2000,
            height: 1500,
            lens: Default::default(),
        },
    ];
    let pitched = Rotation3::from_axis_angle(&Vector3::x_axis(), 25f64.to_radians())
        * Rotation3::from_axis_angle(&Vector3::y_axis(), 0.4);
    multi_row[1].set_rotation(pitched);
    assert_eq!(recommend_projection(&multi_row), Projection::Spherical);
}

#[test]
fn inscribed_aabb_on_mask() {
    let mut mask = GrayImage::new(100, 80);
    for y in 10..70 {
        for x in 20..90 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    let crop = max_inscribed_aabb(&mask);
    assert!((crop.x - 0.2).abs() < 0.02);
    assert!((crop.y - 0.125).abs() < 0.02);
    assert!((crop.width - 0.7).abs() < 0.02);
    assert!((crop.height - 0.75).abs() < 0.02);
}

#[test]
fn wave_correct_levels_horizontal_row() {
    use rapidraw_lib::panorama_utils::camera::{wave_correct, zero_camera_roll};
    let mut poses = [
        CameraPose::pinhole(1500.0, 2000, 1500),
        CameraPose::pinhole(1500.0, 2000, 1500),
        CameraPose::pinhole(1500.0, 2000, 1500),
    ];
    // Introduce a common tilt (wave) plus yaw spread.
    let tilt = Rotation3::from_axis_angle(&Vector3::x_axis(), 8f64.to_radians());
    poses[0].set_rotation(tilt * Rotation3::from_axis_angle(&Vector3::y_axis(), -0.4));
    poses[1].set_rotation(tilt * Rotation3::from_axis_angle(&Vector3::y_axis(), 0.0));
    poses[2].set_rotation(tilt * Rotation3::from_axis_angle(&Vector3::y_axis(), 0.4));
    wave_correct(&mut poses, &[0, 1, 2], true);
    zero_camera_roll(&mut poses, &[0, 1, 2]);
    for p in &poses {
        assert!(
            camera_roll_rad(&p.rotation()).abs() < 1e-3,
            "roll should be near zero after waveCorrect + zero roll"
        );
    }
    // Look directions should remain roughly coplanar / leveled (small pitch spread).
    let pitches: Vec<f64> = poses
        .iter()
        .map(|p| {
            let f = p.rotation() * Vector3::new(0.0, 0.0, 1.0);
            f.y.asin()
        })
        .collect();
    let span = pitches.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - pitches.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        span.to_degrees() < 3.0,
        "pitch span after waveCorrect should shrink, got {:.2}°",
        span.to_degrees()
    );
}

#[test]
fn ram_estimate_and_threshold() {
    let est = estimate_panorama_bytes(4000, 3000, 7, 12000, 4000);
    assert!(est > 0);
    assert_eq!(max_safe_bytes(1_000_000_000), (1_000_000_000f64 * 0.85) as u64);
    let _ = Vector3::<f64>::zeros();
}
