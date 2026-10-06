use nalgebra::Matrix3;
use rapidraw_lib::panorama_stitching::{self, PanoramaLensSetup};
use rapidraw_lib::panorama_utils::v86::ba::bundle_rotations;
use rapidraw_lib::panorama_utils::v86::const_::focal_px_from_35eq;
use rapidraw_lib::panorama_utils::v86::geom::{self, Projection};
use rapidraw_lib::panorama_utils::v86::layout;
use rapidraw_lib::panorama_utils::v86::lens::LensModel;
use rapidraw_lib::panorama_utils::v86::level::level_poses;
use rapidraw_lib::panorama_utils::v86::match_::match_all;
use rapidraw_lib::panorama_utils::v86::stitch::{self, InputFrame};
use rapidraw_lib::panorama_utils::v86::trace;
use rapidraw_lib::panorama_utils::v86::warp::canvas_geom;
use std::path::Path;

pub struct Alignment {
    pub rots: Vec<Matrix3<f64>>,
    pub kept: Vec<usize>,
    pub focal_jpeg: f64,
    pub jpeg_w: u32,
    pub jpeg_h: u32,
    pub focal35: f64,
    pub lens: LensModel,
    pub crop_factor: f64,
    pub calib: f64,
}

pub fn align_jpegs(paths: &[String], db_dir: &Path) -> Result<Alignment, String> {
    let exif = panorama_stitching::exif_of(&paths[0])?;
    let setup: PanoramaLensSetup = panorama_stitching::panorama_lens_setup(db_dir, &exif);
    let mut frames = Vec::with_capacity(paths.len());
    for path in paths {
        let (rgb8, w, h) = panorama_stitching::oriented_embedded_jpeg(path)?;
        trace::line(&format!("jpeg {} {w}x{h}", file_name(path)));
        frames.push(InputFrame { name: file_name(path), width: w, height: h, rgb: srgb8_to_linear(&rgb8) });
    }
    let jpeg_w = frames[0].width;
    let jpeg_h = frames[0].height;
    let focal_jpeg = focal_px_from_35eq(jpeg_w, jpeg_h, setup.focal35);
    let feats: Vec<_> = frames.iter().map(stitch::find_points).collect();
    trace::gate()?;
    let pairs = match_all(&feats);
    for pair in &pairs {
        if !pair.pa.is_empty() {
            trace::line(&format!("match {}<->{}: {} good", frames[pair.a].name, frames[pair.b].name, pair.pa.len()));
        }
    }
    let (rots, kept) = solve(&pairs, jpeg_w, jpeg_h, focal_jpeg, &setup.lens)?;
    trace::line(&format!(
        "aligned kept={} focal35={:.1} jpeg={jpeg_w}x{jpeg_h} f={focal_jpeg:.1}",
        kept.len(),
        setup.focal35
    ));
    Ok(Alignment {
        rots,
        kept,
        focal_jpeg,
        jpeg_w,
        jpeg_h,
        focal35: setup.focal35,
        lens: setup.lens,
        crop_factor: setup.crop_factor,
        calib: setup.calib,
    })
}

pub fn choose_projection(rots: &[Matrix3<f64>], focal: f64, w: u32, h: u32, lens: &LensModel) -> Projection {
    let layout = layout::detect_layout(rots, focal, w, h);
    let recommended = geom::recommend_projection(&layout.yaws, &layout.pitches, layout.frame_hfov, layout.frame_vfov);
    let (_, used) = canvas_geom(rots, focal, w, h, lens, recommended);
    used
}

fn solve(
    pairs: &[rapidraw_lib::panorama_utils::v86::match_::PairMatches],
    w: u32,
    h: u32,
    f: f64,
    lens: &LensModel,
) -> Result<(Vec<Matrix3<f64>>, Vec<usize>), String> {
    let n = pairs.iter().map(|p| p.a.max(p.b)).max().unwrap_or(0) + 1;
    let mut geom_pairs = Vec::new();
    let mut edges = Vec::new();
    let mut strong_pairs = Vec::new();
    for p in pairs {
        if p.pa.len() < 6 {
            continue;
        }
        let ba = geom::bearings(&p.pa, f, w, h, lens);
        let bb = geom::bearings(&p.pb, f, w, h, lens);
        let Some((rot, inl, spread, mask)) = geom::ransac_rotation(&ba, &bb, f, &p.pa, w, h, geom::default_ransac_iters()) else {
            continue;
        };
        let mut pa = Vec::new();
        let mut pb = Vec::new();
        for (i, on) in mask.iter().enumerate() {
            if *on {
                pa.push(p.pa[i]);
                pb.push(p.pb[i]);
            }
        }
        if geom::strong(inl, spread) {
            edges.push((p.a, p.b, inl));
            strong_pairs.push((p.a, p.b, inl, spread));
        }
        geom_pairs.push((p.a, p.b, rot, pa, pb, inl, spread));
    }
    let names: Vec<String> = (0..n).map(|i| i.to_string()).collect();
    let kept = geom::largest_component(&names, &strong_pairs);
    if kept.len() < 2 {
        return Err("Could not align the photos. They may not overlap enough.".into());
    }
    let tree_edges: Vec<(usize, usize, usize)> = edges.into_iter().filter(|(a, b, _)| kept.contains(a) && kept.contains(b)).collect();
    let tree = geom::spanning_tree(&kept, &tree_edges);
    let mut parent = vec![None; n];
    for i in 0..tree.parent.len().min(n) {
        parent[i] = tree.parent[i];
    }
    let rots_all = bundle_rotations(&geom_pairs, tree.root, &parent, &tree.order, w, h, f, lens);
    let leveled = {
        let sub: Vec<Matrix3<f64>> = kept.iter().map(|&i| rots_all.get(i).copied().unwrap_or(Matrix3::identity())).collect();
        level_poses(&sub)
    };
    let mut rots = vec![Matrix3::identity(); n];
    for (k, &i) in kept.iter().enumerate() {
        rots[i] = leveled[k];
    }
    Ok((rots, kept))
}

fn file_name(path: &str) -> String {
    Path::new(path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string())
}

fn srgb8_to_linear(rgb: &[u8]) -> Vec<f32> {
    rgb.iter()
        .map(|&v| {
            let c = v as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        })
        .collect()
}
