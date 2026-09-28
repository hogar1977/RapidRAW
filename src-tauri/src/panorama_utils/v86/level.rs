use nalgebra::{Matrix3, Vector3};

// Turns the whole group upright so the horizon sits level.
pub fn level_poses(rots: &[Matrix3<f64>]) -> Vec<Matrix3<f64>> {
    let x = nelder_mead(rots);
    let l = nalgebra::Rotation3::from_scaled_axis(Vector3::new(x[0], x[1], x[2])).into_inner();
    rots.iter().map(|r| l * r).collect()
}

fn yxz(rot: &Matrix3<f64>) -> (f64, f64, f64) {
    let pitch = (-rot[(1, 2)]).clamp(-1.0, 1.0).asin();
    let roll = rot[(1, 0)].atan2(rot[(1, 1)]);
    let yaw = rot[(0, 2)].atan2(rot[(2, 2)]);
    (yaw, pitch, roll)
}

fn cost(x: &[f64; 3], rots: &[Matrix3<f64>]) -> f64 {
    let l = nalgebra::Rotation3::from_scaled_axis(Vector3::new(x[0], x[1], x[2])).into_inner();
    let mut s = 0.0;
    for r in rots {
        let (_, pitch, roll) = yxz(&(l * r));
        s += pitch * pitch + super::const_::LEVEL_ROLL_W * roll * roll;
    }
    s
}

fn nelder_mead(rots: &[Matrix3<f64>]) -> [f64; 3] {
    let n = 3usize;
    let mut sim = [[0.0f64; 3]; 4];
    sim[1][0] = 0.00025;
    sim[2][1] = 0.00025;
    sim[3][2] = 0.00025;
    let mut f = [0.0; 4];
    for i in 0..4 {
        f[i] = cost(&sim[i], rots);
    }
    let max_iter = n * 200;
    for _ in 0..max_iter {
        let mut order = [0usize, 1, 2, 3];
        order.sort_by(|&a, &b| f[a].partial_cmp(&f[b]).unwrap_or(std::cmp::Ordering::Equal));
        let best = order[0];
        let worst = order[3];
        let second = order[2];
        let mut span = 0.0f64;
        for i in 0..3 {
            for j in 0..4 {
                span = span.max((sim[j][i] - sim[best][i]).abs());
            }
        }
        let fspan = (f[worst] - f[best]).abs();
        if span <= super::const_::LEVEL_XATOL && fspan <= super::const_::LEVEL_FATOL {
            return sim[best];
        }
        let mut cen = [0.0; 3];
        for i in 0..3 {
            cen[i] = sim[order[i]][i];
        }
        for k in 0..3 {
            let mut s = 0.0;
            for i in 0..n {
                s += sim[order[i]][k];
            }
            cen[k] = s / n as f64;
        }
        let mut xr = [0.0; 3];
        for k in 0..3 {
            xr[k] = cen[k] + (cen[k] - sim[worst][k]);
        }
        let fr = cost(&xr, rots);
        if fr < f[best] {
            let mut xe = [0.0; 3];
            for k in 0..3 {
                xe[k] = cen[k] + 2.0 * (xr[k] - cen[k]);
            }
            let fe = cost(&xe, rots);
            if fe < fr {
                sim[worst] = xe;
                f[worst] = fe;
            } else {
                sim[worst] = xr;
                f[worst] = fr;
            }
        } else if fr < f[second] {
            sim[worst] = xr;
            f[worst] = fr;
        } else {
            let mut xc = [0.0; 3];
            if fr < f[worst] {
                for k in 0..3 {
                    xc[k] = cen[k] + 0.5 * (xr[k] - cen[k]);
                }
            } else {
                for k in 0..3 {
                    xc[k] = cen[k] + 0.5 * (sim[worst][k] - cen[k]);
                }
            }
            let fc = cost(&xc, rots);
            if fc < f[worst].min(fr) {
                sim[worst] = xc;
                f[worst] = fc;
            } else {
                for i in 0..4 {
                    if i == best {
                        continue;
                    }
                    for k in 0..3 {
                        sim[i][k] = sim[best][k] + 0.5 * (sim[i][k] - sim[best][k]);
                    }
                    f[i] = cost(&sim[i], rots);
                }
            }
        }
    }
    let mut best_i = 0usize;
    for i in 1..4 {
        if f[i] < f[best_i] {
            best_i = i;
        }
    }
    sim[best_i]
}
