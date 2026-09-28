#[derive(Debug, Clone, Copy)]
pub struct LensModel {
    pub kind: LensKind,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub k: f64,
    pub vig_k1: f64,
    pub vig_k2: f64,
    pub vig_k3: f64,
    pub has_vig: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensKind {
    Identity,
    PtLens,
    Poly3,
}

impl Default for LensModel {
    fn default() -> Self {
        Self::identity()
    }
}

impl LensModel {
    pub fn identity() -> Self {
        Self {
            kind: LensKind::Identity,
            a: 0.0,
            b: 0.0,
            c: 0.0,
            k: 1.0,
            vig_k1: 0.0,
            vig_k2: 0.0,
            vig_k3: 0.0,
            has_vig: false,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.kind == LensKind::Identity || (self.a.abs() + self.b.abs() + self.c.abs()) < 1e-9
    }

    pub fn forward_scale(&self, ru: f64) -> f64 {
        let ru = ru * self.k;
        let r2 = ru * ru;
        match self.kind {
            LensKind::PtLens => {
                self.a * ru * r2 + self.b * r2 + self.c * ru + (1.0 - self.a - self.b - self.c)
            }
            LensKind::Poly3 => 1.0 - self.a + self.a * r2,
            LensKind::Identity => 1.0,
        }
    }

    pub fn invert_radius(&self, rd: f64) -> f64 {
        if self.is_identity() {
            return rd;
        }
        let mut ru = rd;
        for _ in 0..12 {
            let s = self.forward_scale(ru);
            let err = ru * s - rd;
            let s2 = self.forward_scale(ru + 1e-6);
            let deriv = (s + ru * (s2 - s) / 1e-6).max(1e-6);
            ru = (ru - err / deriv).max(0.0);
        }
        ru
    }
}

pub fn scale_k(calibration_crop: f64, camera_crop: f64) -> f64 {
    if camera_crop.abs() < 1e-9 {
        1.0
    } else {
        calibration_crop / camera_crop
    }
}
