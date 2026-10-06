//! Shared helpers for the lean_pano_proto versions.
//!
//! Only genuinely version-independent code lives here (measurement and
//! timing).  Every pipeline stage stays in the individual `vNN` folders so
//! that each version remains a self-contained, independently runnable
//! snapshot.

pub mod metrics;
pub mod timing;

#[allow(unused_imports)]
pub use metrics::{pano_metrics, report, seam_metrics, PanoMetrics, SeamMetrics, StepMetrics};
#[allow(unused_imports)]
pub use timing::StageTimer;
