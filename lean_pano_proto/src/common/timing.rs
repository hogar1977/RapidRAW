//! Stage timing that the prototype logs alongside the existing stitch log, so
//! preview, CPU final and GPU final can be reported separately.

use std::collections::BTreeMap;
use std::time::Instant;

pub struct StageTimer {
    stages: BTreeMap<String, f64>,
    marks: Vec<(String, Instant)>,
    open: Option<(String, Instant)>,
}

impl Default for StageTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl StageTimer {
    pub fn new() -> Self {
        Self { stages: BTreeMap::new(), marks: Vec::new(), open: None }
    }

    /// Time a closure as a named stage.  Re-entrant calls with the same name
    /// accumulate, which is what per-seam stages want.
    pub fn stage<T>(&mut self, name: &str, f: impl FnOnce() -> T) -> T {
        if self.open.is_some() {
            // Nested stage: do not track the inner one separately.
            return f();
        }
        let start = Instant::now();
        self.open = Some((name.to_string(), start));
        let out = f();
        let (_, begin) = self.open.take().unwrap();
        let secs = begin.elapsed().as_secs_f64();
        *self.stages.entry(name.to_string()).or_insert(0.0) += secs;
        self.marks.push((name.to_string(), start));
        out
    }

    pub fn count(&mut self, key: &str, n: usize) {
        let e = self.stages.entry(key.to_string()).or_insert(0.0);
        *e += n as f64;
    }

    pub fn note(&mut self, key: &str, value: f64) {
        *self.stages.entry(key.to_string()).or_insert(0.0) += value;
    }

    pub fn set(&mut self, key: &str, value: f64) {
        self.stages.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> f64 {
        self.stages.get(key).copied().unwrap_or(0.0)
    }

    pub fn into_map(self) -> BTreeMap<String, f64> {
        self.stages
    }

    pub fn merge(&mut self, other: BTreeMap<String, f64>) {
        for (k, v) in other {
            *self.stages.entry(k).or_insert(0.0) += v;
        }
    }
}
