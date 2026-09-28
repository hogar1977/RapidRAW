/// Memory budget for the stitch: prefer MemAvailable, but never treat the machine as
/// having less than ~40% of total RAM — preview buffers already sit in the process and
/// would otherwise make a 32 GB box look like an 8 GB one.
pub fn stitch_memory_budget_bytes() -> (u64, u64) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available = sys.available_memory();
    let total = sys.total_memory();
    let floor = ((total as f64) * 0.40) as u64;
    let budget = available.max(floor);
    (budget, total)
}

pub fn available_memory_bytes() -> u64 {
    stitch_memory_budget_bytes().0
}
