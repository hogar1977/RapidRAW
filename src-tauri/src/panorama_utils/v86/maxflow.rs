struct Edge {
    to: usize,
    rev: usize,
    cap: i32,
}

// Pushes as much flow as the links allow, then reports which nodes stay connected to the source.
pub fn min_cut(n: usize, edges: &[(usize, usize, i32)], src: usize, snk: usize) -> (Vec<bool>, i32) {
    let mut g: Vec<Vec<Edge>> = (0..n).map(|_| Vec::new()).collect();
    for &(u, v, c) in edges {
        if c <= 0 {
            continue;
        }
        let ui = g[u].len();
        let vi = g[v].len();
        g[u].push(Edge { to: v, rev: vi, cap: c });
        g[v].push(Edge { to: u, rev: ui, cap: 0 });
    }
    let mut flow = 0i32;
    let mut level = vec![0i32; n];
    let mut it = vec![0usize; n];
    while bfs(&g, src, snk, &mut level) {
        if super::trace::halted() {
            break;
        }
        it.fill(0);
        loop {
            let f = dfs(&mut g, src, snk, i32::MAX, &level, &mut it);
            if f == 0 {
                break;
            }
            flow = flow.saturating_add(f);
        }
    }
    let mut seen = vec![false; n];
    let mut q = std::collections::VecDeque::from([src]);
    seen[src] = true;
    while let Some(u) = q.pop_front() {
        for e in &g[u] {
            if e.cap > 0 && !seen[e.to] {
                seen[e.to] = true;
                q.push_back(e.to);
            }
        }
    }
    (seen, flow)
}

fn bfs(g: &[Vec<Edge>], src: usize, snk: usize, level: &mut [i32]) -> bool {
    level.fill(-1);
    level[src] = 0;
    let mut q = std::collections::VecDeque::from([src]);
    while let Some(u) = q.pop_front() {
        for e in &g[u] {
            if e.cap > 0 && level[e.to] < 0 {
                level[e.to] = level[u] + 1;
                q.push_back(e.to);
            }
        }
    }
    level[snk] >= 0
}

fn dfs(g: &mut [Vec<Edge>], u: usize, snk: usize, f: i32, level: &[i32], it: &mut [usize]) -> i32 {
    if u == snk {
        return f;
    }
    while it[u] < g[u].len() {
        let ei = it[u];
        let (to, cap, rev) = {
            let e = &g[u][ei];
            (e.to, e.cap, e.rev)
        };
        if cap > 0 && level[u] < level[to] {
            let pushed = dfs(g, to, snk, f.min(cap), level, it);
            if pushed > 0 {
                g[u][ei].cap -= pushed;
                let back = g[u][ei].rev;
                g[to][back].cap += pushed;
                let _ = rev;
                return pushed;
            }
        }
        it[u] += 1;
    }
    0
}
