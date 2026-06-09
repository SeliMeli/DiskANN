/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-source candidate accumulator: a drop-in replacement for HashPrune that drops
//! the LSH bucketing/reservoir entirely. Used by the experimental RobustPrune-merge
//! flows (`LeafPruneMode::RobustNoGemm` / `GemmTopKRobust`): each leaf emits
//! already-RobustPruned edges per source point, those accumulate per source (deduped),
//! and a single final RobustPrune pass runs after all leaves.
//!
//! Two variants:
//! - [`CandidatePool`] — bounded to `l_max` per source via closest-by-distance
//!   farthest-eviction. Peak memory `npoints * l_max * 8 B` (WSL-safe). Approximates
//!   "keep all candidates" without HashPrune's LSH eviction.
//! - [`AppendOnlyPool`] — unbounded (every emitted edge retained). Faithful "keep all"
//!   but memory grows with total leaf edges; run behind the RSS watchdog.
//!
//! Both dedup by neighbor id at extraction (keep min distance) and return per-source
//! `Vec<(neighbor_id, distance)>` sorted ascending by distance — the order
//! `prune::final_robust_prune` / `final_prune_from_candidates` expect.

use parking_lot::Mutex;
use rayon::prelude::*;

use crate::leaf_build::Edge;

/// Bounded per-source pool: keeps the closest `l_max` candidates by distance.
struct BoundedPool {
    entries: Vec<(u32, f32)>,
    farthest_idx: u32,
    farthest_dist: f32,
}

impl BoundedPool {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            farthest_idx: 0,
            farthest_dist: 0.0,
        }
    }

    #[inline]
    fn insert(&mut self, neighbor: u32, distance: f32, l_max: usize) {
        if self.entries.len() < l_max {
            if self.entries.is_empty() {
                self.entries.reserve_exact(l_max);
            }
            let idx = self.entries.len() as u32;
            self.entries.push((neighbor, distance));
            if distance >= self.farthest_dist {
                self.farthest_dist = distance;
                self.farthest_idx = idx;
            }
            return;
        }
        // Full: only keep if strictly closer than the current farthest.
        if distance >= self.farthest_dist {
            return;
        }
        self.entries[self.farthest_idx as usize] = (neighbor, distance);
        self.recompute_farthest();
    }

    fn recompute_farthest(&mut self) {
        let mut max_d = f32::NEG_INFINITY;
        let mut max_i = 0u32;
        for (i, &(_, d)) in self.entries.iter().enumerate() {
            if d > max_d {
                max_d = d;
                max_i = i as u32;
            }
        }
        self.farthest_dist = max_d;
        self.farthest_idx = max_i;
    }
}

/// Dedup `(id, dist)` by id (keep min distance), then sort ascending by distance.
fn dedup_sorted(mut entries: Vec<(u32, f32)>) -> Vec<(u32, f32)> {
    if entries.is_empty() {
        return entries;
    }
    entries.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    entries.dedup_by_key(|e| e.0);
    entries.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    entries
}

/// Bounded accumulator (closest `l_max` per source).
pub struct CandidatePool {
    npoints: usize,
    l_max: usize,
    pools: Vec<Mutex<BoundedPool>>,
}

impl CandidatePool {
    pub fn new(npoints: usize, l_max: usize) -> Self {
        let pools = (0..npoints).map(|_| Mutex::new(BoundedPool::new())).collect();
        Self {
            npoints,
            l_max,
            pools,
        }
    }

    /// Accumulate a batch of (already-pruned) leaf edges. Caches the last source's
    /// lock to avoid re-locking for consecutive same-source edges (mirrors
    /// `HashPrune::add_edges_batched`).
    pub fn add_edges_batched(&self, edges: &[Edge]) {
        let mut last_src = usize::MAX;
        let mut guard: Option<parking_lot::MutexGuard<'_, BoundedPool>> = None;
        for e in edges {
            debug_assert!(e.src < self.npoints, "src out of range");
            if e.src != last_src {
                drop(guard.take());
                last_src = e.src;
                guard = Some(self.pools[e.src].lock());
            }
            guard
                .as_mut()
                .unwrap()
                .insert(e.dst as u32, e.distance, self.l_max);
        }
    }

    // Callers run inside an installed rayon pool.
    #[allow(clippy::disallowed_methods)]
    pub fn extract_dedupped_sorted(self) -> Vec<Vec<(u32, f32)>> {
        self.pools
            .into_par_iter()
            .map(|m| dedup_sorted(m.into_inner().entries))
            .collect()
    }
}

/// Unbounded accumulator (retains every emitted edge; dedup at extraction).
pub struct AppendOnlyPool {
    npoints: usize,
    pools: Vec<Mutex<Vec<(u32, f32)>>>,
}

impl AppendOnlyPool {
    pub fn new(npoints: usize) -> Self {
        let pools = (0..npoints).map(|_| Mutex::new(Vec::new())).collect();
        Self { npoints, pools }
    }

    pub fn add_edges_batched(&self, edges: &[Edge]) {
        let mut last_src = usize::MAX;
        let mut guard: Option<parking_lot::MutexGuard<'_, Vec<(u32, f32)>>> = None;
        for e in edges {
            debug_assert!(e.src < self.npoints, "src out of range");
            if e.src != last_src {
                drop(guard.take());
                last_src = e.src;
                guard = Some(self.pools[e.src].lock());
            }
            guard.as_mut().unwrap().push((e.dst as u32, e.distance));
        }
    }

    // Callers run inside an installed rayon pool.
    #[allow(clippy::disallowed_methods)]
    pub fn extract_dedupped_sorted(self) -> Vec<Vec<(u32, f32)>> {
        self.pools
            .into_par_iter()
            .map(|m| {
                let mut v = dedup_sorted(m.into_inner());
                v.shrink_to_fit();
                v
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(src: usize, dst: usize, d: f32) -> Edge {
        Edge {
            src,
            dst,
            distance: d,
        }
    }

    #[test]
    fn test_candidate_pool_dedup_keeps_min() {
        let pool = CandidatePool::new(8, 16);
        pool.add_edges_batched(&[edge(0, 5, 2.0), edge(0, 5, 1.0), edge(0, 7, 3.0)]);
        let out = pool.extract_dedupped_sorted();
        assert_eq!(out[0], vec![(5, 1.0), (7, 3.0)], "dedup keeps min dist, sorted asc");
        assert!(out[1].is_empty());
    }

    #[test]
    fn test_candidate_pool_bounded_evicts_farthest() {
        let pool = CandidatePool::new(4, 2);
        // three distinct neighbors, l_max=2 → keep the two closest.
        pool.add_edges_batched(&[edge(1, 10, 3.0), edge(1, 11, 1.0), edge(1, 12, 2.0)]);
        let out = pool.extract_dedupped_sorted();
        assert_eq!(out[1].len(), 2);
        assert_eq!(out[1][0], (11, 1.0));
        assert_eq!(out[1][1], (12, 2.0));
        assert!(!out[1].iter().any(|&(id, _)| id == 10), "farthest evicted");
    }

    #[test]
    fn test_append_only_unbounded_dedup() {
        let pool = AppendOnlyPool::new(4);
        pool.add_edges_batched(&[edge(2, 1, 5.0), edge(2, 1, 4.0), edge(2, 3, 1.0), edge(2, 9, 2.0)]);
        let out = pool.extract_dedupped_sorted();
        assert_eq!(out[2], vec![(3, 1.0), (9, 2.0), (1, 4.0)]);
    }
}
