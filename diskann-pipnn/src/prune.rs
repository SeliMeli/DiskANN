/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Faithful port of DiskANN/Vamana's `occlude_list` RobustPrune (see
//! `diskann/src/graph/index.rs::occlude_list`) as a standalone function operating
//! on a candidate list, for use by the experimental PiPNN leaf-prune + merge flows
//! (`LeafPruneMode::RobustNoGemm` / `GemmTopKRobust`).
//!
//! Algorithm (iterative-alpha greedy diversification, non-MIPS / L2-cosine form):
//! 1. Sort candidates ascending by distance-to-node.
//! 2. `current_alpha` starts at 1.0 and grows by `min(alpha, 1.2)` each round until
//!    it reaches `alpha` or the degree budget is filled.
//! 3. In each round, walk candidates in sorted order; a candidate `q` is occluded by
//!    an already-selected `p` when `occlude_factor(q) = max_p dist(node,q)/dist(p,q)`
//!    exceeds `current_alpha` (i.e. `current_alpha * dist(p,q) < dist(node,q)`).
//! 4. Optional saturation: fill remaining degree slots with the closest unselected.
//!
//! This is the same `update_occlude_factor` ratio DiskANN uses for Euclidean/cosine
//! prune kinds (PiPNN rejects InnerProduct, so the MIPS branch is not needed).

use diskann::utils::VectorRepr;
use diskann_vector::distance::{Distance, DistanceProvider, Metric};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Run DiskANN's iterative-alpha `occlude_list` over a single node's candidate set.
///
/// - `cand_vecs`: `nc * ndims` row-major f32 candidate vectors (row `i` = candidate `i`).
/// - `node_dists`: `nc` distances from the node being pruned to each candidate (any order;
///   sorted internally).
/// - `dist_fn`: dimension-specialized f32 distance for candidate-candidate comparisons.
///
/// Returns the selected **local candidate indices** (`0..nc`) in selection order,
/// `len <= min(max_degree, nc)`.
#[allow(clippy::too_many_arguments)]
pub fn robust_prune_candidates(
    cand_vecs: &[f32],
    node_dists: &[f32],
    nc: usize,
    ndims: usize,
    dist_fn: &Distance<f32, f32>,
    max_degree: usize,
    alpha: f32,
    saturate: bool,
) -> Vec<u32> {
    robust_prune_occlude(nc, node_dists, max_degree, alpha, saturate, |qi, pi| {
        dist_fn.call(
            &cand_vecs[qi * ndims..(qi + 1) * ndims],
            &cand_vecs[pi * ndims..(pi + 1) * ndims],
        )
    })
}

/// Generic iterative-alpha `occlude_list` core. `node_dists[i]` is the distance from
/// the node being pruned to candidate `i`; `pair_dist(i, j)` returns the distance
/// between candidates `i` and `j` (local indices into `0..nc`). The candidate-candidate
/// distance source is abstracted so callers can supply either on-the-fly SIMD distance
/// (`robust_prune_candidates`) or a precomputed leaf distance matrix (Exp 1, no GEMM).
/// Returns selected local candidate indices in selection order, `len <= min(max_degree, nc)`.
pub fn robust_prune_occlude(
    nc: usize,
    node_dists: &[f32],
    max_degree: usize,
    alpha: f32,
    saturate: bool,
    pair_dist: impl Fn(usize, usize) -> f32,
) -> Vec<u32> {
    if nc == 0 {
        return Vec::new();
    }
    let degree = max_degree.min(nc);

    // `order[s]` = local candidate index at sorted position `s` (ascending node distance).
    let mut order: Vec<u32> = (0..nc as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        node_dists[a as usize]
            .partial_cmp(&node_dists[b as usize])
            .unwrap_or(Ordering::Equal)
    });

    // Indexed by sorted position.
    let mut occlude = vec![0.0f32; nc];
    let mut last_checked = vec![0u32; nc];
    let mut is_selected = vec![false; nc];
    // `selected` stores sorted positions, in selection order.
    let mut selected: Vec<u32> = Vec::with_capacity(degree);

    let mut cur_alpha = 1.0f32;
    let increment = alpha.min(1.2);

    loop {
        let mut s = 0usize;
        while s < nc && selected.len() < degree {
            if occlude[s] > cur_alpha {
                s += 1;
                continue;
            }
            // Compare against already-selected results that appear earlier in sorted
            // order, resuming from where we left off in a prior alpha round.
            let mut skip = false;
            while (last_checked[s] as usize) != selected.len() {
                let rc = last_checked[s] as usize;
                last_checked[s] += 1;
                let sel_pos = selected[rc] as usize;
                // The selected result must appear before this candidate in sorted order.
                if sel_pos >= s {
                    continue;
                }
                let qi = order[s] as usize;
                let pi = order[sel_pos] as usize;
                let d_pq = pair_dist(qi, pi);
                let ratio = if d_pq > 0.0 {
                    node_dists[qi] / d_pq
                } else {
                    f32::MAX
                };
                if ratio > occlude[s] {
                    occlude[s] = ratio;
                }
                if occlude[s] > cur_alpha {
                    skip = true;
                    break;
                }
            }
            if skip || occlude[s] > cur_alpha {
                s += 1;
                continue;
            }
            occlude[s] = f32::MAX;
            is_selected[s] = true;
            selected.push(s as u32);
            s += 1;
        }

        if selected.len() >= degree || cur_alpha >= alpha {
            break;
        }
        cur_alpha = (cur_alpha * increment).min(alpha);
    }

    // Saturation: fill remaining slots with closest unselected candidates.
    if saturate && selected.len() < degree {
        for s in 0..nc {
            if selected.len() >= degree {
                break;
            }
            if !is_selected[s] {
                is_selected[s] = true;
                selected.push(s as u32);
            }
        }
    }

    selected.into_iter().map(|s| order[s as usize]).collect()
}

/// Parallel final RobustPrune over per-node candidate lists, mirroring
/// `builder::final_prune_from_candidates` but using the faithful iterative-alpha
/// [`robust_prune_candidates`] core. Distances are recomputed fresh from f32 data
/// (the stored candidate distances are ignored). Returns adjacency lists.
// Callers run inside an installed rayon pool.
#[allow(clippy::disallowed_methods)]
pub fn final_robust_prune<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    candidates_per_node: &[Vec<(u32, f32)>],
    max_degree: usize,
    metric: Metric,
    alpha: f32,
    saturate: bool,
) -> Vec<Vec<u32>> {
    let dist_fn = <f32 as DistanceProvider<f32>>::distance_comparer(metric, Some(ndims));

    thread_local! {
        static BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    candidates_per_node
        .par_iter()
        .enumerate()
        .map(|(node_id, candidates)| {
            let nc = candidates.len();
            if nc == 0 {
                return Vec::new();
            }
            BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                let total = (nc + 1) * ndims;
                if buf.len() < total {
                    buf.resize(total, 0.0);
                }
                let (node_f32, cand_f32) = buf[..total].split_at_mut(ndims);
                T::as_f32_into(&data[node_id * ndims..(node_id + 1) * ndims], node_f32)
                    .expect("f32 conversion");

                let mut node_dists = vec![0.0f32; nc];
                for (ci, &(id, _)) in candidates.iter().enumerate() {
                    let dst = &mut cand_f32[ci * ndims..(ci + 1) * ndims];
                    T::as_f32_into(
                        &data[id as usize * ndims..(id as usize + 1) * ndims],
                        dst,
                    )
                    .expect("f32 conversion");
                    node_dists[ci] = dist_fn.call(node_f32, dst);
                }

                let selected = robust_prune_candidates(
                    cand_f32, &node_dists, nc, ndims, &dist_fn, max_degree, alpha, saturate,
                );
                selected
                    .into_iter()
                    .map(|local| candidates[local as usize].0)
                    .collect()
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(ndims: usize) -> Distance<f32, f32> {
        <f32 as DistanceProvider<f32>>::distance_comparer(Metric::L2, Some(ndims))
    }

    #[test]
    fn test_robust_prune_occludes_near_duplicate_keeps_orthogonal() {
        // node at origin (0,0):
        //  A=(1.0, 0)    dist 1.0     — selected first
        //  B=(1.05, 0)   dist 1.1025  — near-duplicate of A; dist(A,B)²≈0.0025,
        //                               ratio 1.1025/0.0025 ≫ 1 → occluded at alpha=1.0
        //  C=(0, 3.0)    dist 9.0     — orthogonal; dist(A,C)²=10 > node dist 9,
        //                               ratio 9/10=0.9 ≤ 1.0 → NOT occluded → kept
        let ndims = 2;
        let cand_vecs = vec![1.0, 0.0, 1.05, 0.0, 0.0, 3.0];
        let node_dists = vec![1.0f32, 1.1025, 9.0];
        let sel = robust_prune_candidates(&cand_vecs, &node_dists, 3, ndims, &dist(ndims), 64, 1.0, false);
        assert!(sel.contains(&0), "closest should be selected");
        assert!(!sel.contains(&1), "near-duplicate of A should be occluded at alpha=1.0");
        assert!(sel.contains(&2), "orthogonal candidate (closer to node than to A) should survive");
    }

    #[test]
    fn test_robust_prune_respects_max_degree() {
        let ndims = 1;
        // 5 collinear candidates; occlusion keeps few, saturation fills to the cap.
        // max_degree 3 → exactly 3 kept (and never more), closest first.
        let cand_vecs = vec![1.0, 5.0, 9.0, 13.0, 17.0];
        let node_dists = vec![1.0f32, 25.0, 81.0, 169.0, 289.0];
        let sel = robust_prune_candidates(&cand_vecs, &node_dists, 5, ndims, &dist(ndims), 3, 1.2, true);
        assert_eq!(sel.len(), 3, "saturation fills exactly max_degree, never exceeds");
        assert_eq!(sel[0], 0, "closest selected first");
        // No duplicate local indices.
        let mut s = sel.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), sel.len(), "selected indices must be unique");
    }

    #[test]
    fn test_robust_prune_saturation_fills_degree() {
        // All candidates mutually occluding (collinear, tightly packed) so occlusion
        // alone keeps ~1; saturation must fill up to max_degree.
        let ndims = 1;
        let cand_vecs = vec![1.0, 2.0, 3.0, 4.0];
        let node_dists = vec![1.0f32, 4.0, 9.0, 16.0];
        let sel = robust_prune_candidates(&cand_vecs, &node_dists, 4, ndims, &dist(ndims), 4, 1.2, true);
        assert_eq!(sel.len(), 4, "saturation should fill all 4 slots");
    }

    #[test]
    fn test_robust_prune_empty() {
        let sel = robust_prune_candidates(&[], &[], 0, 4, &dist(4), 8, 1.2, true);
        assert!(sel.is_empty());
    }
}
