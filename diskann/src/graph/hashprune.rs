/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Self-contained LSH sketches for HashPrune-style neighbor selection during Vamana graph
//! construction (experimental — the `occlude_list` HashPrune branch).
//!
//! HashPrune selects, from a candidate pool, the closest candidate per LSH angular bucket
//! (relative to the node being pruned), up to the degree budget — a cheaper, memory-bounded
//! alternative to RobustPrune's occlusion. In Vamana this is a *prune-speed* experiment:
//! Vamana is already incremental/bounded, so HashPrune brings no memory benefit here.
//!
//! Sketches are filled incrementally: each node's sketch is computed (from its f32 vector) in
//! `insert_vector` and stored write-once in a per-id `OnceLock`, then read by `occlude_list`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::OnceLock;

/// LSH hyperplanes + per-node sketches for HashPrune selection. Sketches are write-once per id.
#[derive(Debug)]
pub struct LshState {
    num_planes: usize,
    ndims: usize,
    /// Reservoir size (candidates kept per node = degree budget).
    l_max: usize,
    /// `num_planes * ndims` random-projection hyperplanes.
    hyperplanes: Vec<f32>,
    /// Per-node sketch (`num_planes` f32 dot products), filled once at insert.
    sketches: Vec<OnceLock<Box<[f32]>>>,
}

impl LshState {
    /// Create state for up to `capacity` nodes of dimension `ndims`. `num_planes` in [1,16].
    pub fn new(num_planes: usize, ndims: usize, capacity: usize, l_max: usize, seed: u64) -> Self {
        let num_planes = num_planes.clamp(1, 16);
        let mut rng = StdRng::seed_from_u64(seed);
        let hyperplanes: Vec<f32> = (0..num_planes * ndims)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
        let sketches = (0..capacity).map(|_| OnceLock::new()).collect();
        Self {
            num_planes,
            ndims,
            l_max,
            hyperplanes,
            sketches,
        }
    }

    #[inline]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    /// Compute and store node `id`'s sketch from its f32 vector (`len == ndims`). Write-once.
    pub fn set_sketch(&self, id: usize, vec_f32: &[f32]) {
        if id >= self.sketches.len() || vec_f32.len() != self.ndims {
            return;
        }
        let mut s = vec![0.0f32; self.num_planes];
        for (j, slot) in s.iter_mut().enumerate() {
            let plane = &self.hyperplanes[j * self.ndims..(j + 1) * self.ndims];
            let mut dot = 0.0f32;
            for d in 0..self.ndims {
                dot += vec_f32[d] * plane[d];
            }
            *slot = dot;
        }
        let _ = self.sketches[id].set(s.into_boxed_slice());
    }

    /// Hash of candidate `c` relative to node `p`: sign bits of (sketch(c) − sketch(p)).
    /// Returns 0 (one shared bucket) if either sketch is missing / out of range.
    #[inline]
    pub fn relative_hash(&self, p: usize, c: usize) -> u16 {
        if p >= self.sketches.len() || c >= self.sketches.len() {
            return 0;
        }
        let ps = match self.sketches[p].get() {
            Some(s) => s,
            None => return 0,
        };
        let cs = match self.sketches[c].get() {
            Some(s) => s,
            None => return 0,
        };
        let mut h: u16 = 0;
        for j in 0..self.num_planes {
            if cs[j] - ps[j] >= 0.0 {
                h |= 1u16 << j;
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_relative_hash_self_all_ones() {
        let lsh = LshState::new(4, 2, 8, 8, 42);
        lsh.set_sketch(0, &[1.0, 0.0]);
        // self-hash: all diffs 0, 0>=0 true -> all 4 bits set.
        assert_eq!(lsh.relative_hash(0, 0), (1u16 << 4) - 1);
        assert_eq!(lsh.l_max(), 8);
    }

    #[test]
    fn test_missing_sketch_returns_zero() {
        let lsh = LshState::new(6, 3, 4, 8, 7);
        lsh.set_sketch(1, &[1.0, 2.0, 3.0]);
        // id 2 has no sketch -> 0
        assert_eq!(lsh.relative_hash(1, 2), 0);
        assert_eq!(lsh.relative_hash(1, 99), 0); // out of range
    }
}
