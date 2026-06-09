/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN (Pick-in-Partitions Nearest Neighbors) index builder.
//!
//! Implements the PiPNN algorithm from arXiv:2602.21247, which builds graph-based
//! ANN indexes significantly faster than Vamana/HNSW by:
//! 1. Partitioning the dataset into overlapping clusters via Randomized Ball Carving
//! 2. Building local graphs within each leaf cluster using GEMM-based all-pairs distance
//! 3. Merging edges from overlapping partitions using HashPrune (LSH-based online pruning)

pub mod builder;
pub mod candidate_pool;
pub mod gemm;
pub mod hash_prune;
pub mod leaf_build;
pub mod partition;
pub mod profile;
pub mod prune;
pub mod quantize;

use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during PiPNN index construction.
#[derive(Debug, Error)]
pub enum PiPNNError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("data dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error(
        "data length mismatch: expected {expected} elements ({npoints} x {ndims}), got {actual}"
    )]
    DataLengthMismatch {
        expected: usize,
        actual: usize,
        npoints: usize,
        ndims: usize,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for PiPNN operations.
pub type PiPNNResult<T> = Result<T, PiPNNError>;

/// Custom serde module for `Metric`, which does not derive Serialize/Deserialize.
/// Serializes as a string representation (e.g. "l2", "cosine").
mod metric_serde {
    use diskann_vector::distance::Metric;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(metric: &Metric, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(metric.as_str())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Metric, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<Metric>().map_err(serde::de::Error::custom)
    }
}

/// Configuration for the PiPNN index builder.
fn default_leader_cap() -> usize { 1000 }
fn default_true() -> bool { true }
fn default_merge_l_max() -> usize { 256 }

/// Cross-leaf merge strategy for the RobustPrune leaf modes (ignored for `Baseline`,
/// which always uses HashPrune). Lets leaf candidates feed either the bounded HashPrune
/// reservoir (the paper's design — memory = npoints × l_max regardless of fanout) or an
/// accumulator (keeps more candidates at higher memory). `Accumulate` is the default so
/// existing RobustPrune-merge configs keep their behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MergeMode {
    /// Stream leaf candidates into the bounded HashPrune reservoir, then optional final prune.
    HashPrune,
    /// Accumulate per source (bounded by `merge_l_max`, 0 = unbounded), then final RobustPrune.
    #[default]
    Accumulate,
}

/// Leaf-build + cross-leaf merge strategy.
///
/// `Baseline` is the production path (GEMM top-k leaf k-NN → HashPrune reservoir merge).
/// The two RobustPrune variants (experimental) replace HashPrune entirely: each leaf
/// runs Vamana's RobustPrune per point, the pruned edges accumulate per source
/// ([`candidate_pool`]), and a single final RobustPrune ([`prune`]) produces the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LeafPruneMode {
    /// Production: GEMM top-k leaf k-NN → HashPrune reservoir merge.
    #[default]
    Baseline,
    /// Exp 1: all-to-all RobustPrune in leaf (no GEMM) → accumulate → final RobustPrune.
    RobustNoGemm,
    /// Exp 2: GEMM top-`k` (k ≫ 2) → RobustPrune in leaf → accumulate → final RobustPrune.
    GemmTopKRobust,
    /// Bi-directed k-NN leaf (identical to `Baseline`) but routable to either merge via
    /// `merge_mode` — so the SAME leaf candidates can feed HashPrune vs accumulate+RobustPrune
    /// for a clean iso-leaf, iso-output-degree merge comparison.
    KnnBidir,
    /// Control (paper's bi-directed-kNN style): GEMM top-`k`, NO leaf prune → accumulate →
    /// single final RobustPrune. Isolates whether RobustPrune *in the leaf* hurts (the paper
    /// claims it produces excessively dense candidate lists).
    GemmTopKNoPrune,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiPNNConfig {
    /// Number of LSH hyperplanes for HashPrune.
    pub num_hash_planes: usize,
    /// Maximum leaf partition size.
    pub c_max: usize,
    /// Minimum cluster size before merging.
    pub c_min: usize,
    /// Sampling fraction for RBC leaders.
    pub p_samp: f64,
    /// Fanout at each partitioning level (overlap factor).
    pub fanout: Vec<usize>,
    /// k for k-NN in leaf building.
    pub k: usize,
    /// Maximum graph degree (R).
    pub max_degree: usize,
    /// Number of independent partitioning passes (replicas).
    pub replicas: usize,
    /// Maximum reservoir size per node in HashPrune.
    pub l_max: usize,
    /// Distance metric.
    #[serde(with = "metric_serde")]
    pub metric: Metric,
    /// Whether to apply a final diversity-prune pass (occlusion-based, similar to RobustPrune).
    pub final_prune: bool,
    /// Alpha (occlusion factor) for final diversity prune. Same as DiskANN's `alpha` parameter.
    /// Higher values yield sparser graphs. Default: 1.2 (matches DiskANN default).
    pub alpha: f32,
    /// Number of threads to use. 0 means use all available cores.
    #[serde(default)]
    pub num_threads: usize,
    /// Maximum leaders per partition level. Default: 1000 (paper recommendation).
    #[serde(default = "default_leader_cap")]
    pub leader_cap: usize,
    /// Whether to saturate after final prune (fill remaining degree slots with
    /// closest non-selected candidates). Default: true.
    #[serde(default = "default_true")]
    pub saturate_after_prune: bool,
    /// Leaf-build + merge strategy (experiment selector). Default: `Baseline`.
    #[serde(default)]
    pub leaf_prune_mode: LeafPruneMode,
    /// For the RobustPrune merge modes: per-source candidate accumulator cap
    /// (closest-by-distance, deduped). 0 = unbounded (faithful "keep all candidates",
    /// memory-heavy — run behind the RSS watchdog). Default: 256.
    #[serde(default = "default_merge_l_max")]
    pub merge_l_max: usize,
    /// Merge strategy for RobustPrune leaf modes: feed bounded HashPrune (paper design) or
    /// accumulate. Ignored for `Baseline`. Default: `Accumulate`.
    #[serde(default)]
    pub merge_mode: MergeMode,
    /// Per-leaf RobustPrune target degree (candidates kept per point in each leaf before
    /// merge). 0 = use `max_degree`. Lets you keep e.g. 128 leaf candidates feeding the merge.
    #[serde(default)]
    pub leaf_prune_degree: usize,
}

impl PiPNNConfig {
    /// Validate the configuration, returning an error if any parameter is invalid.
    pub fn validate(&self) -> PiPNNResult<()> {
        if self.c_max == 0 {
            return Err(PiPNNError::Config("c_max must be > 0".into()));
        }
        if self.c_min == 0 {
            return Err(PiPNNError::Config("c_min must be > 0".into()));
        }
        if self.c_min > self.c_max {
            return Err(PiPNNError::Config(format!(
                "c_min ({}) must be <= c_max ({})",
                self.c_min, self.c_max
            )));
        }
        if self.max_degree == 0 {
            return Err(PiPNNError::Config("max_degree must be > 0".into()));
        }
        if self.k == 0 {
            return Err(PiPNNError::Config("k must be > 0".into()));
        }
        if self.replicas == 0 {
            return Err(PiPNNError::Config("replicas must be > 0".into()));
        }
        if self.l_max == 0 {
            return Err(PiPNNError::Config("l_max must be > 0".into()));
        }
        if !self.p_samp.is_finite() {
            return Err(PiPNNError::Config("p_samp must be finite".into()));
        }
        if self.p_samp <= 0.0 || self.p_samp > 1.0 {
            return Err(PiPNNError::Config(format!(
                "p_samp ({}) must be in (0.0, 1.0]",
                self.p_samp
            )));
        }
        if self.fanout.is_empty() {
            return Err(PiPNNError::Config("fanout must not be empty".into()));
        }
        if self.fanout.contains(&0) {
            return Err(PiPNNError::Config("all fanout values must be > 0".into()));
        }
        if self.num_hash_planes == 0 || self.num_hash_planes > 16 {
            return Err(PiPNNError::Config(format!(
                "num_hash_planes ({}) must be in [1, 16]",
                self.num_hash_planes
            )));
        }
        if self.alpha < 1.0 {
            return Err(PiPNNError::Config(format!(
                "alpha ({}) must be >= 1.0",
                self.alpha
            )));
        }
        if !self.alpha.is_finite() {
            return Err(PiPNNError::Config("alpha must be finite".into()));
        }
        if self.metric == Metric::InnerProduct {
            return Err(PiPNNError::Config(
                "InnerProduct metric is not supported by PiPNN; use L2, Cosine, or CosineNormalized".into(),
            ));
        }
        Ok(())
    }
}

impl Default for PiPNNConfig {
    fn default() -> Self {
        Self {
            num_hash_planes: 12,
            c_max: 1024,
            c_min: 256,
            p_samp: 0.005,
            fanout: vec![10, 3],
            k: 3,
            max_degree: 64,
            replicas: 1,
            l_max: 128,
            metric: Metric::L2,
            final_prune: false,
            alpha: 1.2,
            num_threads: 0,
            leader_cap: 1000,
            saturate_after_prune: true,
            leaf_prune_mode: LeafPruneMode::Baseline,
            merge_l_max: 256,
            merge_mode: MergeMode::Accumulate,
            leaf_prune_degree: 0,
        }
    }
}
