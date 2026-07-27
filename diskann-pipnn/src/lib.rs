/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN graph construction.

pub mod leaf;
pub mod partition;
mod leaf_build;
mod partitioning;

use diskann::{graph::Config, ANNError, ANNResult};
use diskann_vector::distance::Metric;
use rayon::ThreadPool;

/// Configuration of PiPNN's partitioning and local-neighbor algorithm.
///
/// Graph degree, pruning policy, and alpha belong to DiskANN's graph
/// configuration and are supplied separately through [`PiPNNBuildContext`].
#[derive(Clone, Debug, PartialEq)]
pub struct PiPNNConfig {
    /// Maximum number of points in a leaf.
    pub c_max: usize,
    /// Minimum leaf size used by global small-leaf merging.
    pub c_min: usize,
    /// Fraction of a cluster sampled as partition leaders.
    pub p_samp: f64,
    /// Number of nearest leaders retained at each overlapping partition level.
    pub fanout: Vec<usize>,
    /// Number of nearest neighbors selected within each leaf.
    pub k: usize,
    /// Number of independent partition passes over the dataset.
    pub replicas: usize,
}

impl PiPNNConfig {
    fn validate(&self) -> ANNResult<()> {
        if self.c_max == 0 {
            return Err(config_error("c_max must be greater than zero"));
        }
        if self.c_min == 0 {
            return Err(config_error("c_min must be greater than zero"));
        }
        if self.c_min > self.c_max {
            return Err(config_error(format!(
                "c_min ({}) must not exceed c_max ({})",
                self.c_min, self.c_max
            )));
        }
        if !self.p_samp.is_finite() || !(0.0..=1.0).contains(&self.p_samp) || self.p_samp == 0.0 {
            return Err(config_error(format!(
                "p_samp ({}) must be finite and in (0, 1]",
                self.p_samp
            )));
        }
        if self.fanout.is_empty() {
            return Err(config_error("fanout must not be empty"));
        }
        if let Some(&fanout) = self
            .fanout
            .iter()
            .find(|&&fanout| !(1..=partition::MAX_PARTITION_FANOUT).contains(&fanout))
        {
            return Err(config_error(format!(
                "fanout ({fanout}) must be in [1, {}]",
                partition::MAX_PARTITION_FANOUT
            )));
        }
        if self.k == 0 {
            return Err(config_error("k must be greater than zero"));
        }
        if self.replicas == 0 {
            return Err(config_error("replicas must be greater than zero"));
        }
        Ok(())
    }
}

/// Validated, borrowed policy and execution context for one PiPNN graph build.
#[derive(Debug)]
pub struct PiPNNBuildContext<'a> {
    pub(crate) config: PiPNNConfig,
    pub(crate) graph: &'a Config,
    pub(crate) metric: Metric,
    pub(crate) pool: &'a ThreadPool,
}

impl<'a> PiPNNBuildContext<'a> {
    /// Validate and combine PiPNN configuration with outer graph policy.
    pub fn new(
        config: PiPNNConfig,
        graph: &'a Config,
        metric: Metric,
        pool: &'a ThreadPool,
    ) -> ANNResult<Self> {
        config.validate()?;
        if !graph.alpha().is_finite() || graph.alpha() < 1.0 {
            return Err(config_error(format!(
                "graph alpha ({}) must be finite and at least 1",
                graph.alpha()
            )));
        }
        if graph.prune_kind() != metric.into() {
            return Err(config_error(format!(
                "graph prune kind {:?} is incompatible with metric {metric:?}",
                graph.prune_kind()
            )));
        }

        Ok(Self {
            config,
            graph,
            metric,
            pool,
        })
    }
}

#[track_caller]
fn config_error(message: impl std::fmt::Display) -> ANNError {
    ANNError::log_index_config_error("PiPNN".into(), message.to_string())
}
