/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Main PiPNN builder: orchestrates partitioning, leaf building, and edge merging.
//!
//! Algorithm (from arXiv:2602.21247):
//! 1. G <- empty graph
//! 2. B <- Partition(X) via RBC
//! 3. For each leaf b_i in B (in parallel):
//!    edges <- Pick(b_i)  // GEMM + bi-directed k-NN
//!    G.Prune_And_Add_Edges(edges)  // stream to HashPrune
//! 4. Optional: final diversity prune on each node
//! 5. return G

use std::sync::atomic::Ordering;
use std::sync::OnceLock;
use std::time::Instant;

use diskann::utils::VectorRepr;
use rayon::prelude::*;

use crate::hash_prune::HashPrune;
use crate::leaf_build;
use crate::partition::PartitionConfig;
use crate::{PiPNNConfig, PiPNNError, PiPNNResult};

use diskann_vector::distance::{Distance, DistanceProvider, Metric};

/// Env switch: `PIPNN_LEAF_SORT=1` enables the leaf-order sort. Off by
/// default because the May 2026 experiment showed that sorting by min point
/// ID barely raises cache hit rate (3.5% vs 2.4% without sort on Enron 1M).
fn leaf_sort_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PIPNN_LEAF_SORT").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("on")
        )
    })
}

/// Create a DiskANN distance functor for the given metric.
///
/// Uses the exact same SIMD-accelerated distance implementations as DiskANN:
/// - `L2` → `SquaredL2` (squared euclidean)
/// - `Cosine` → `Cosine` (normalizes + 1 - dot)
/// - `CosineNormalized` → `CosineNormalized` (1 - dot, assumes pre-normalized)
/// - `InnerProduct` → `InnerProduct` (-dot)
fn make_dist_fn(metric: Metric) -> Distance<f32, f32> {
    <f32 as DistanceProvider<f32>>::distance_comparer(metric, None)
}

/// Timing breakdown for the PiPNN build phases.
#[derive(Debug, Clone, Default)]
pub struct PiPNNBuildStats {
    pub total_secs: f64,
    pub sketch_secs: f64,
    pub partition_secs: f64,
    pub leaf_build_secs: f64,
    pub extract_secs: f64,
    pub final_prune_secs: f64,
    pub num_leaves: usize,
    pub total_edges: usize,
}

impl std::fmt::Display for PiPNNBuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PiPNN Build Timing")?;
        writeln!(f, "  LSH sketches:   {:.3}s", self.sketch_secs)?;
        writeln!(
            f,
            "  Partition:      {:.3}s  ({} leaves)",
            self.partition_secs, self.num_leaves
        )?;
        writeln!(
            f,
            "  Leaf build:     {:.3}s  ({} edges)",
            self.leaf_build_secs, self.total_edges
        )?;
        writeln!(f, "  Graph extract:  {:.3}s", self.extract_secs)?;
        writeln!(f, "  Final prune:    {:.3}s", self.final_prune_secs)?;
        writeln!(f, "  Total:          {:.3}s", self.total_secs)
    }
}

/// The result of building a PiPNN index.
#[derive(Debug)]
pub struct PiPNNGraph {
    /// Adjacency lists: graph[i] contains the neighbor indices for point i.
    pub adjacency: Vec<Vec<u32>>,
    /// Number of points.
    pub npoints: usize,
    /// Number of dimensions.
    pub ndims: usize,
    /// Cached medoid (entry point for search).
    pub medoid: usize,
    /// Distance metric used to build this graph.
    pub metric: Metric,
    /// Build timing breakdown.
    pub build_stats: PiPNNBuildStats,
}

impl PiPNNGraph {
    /// Get neighbors of a point.
    pub fn neighbors(&self, idx: usize) -> &[u32] {
        &self.adjacency[idx]
    }

    /// Get the average out-degree.
    pub fn avg_degree(&self) -> f64 {
        let total: usize = self.adjacency.iter().map(|adj| adj.len()).sum();
        total as f64 / self.npoints as f64
    }

    /// Get the max out-degree.
    pub fn max_degree(&self) -> usize {
        self.adjacency
            .iter()
            .map(|adj| adj.len())
            .max()
            .unwrap_or(0)
    }

    /// Count the number of points with zero out-degree.
    pub fn num_isolated(&self) -> usize {
        self.adjacency.iter().filter(|adj| adj.is_empty()).count()
    }

    /// Save the graph in DiskANN's canonical graph format.
    ///
    /// Format:
    ///   Header (24 bytes):
    ///     - u64 LE: total file size (header + data)
    ///     - u32 LE: max degree (observed)
    ///     - u32 LE: start point ID (medoid)
    ///     - u64 LE: number of additional/frozen points
    ///   Per node:
    ///     - u32 LE: number of neighbors
    ///     - N x u32 LE: neighbor IDs
    pub fn save_graph(&self, path: &std::path::Path) -> PiPNNResult<()> {
        use std::fs::File;
        use std::io::{BufWriter, Seek, SeekFrom, Write};

        let mut f = BufWriter::new(File::create(path)?);

        let mut index_size: u64 = 24;
        let mut observed_max_degree: u32 = 0;
        let start_point = self.medoid as u32;

        // Write placeholder header
        f.write_all(&index_size.to_le_bytes())?;
        f.write_all(&observed_max_degree.to_le_bytes())?;
        f.write_all(&start_point.to_le_bytes())?;
        // Must be 1 to indicate the medoid is a frozen/start point.
        // The disk layout writer uses this to record the frozen point location.
        let num_additional: u64 = 1;
        f.write_all(&num_additional.to_le_bytes())?;

        // Write per-node adjacency lists (npoints real nodes + 1 frozen start point)
        for adj in &self.adjacency {
            let num_neighbors = adj.len() as u32;
            f.write_all(&num_neighbors.to_le_bytes())?;
            for &neighbor in adj {
                f.write_all(&neighbor.to_le_bytes())?;
            }
            observed_max_degree = observed_max_degree.max(num_neighbors);
            index_size += (4 + adj.len() * 4) as u64;
        }

        // Write the frozen start point (copy of medoid's adjacency list).
        // The header declares num_additional=1, so loaders expect exactly one
        // extra node after the npoints real nodes.
        let medoid_adj = &self.adjacency[self.medoid];
        let num_neighbors = medoid_adj.len() as u32;
        f.write_all(&num_neighbors.to_le_bytes())?;
        for &neighbor in medoid_adj {
            f.write_all(&neighbor.to_le_bytes())?;
        }
        index_size += (4 + medoid_adj.len() * 4) as u64;

        // Seek back and write correct header
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&index_size.to_le_bytes())?;
        f.write_all(&observed_max_degree.to_le_bytes())?;
        f.flush()?;

        tracing::info!(
            path = %path.display(),
            npoints = self.npoints,
            max_degree = observed_max_degree,
            start_point = start_point,
            "Saved PiPNN graph in DiskANN format"
        );

        Ok(())
    }
}

/// Search is only available for testing.
/// Production search goes through DiskANN's disk-based search pipeline.
#[cfg(test)]
impl PiPNNGraph {
    /// Perform greedy graph search starting from the cached medoid.
    ///
    /// This method is for testing and benchmarking only. Production search
    /// should use DiskANN's disk-based search pipeline which operates on the
    /// saved graph format.
    ///
    /// Returns the indices and distances of the `k` approximate nearest neighbors.
    pub fn search(
        &self,
        data: &[f32],
        query: &[f32],
        k: usize,
        search_list_size: usize,
    ) -> Vec<(usize, f32)> {
        let ndims = self.ndims;
        let npoints = self.npoints;

        if npoints == 0 {
            return Vec::new();
        }

        let dist_fn = make_dist_fn(self.metric);

        let start = self.medoid;

        // Greedy beam search.
        let l = search_list_size.max(k);
        let mut visited = vec![false; npoints];
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(l + 1);

        let start_dist = dist_fn.call(&data[start * ndims..(start + 1) * ndims], query);
        candidates.push((start, start_dist));
        visited[start] = true;

        let mut pointer = 0;

        while pointer < candidates.len() {
            let (current, _) = candidates[pointer];
            pointer += 1;

            for &neighbor in &self.adjacency[current] {
                let neighbor = neighbor as usize;
                if neighbor >= npoints || visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;

                let dist = dist_fn.call(&data[neighbor * ndims..(neighbor + 1) * ndims], query);

                if candidates.len() < l || dist < candidates.last().map(|c| c.1).unwrap_or(f32::MAX)
                {
                    let pos = candidates
                        .binary_search_by(|c| {
                            c.1.partial_cmp(&dist).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or_else(|e| e);
                    candidates.insert(pos, (neighbor, dist));
                    if candidates.len() > l {
                        candidates.truncate(l);
                    }

                    if pos < pointer {
                        pointer = pos;
                    }
                }
            }
        }

        candidates.truncate(k);
        candidates
    }
}

/// Find the approximate medoid via sampling.
///
/// Samples a subset of points to estimate the centroid, then finds the
/// nearest sampled point. For 10M+ datasets this is ~100x faster than
/// the exact scan with near-identical results.
fn find_medoid<T: VectorRepr>(data: &[T], npoints: usize, ndims: usize) -> usize {
    use rand::prelude::IndexedRandom;
    use rand::SeedableRng;

    let dist_fn = make_dist_fn(Metric::L2);

    // Sample up to 100k points for centroid estimation.
    let sample_size = npoints.min(100_000);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let all_indices: Vec<usize> = (0..npoints).collect();
    let samples: Vec<usize> = all_indices
        .choose_multiple(&mut rng, sample_size)
        .copied()
        .collect();

    // Compute centroid from samples.
    let mut centroid = vec![0.0f32; ndims];
    let mut point_buf = vec![0.0f32; ndims];
    for &i in &samples {
        T::as_f32_into(&data[i * ndims..(i + 1) * ndims], &mut point_buf).expect("f32 conversion");
        for d in 0..ndims {
            centroid[d] += point_buf[d];
        }
    }
    let inv_n = 1.0 / sample_size as f32;
    for c in &mut centroid {
        *c *= inv_n;
    }

    // Find nearest sampled point to centroid.
    let mut best_idx = samples[0];
    let mut best_dist = f32::MAX;
    for &i in &samples {
        T::as_f32_into(&data[i * ndims..(i + 1) * ndims], &mut point_buf).expect("f32 conversion");
        let dist = dist_fn.call(&point_buf, &centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    best_idx
}

/// Build a PiPNN index from typed vector data.
///
/// Keeps data in its native type T and converts to f32 on-the-fly at each access point,
/// avoiding a full f32 copy of the dataset.
/// `data` is a flat slice of `T` in row-major order: npoints x ndims.
pub fn build_typed<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
) -> PiPNNResult<PiPNNGraph> {
    config.validate()?;

    let expected_len = npoints * ndims;
    if data.len() != expected_len {
        return Err(PiPNNError::DataLengthMismatch {
            expected: expected_len,
            actual: data.len(),
            npoints,
            ndims,
        });
    }

    if npoints == 0 || ndims == 0 {
        return Err(PiPNNError::Config("npoints and ndims must be > 0".into()));
    }

    tracing::info!(
        npoints = npoints,
        ndims = ndims,
        k = config.k,
        max_degree = config.max_degree,
        c_max = config.c_max,
        replicas = config.replicas,
        "PiPNN build started (typed)"
    );

    build_internal(data, npoints, ndims, config, None)
}

/// Build a PiPNN index.
///
/// `data` is row-major: npoints x ndims.
pub fn build(
    data: &[f32],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
) -> PiPNNResult<PiPNNGraph> {
    config.validate()?;

    if npoints == 0 || ndims == 0 {
        return Err(PiPNNError::Config("npoints and ndims must be > 0".into()));
    }

    if data.len() != npoints * ndims {
        return Err(PiPNNError::DataLengthMismatch {
            expected: npoints * ndims,
            actual: data.len(),
            npoints,
            ndims,
        });
    }

    tracing::info!(
        npoints = npoints,
        ndims = ndims,
        k = config.k,
        max_degree = config.max_degree,
        c_max = config.c_max,
        replicas = config.replicas,
        "PiPNN build started"
    );

    // The build() path always builds at full precision with f32 data.
    // For quantized builds, use build_with_sq() which accepts pre-trained SQ params.
    build_internal::<f32>(data, npoints, ndims, config, None)
}

/// Pre-trained scalar quantizer parameters for 1-bit quantization.
///
/// These can be extracted from DiskANN's trained `ScalarQuantizer` to ensure
/// identical quantization between Vamana and PiPNN builds.
pub struct SQParams {
    /// Per-dimension shift (length = ndims).
    pub shift: Vec<f32>,
    /// Global inverse scale (1.0 / scale).
    pub inverse_scale: f32,
}

/// Build a PiPNN index using a pre-trained scalar quantizer for 1-bit mode.
///
/// Build a PiPNN index using pre-trained SQ parameters.
///
/// Generic over `T: VectorRepr` — works with f16, f32, u8, etc.
/// Converts T→f32 per-vector streaming during quantization and LSH sketch
/// computation, without materializing a full f32 copy of the dataset.
///
/// `data` is row-major: npoints x ndims in native type T.
pub fn build_with_sq<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    sq_params: &SQParams,
) -> PiPNNResult<PiPNNGraph> {
    config.validate()?;

    if data.len() != npoints * ndims {
        return Err(PiPNNError::DataLengthMismatch {
            expected: npoints * ndims,
            actual: data.len(),
            npoints,
            ndims,
        });
    }
    if npoints == 0 || ndims == 0 {
        return Err(PiPNNError::Config("npoints and ndims must be > 0".into()));
    }
    if config.final_prune {
        return Err(PiPNNError::Config(
            "final_prune=true is not supported for quantized builds (requires f32 data for distance recomputation)".into(),
        ));
    }
    if sq_params.shift.len() != ndims {
        return Err(PiPNNError::DimensionMismatch {
            expected: ndims,
            actual: sq_params.shift.len(),
        });
    }

    tracing::info!(
        npoints = npoints,
        ndims = ndims,
        k = config.k,
        max_degree = config.max_degree,
        "PiPNN build started (with pre-trained SQ, native type)"
    );

    // Quantize from native T (streaming T→f32 per vector, no full f32 copy).
    // Quantize and compute medoid from native data, then release the borrow.
    let t = Instant::now();
    let qdata = crate::quantize::quantize_1bit(
        data,
        npoints,
        ndims,
        &sq_params.shift,
        sq_params.inverse_scale,
    );
    let medoid = find_medoid(data, npoints, ndims);
    tracing::info!(
        elapsed_secs = t.elapsed().as_secs_f64(),
        "1-bit quantization + medoid complete"
    );
    // `data` borrow ends here — caller can drop native T data.

    // Compute LSH sketches from 1-bit vectors directly — no f16/f32 data needed.
    // dot(1bit_vec, hyperplane) = sum of hyperplane[d] where bit d is set.
    let sketches = crate::hash_prune::LshSketches::new_from_quantized(
        &qdata,
        npoints,
        ndims,
        config.num_hash_planes,
        42,
    );

    build_internal_sq(npoints, ndims, config, qdata, sketches, medoid)
}

/// Build a PiPNN index from pre-quantized data + pre-computed medoid.
///
/// Lowest-memory entry point for SQ builds: the caller quantizes and computes
/// medoid, then drops native data before calling this. Only the 1-bit quantized
/// data needs to be in memory during the graph build.
pub fn build_from_quantized(
    qdata: crate::quantize::QuantizedData,
    npoints: usize,
    ndims: usize,
    medoid: usize,
    config: &PiPNNConfig,
) -> PiPNNResult<PiPNNGraph> {
    config.validate()?;
    if npoints == 0 || ndims == 0 {
        return Err(PiPNNError::Config("npoints and ndims must be > 0".into()));
    }
    if config.final_prune {
        return Err(PiPNNError::Config(
            "final_prune=true is not supported for quantized builds (requires f32 data for distance recomputation)".into(),
        ));
    }
    // Validate consistency with QuantizedData metadata.
    if qdata.npoints() != npoints || qdata.ndims() != ndims {
        return Err(PiPNNError::DataLengthMismatch {
            expected: npoints * ndims,
            actual: qdata.npoints() * qdata.ndims(),
            npoints,
            ndims,
        });
    }

    tracing::info!(
        npoints = npoints,
        ndims = ndims,
        k = config.k,
        max_degree = config.max_degree,
        "PiPNN build from pre-quantized data"
    );

    let sketches = crate::hash_prune::LshSketches::new_from_quantized(
        &qdata,
        npoints,
        ndims,
        config.num_hash_planes,
        42,
    );

    build_internal_sq(npoints, ndims, config, qdata, sketches, medoid)
}

fn build_internal_sq(
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    qdata: crate::quantize::QuantizedData,
    sketches: crate::hash_prune::LshSketches,
    medoid: usize,
) -> PiPNNResult<PiPNNGraph> {
    let run =
        |sketches, qdata| build_internal_sq_impl(npoints, ndims, config, qdata, sketches, medoid);
    if config.num_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build()
            .map_err(|e| PiPNNError::Config(format!("Failed to create thread pool: {}", e)))?;
        return pool.install(|| run(sketches, qdata));
    }
    run(sketches, qdata)
}

// The caller (`build_internal_sq`) installs a dedicated rayon thread pool via
// `pool.install(|| ...)`, so all `par_iter` work here already executes on that pool.
#[allow(clippy::disallowed_methods)]
fn build_internal_sq_impl(
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    qdata: crate::quantize::QuantizedData,
    sketches: crate::hash_prune::LshSketches,
    medoid: usize,
) -> PiPNNResult<PiPNNGraph> {
    let t_total = Instant::now();

    let t0 = Instant::now();
    let hash_prune = HashPrune::from_sketches(sketches, npoints, config.l_max, config.max_degree);
    let sketch_secs = t0.elapsed().as_secs_f64();

    let mut partition_secs = 0.0f64;
    let mut leaf_build_secs = 0.0f64;
    let mut total_leaves = 0usize;
    let mut total_edges_count = 0usize;

    for replica in 0..config.replicas {
        let seed = 1000 + replica as u64 * 7919;
        let t1 = Instant::now();
        let partition_config = PartitionConfig {
            c_max: config.c_max,
            c_min: config.c_min,
            p_samp: config.p_samp,
            fanout: config.fanout.clone(),
            metric: config.metric,
            leader_cap: config.leader_cap,
        };
        let leaves =
            crate::partition::partition_quantized(&qdata, npoints, &partition_config, seed);
        total_leaves += leaves.len();
        partition_secs += t1.elapsed().as_secs_f64();

        // Release partition stripe buffers before leaf build.
        (0..rayon::current_num_threads())
            .into_par_iter()
            .for_each(|_| {
                crate::partition::release_stripe_buffers();
            });

        let t2 = Instant::now();
        use std::sync::atomic::{AtomicUsize, Ordering};
        let total_edges = AtomicUsize::new(0);
        leaves.par_iter().for_each(|leaf| {
            let indices_usize: Vec<usize> = leaf.indices.iter().map(|&i| i as usize).collect();
            let edges = crate::leaf_build::build_leaf_quantized(&qdata, &indices_usize, config.k);
            total_edges.fetch_add(edges.len(), Ordering::Relaxed);
            hash_prune.add_edges_batched(&edges);
        });
        total_edges_count += total_edges.load(Ordering::Relaxed);
        leaf_build_secs += t2.elapsed().as_secs_f64();
    }

    (0..rayon::current_num_threads())
        .into_par_iter()
        .for_each(|_| {
            crate::leaf_build::release_thread_buffers();
        });

    // final_prune=true is rejected at entry points (build_with_sq / build_from_quantized).
    debug_assert!(!config.final_prune, "SQ path does not support final_prune");
    let t3 = Instant::now();
    let adjacency = hash_prune.extract_graph();
    let extract_secs = t3.elapsed().as_secs_f64();
    let final_prune_secs = 0.0;

    let total_secs = t_total.elapsed().as_secs_f64();
    let stats = PiPNNBuildStats {
        sketch_secs,
        partition_secs,
        leaf_build_secs,
        extract_secs,
        final_prune_secs,
        total_secs,
        num_leaves: total_leaves,
        total_edges: total_edges_count,
    };
    print!("{}", stats);

    Ok(PiPNNGraph {
        adjacency,
        npoints,
        ndims,
        medoid,
        metric: config.metric,
        build_stats: stats,
    })
}

/// Internal build logic shared between `build()` and `build_typed()`.
fn build_internal<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    qdata: Option<crate::quantize::QuantizedData>,
) -> PiPNNResult<PiPNNGraph> {
    // Respect num_threads: install a scoped rayon pool so all par_iter() calls
    // within this build use the configured thread count instead of all cores.
    if config.num_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build()
            .map_err(|e| PiPNNError::Config(format!("Failed to create thread pool: {}", e)))?;
        return pool.install(|| build_internal_impl(data, npoints, ndims, config, qdata));
    }
    build_internal_impl(data, npoints, ndims, config, qdata)
}

// The caller (`build_internal`) installs a dedicated rayon thread pool via
// `pool.install(|| ...)`, so all `par_iter` work here already executes on that pool.
#[allow(clippy::disallowed_methods)]
fn build_internal_impl<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    qdata: Option<crate::quantize::QuantizedData>,
) -> PiPNNResult<PiPNNGraph> {
    let t_total = Instant::now();

    // Report which SIMD tier was compiled in.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    eprintln!("SIMD: AVX-512 (compile-time)");
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        target_feature = "avx2"
    ))]
    eprintln!("SIMD: AVX2 (compile-time)");
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    eprintln!("SIMD: scalar (compile-time)");
    #[cfg(not(target_arch = "x86_64"))]
    eprintln!("SIMD: scalar (non-x86)");

    // Compute medoid once upfront.
    let medoid = find_medoid(data, npoints, ndims);

    // Experimental RobustPrune merge modes (Exp 1 / Exp 2): replace HashPrune with a
    // per-leaf RobustPrune + candidate accumulation + final RobustPrune. f32 path only.
    if config.leaf_prune_mode != crate::LeafPruneMode::Baseline
        && config.merge_mode == crate::MergeMode::Accumulate
        && qdata.is_none()
    {
        return build_internal_robust(data, npoints, ndims, config, medoid);
    }
    // Otherwise: HashPrune merge (Baseline k-NN, or — when merge_mode=HashPrune — leaf
    // RobustPrune candidates streamed into the bounded reservoir; the paper's design).

    // Initialize HashPrune for edge merging.
    let t0 = Instant::now();
    let hash_prune = HashPrune::new(
        data,
        npoints,
        ndims,
        config.num_hash_planes,
        config.l_max,
        config.max_degree,
        42,
    );
    let sketch_secs = t0.elapsed().as_secs_f64();
    tracing::info!(elapsed_secs = sketch_secs, "HashPrune init complete");

    // Run multiple replicas of partitioning + leaf building.
    let mut partition_secs = 0.0f64;
    let mut leaf_build_secs = 0.0f64;
    let mut total_leaves = 0usize;
    let mut total_edges_count = 0usize;

    for replica in 0..config.replicas {
        let seed = 1000 + replica as u64 * 7919;

        let t1 = Instant::now();
        let partition_config = PartitionConfig {
            c_max: config.c_max,
            c_min: config.c_min,
            p_samp: config.p_samp,
            fanout: config.fanout.clone(),
            metric: config.metric,
            leader_cap: config.leader_cap,
        };

        let mut leaves = if let Some(ref q) = qdata {
            crate::partition::partition_quantized(q, npoints, &partition_config, seed)
        } else {
            crate::partition::partition(data, ndims, npoints, &partition_config, seed)
        };
        partition_secs += t1.elapsed().as_secs_f64();
        total_leaves += leaves.len();

        // Sort leaves so the point-cache (in LeafBuffers) gets repeated hits:
        // each point appears in ~fanout-product leaves; if those leaves are
        // adjacent in the work list, consecutive batches on the same thread
        // hit the warm cache instead of cold DRAM. Sorting by the leaf's
        // smallest point ID groups leaves drawn from the same partition
        // subtree (point IDs in a subtree cluster numerically by birth order).
        // PIPNN_LEAF_SORT=0 disables for A/B testing.
        if leaf_sort_enabled() {
            leaves.sort_unstable_by_key(|l| l.indices.iter().copied().min().unwrap_or(0));
        }

        // Release partition stripe buffers (~1 GB) before leaf build starts.
        (0..rayon::current_num_threads())
            .into_par_iter()
            .for_each(|_| {
                crate::partition::release_stripe_buffers();
            });

        // Build leaves in parallel, streaming edges to HashPrune per-leaf.
        let t2 = Instant::now();

        use std::sync::atomic::{AtomicUsize, Ordering};
        let total_edges = AtomicUsize::new(0);

        // Leaves processed in parallel via par_chunks. Each chunk shares one
        // thread-local buffer set, amortizing TLS + RefCell + Vec allocation
        // overhead across multiple leaves.
        const LEAF_BATCH: usize = 64;
        leaves.par_chunks(LEAF_BATCH).for_each(|chunk| {
            leaf_build::LEAF_BUFFERS.with(|cell| {
                let mut bufs = cell.borrow_mut();
                for leaf in chunk {
                    let indices_usize: Vec<usize> =
                        leaf.indices.iter().map(|&i| i as usize).collect();
                    // Per-leaf candidate generation. Baseline = bi-directed k-NN. The
                    // RobustPrune leaf modes (when merge_mode=HashPrune) feed their pruned
                    // candidates into the same bounded HashPrune reservoir — the paper's
                    // RobustPrune-as-candidate-feeder ablation.
                    let leaf_deg = if config.leaf_prune_degree > 0 {
                        config.leaf_prune_degree
                    } else {
                        config.max_degree
                    };
                    let edges = if let Some(ref q) = qdata {
                        leaf_build::build_leaf_quantized(q, &indices_usize, config.k)
                    } else {
                        match config.leaf_prune_mode {
                            crate::LeafPruneMode::Baseline | crate::LeafPruneMode::KnnBidir => {
                                leaf_build::build_leaf_with_buffers(
                                    data,
                                    ndims,
                                    &indices_usize,
                                    config.k,
                                    config.metric,
                                    &mut bufs,
                                )
                            }
                            crate::LeafPruneMode::RobustNoGemm => {
                                leaf_build::build_leaf_robust_no_gemm(
                                    data,
                                    ndims,
                                    &indices_usize,
                                    leaf_deg,
                                    config.metric,
                                    config.alpha,
                                    &mut bufs,
                                )
                            }
                            crate::LeafPruneMode::GemmTopKRobust
                            | crate::LeafPruneMode::GemmTopKNoPrune => {
                                let leaf_prune = matches!(
                                    config.leaf_prune_mode,
                                    crate::LeafPruneMode::GemmTopKRobust
                                );
                                leaf_build::build_leaf_gemm_topk_robust(
                                    data,
                                    ndims,
                                    &indices_usize,
                                    config.k,
                                    leaf_deg,
                                    config.metric,
                                    config.alpha,
                                    leaf_prune,
                                    &mut bufs,
                                )
                            }
                        }
                    };
                    total_edges.fetch_add(edges.len(), Ordering::Relaxed);
                    hash_prune.add_edges_batched(&edges);
                }
            });
        });

        let replica_edges = total_edges.load(Ordering::Relaxed);
        total_edges_count += replica_edges;
        leaf_build_secs += t2.elapsed().as_secs_f64();

        tracing::info!(
            replica = replica,
            elapsed_secs = t2.elapsed().as_secs_f64(),
            total_edges = replica_edges,
            "Leaf build and merge complete"
        );
    }

    // Release thread-local leaf buffers so their arena pages can be reclaimed.
    // The release path also folds each thread's hit/miss counters into the
    // global PIPNN point-cache counters, so we read them right after.
    (0..rayon::current_num_threads())
        .into_par_iter()
        .for_each(|_| {
            leaf_build::release_thread_buffers();
        });
    let pc_hits = leaf_build::POINT_CACHE_HITS.swap(0, Ordering::Relaxed);
    let pc_misses = leaf_build::POINT_CACHE_MISSES.swap(0, Ordering::Relaxed);
    if pc_hits + pc_misses > 0 {
        let total = (pc_hits + pc_misses) as f64;
        eprintln!(
            "PointCache: hits={} misses={} hit_rate={:.1}%",
            pc_hits,
            pc_misses,
            100.0 * pc_hits as f64 / total
        );
    }

    // Extract graph and optionally apply diversity-aware final prune.
    let t3 = Instant::now();
    let (adjacency, extract_secs, final_prune_secs) = if config.final_prune {
        // Extract full reservoir (l_max candidates with distances) for diversity prune.
        let candidates = hash_prune.extract_graph_for_prune();
        let extract_secs = t3.elapsed().as_secs_f64();

        let t4 = Instant::now();
        // Use the iterative-alpha occlude_list port so the HashPrune-extract path and the
        // accumulate path share one final-prune impl (keeps comparisons apples-to-apples).
        let adj = crate::prune::final_robust_prune(
            data,
            ndims,
            &candidates,
            config.max_degree,
            config.metric,
            config.alpha,
            config.saturate_after_prune,
        );

        let final_prune_secs = t4.elapsed().as_secs_f64();
        (adj, extract_secs, final_prune_secs)
    } else {
        // No prune: truncate to max_degree by distance (original path).
        let adj = hash_prune.extract_graph();
        let extract_secs = t3.elapsed().as_secs_f64();
        (adj, extract_secs, 0.0)
    };

    let total_secs = t_total.elapsed().as_secs_f64();

    let build_stats = PiPNNBuildStats {
        total_secs,
        sketch_secs,
        partition_secs,
        leaf_build_secs,
        extract_secs,
        final_prune_secs,
        num_leaves: total_leaves,
        total_edges: total_edges_count,
    };

    let graph = PiPNNGraph {
        adjacency,
        npoints,
        ndims,
        medoid,
        metric: config.metric,
        build_stats,
    };

    // Return all freed memory (reservoirs, sketches, partition buffers, leaf buffers)
    // to the OS before handing off to the disk layout phase.

    eprintln!(
        "PiPNN graph: avg_degree={:.3} max_degree={} isolated={} (HashPrune baseline)",
        graph.avg_degree(),
        graph.max_degree(),
        graph.num_isolated(),
    );
    tracing::info!(
        avg_degree = graph.avg_degree(),
        max_degree = graph.max_degree(),
        isolated = graph.num_isolated(),
        "PiPNN build complete"
    );

    Ok(graph)
}

/// Unifies the two candidate accumulators behind one call so the leaf loop is generic.
trait EdgeSink: Sync {
    fn sink_edges(&self, edges: &[leaf_build::Edge]);
}
impl EdgeSink for crate::candidate_pool::CandidatePool {
    fn sink_edges(&self, edges: &[leaf_build::Edge]) {
        self.add_edges_batched(edges);
    }
}
impl EdgeSink for crate::candidate_pool::AppendOnlyPool {
    fn sink_edges(&self, edges: &[leaf_build::Edge]) {
        self.add_edges_batched(edges);
    }
}

/// Partition + per-leaf RobustPrune for the experimental merge modes, streaming pruned
/// edges into `sink`. Mirrors the baseline replica/partition/par_chunks structure.
// Runs inside the rayon pool installed by `build_internal`.
#[allow(clippy::disallowed_methods)]
fn run_leaves_robust<T: VectorRepr + Send + Sync, S: EdgeSink>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    sink: &S,
) -> (f64, usize, usize) {
    let mut partition_secs = 0.0f64;
    let mut total_leaves = 0usize;
    let raw_edges = std::sync::atomic::AtomicUsize::new(0);
    for replica in 0..config.replicas {
        let seed = 1000 + replica as u64 * 7919;
        let partition_config = PartitionConfig {
            c_max: config.c_max,
            c_min: config.c_min,
            p_samp: config.p_samp,
            fanout: config.fanout.clone(),
            metric: config.metric,
            leader_cap: config.leader_cap,
        };
        let t_part = Instant::now();
        let leaves = crate::partition::partition(data, ndims, npoints, &partition_config, seed);
        partition_secs += t_part.elapsed().as_secs_f64();
        total_leaves += leaves.len();
        (0..rayon::current_num_threads())
            .into_par_iter()
            .for_each(|_| crate::partition::release_stripe_buffers());

        const LEAF_BATCH: usize = 64;
        leaves.par_chunks(LEAF_BATCH).for_each(|chunk| {
            leaf_build::LEAF_BUFFERS.with(|cell| {
                let mut bufs = cell.borrow_mut();
                for leaf in chunk {
                    let indices_usize: Vec<usize> =
                        leaf.indices.iter().map(|&i| i as usize).collect();
                    let leaf_deg = if config.leaf_prune_degree > 0 {
                        config.leaf_prune_degree
                    } else {
                        config.max_degree
                    };
                    let edges = match config.leaf_prune_mode {
                        crate::LeafPruneMode::RobustNoGemm => leaf_build::build_leaf_robust_no_gemm(
                            data,
                            ndims,
                            &indices_usize,
                            leaf_deg,
                            config.metric,
                            config.alpha,
                            &mut bufs,
                        ),
                        // GemmTopKRobust = prune in leaf; GemmTopKNoPrune = keep top-k
                        // (the paper's bi-directed-kNN-style control: no leaf prune).
                        crate::LeafPruneMode::GemmTopKRobust
                        | crate::LeafPruneMode::GemmTopKNoPrune => {
                            let leaf_prune = matches!(
                                config.leaf_prune_mode,
                                crate::LeafPruneMode::GemmTopKRobust
                            );
                            leaf_build::build_leaf_gemm_topk_robust(
                                data,
                                ndims,
                                &indices_usize,
                                config.k,
                                leaf_deg,
                                config.metric,
                                config.alpha,
                                leaf_prune,
                                &mut bufs,
                            )
                        }
                        // Bi-directed k-NN candidates feeding the accumulator (clean
                        // iso-leaf merge comparison vs HashPrune).
                        crate::LeafPruneMode::KnnBidir => leaf_build::build_leaf_with_buffers(
                            data,
                            ndims,
                            &indices_usize,
                            config.k,
                            config.metric,
                            &mut bufs,
                        ),
                        crate::LeafPruneMode::Baseline => Vec::new(),
                    };
                    raw_edges.fetch_add(edges.len(), std::sync::atomic::Ordering::Relaxed);
                    sink.sink_edges(&edges);
                }
            });
        });
    }
    (0..rayon::current_num_threads())
        .into_par_iter()
        .for_each(|_| leaf_build::release_thread_buffers());
    (partition_secs, total_leaves, raw_edges.into_inner())
}

/// RobustPrune merge build (Exp 1 / Exp 2): accumulate per-leaf RobustPruned edges, then
/// run one final RobustPrune. No HashPrune. f32/typed path only (final prune needs f32).
fn build_internal_robust<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    medoid: usize,
) -> PiPNNResult<PiPNNGraph> {
    let t_total = Instant::now();
    eprintln!(
        "PiPNN RobustPrune merge: mode={:?} merge_l_max={} leaf_k={} alpha={}",
        config.leaf_prune_mode, config.merge_l_max, config.k, config.alpha
    );

    let t_leaf = Instant::now();
    let (candidates, partition_secs, total_leaves, raw_edges): (
        Vec<Vec<(u32, f32)>>,
        f64,
        usize,
        usize,
    ) = if config.merge_l_max > 0 {
        let pool = crate::candidate_pool::CandidatePool::new(npoints, config.merge_l_max);
        let (ps, nl, raw) = run_leaves_robust(data, npoints, ndims, config, &pool);
        (pool.extract_dedupped_sorted(), ps, nl, raw)
    } else {
        let pool = crate::candidate_pool::AppendOnlyPool::new(npoints);
        let (ps, nl, raw) = run_leaves_robust(data, npoints, ndims, config, &pool);
        (pool.extract_dedupped_sorted(), ps, nl, raw)
    };
    // leaf_build_secs excludes the partition time tracked separately.
    let leaf_build_secs = (t_leaf.elapsed().as_secs_f64() - partition_secs).max(0.0);
    let accumulated_edges: usize = candidates.iter().map(|c| c.len()).sum();
    let max_fanin = candidates.iter().map(|c| c.len()).max().unwrap_or(0);
    eprintln!(
        "  PRUNE FAN-IN: raw_accumulated={} dedupped={} avg_per_node={:.1} max_per_node={} -> final RobustPrune to max_degree={}",
        raw_edges,
        accumulated_edges,
        accumulated_edges as f64 / npoints as f64,
        max_fanin,
        config.max_degree
    );

    let t_fp = Instant::now();
    let adjacency = crate::prune::final_robust_prune(
        data,
        ndims,
        &candidates,
        config.max_degree,
        config.metric,
        config.alpha,
        config.saturate_after_prune,
    );
    drop(candidates);
    let final_prune_secs = t_fp.elapsed().as_secs_f64();

    let total_secs = t_total.elapsed().as_secs_f64();
    let build_stats = PiPNNBuildStats {
        total_secs,
        sketch_secs: 0.0,
        partition_secs,
        leaf_build_secs,
        extract_secs: 0.0,
        final_prune_secs,
        num_leaves: total_leaves,
        total_edges: accumulated_edges,
    };
    print!("{}", build_stats);

    let graph = PiPNNGraph {
        adjacency,
        npoints,
        ndims,
        medoid,
        metric: config.metric,
        build_stats,
    };
    eprintln!(
        "PiPNN graph: avg_degree={:.3} max_degree={} isolated={} (mode={:?}, {} leaves, {} accumulated cand)",
        graph.avg_degree(),
        graph.max_degree(),
        graph.num_isolated(),
        config.leaf_prune_mode,
        total_leaves,
        accumulated_edges,
    );
    tracing::info!(
        avg_degree = graph.avg_degree(),
        max_degree = graph.max_degree(),
        isolated = graph.num_isolated(),
        "PiPNN RobustPrune-merge build complete"
    );
    Ok(graph)
}

/// Final diversity prune matching DiskANN's occlude_list algorithm:
/// - Iterative alpha: starts at 1.0, increments by min(alpha, 1.2) each round
/// - Accumulated occlusion factor per candidate: max(dist_to_point / dist_to_selected)
/// - Resumable inner loop via last_checked positions
///
/// Candidates already have distances from HashPrune — sorted by distance ascending.
// Called from within `build_internal_impl` which already runs inside a dedicated rayon
// thread pool installed by `build_internal`, so `par_iter` work executes on that pool.
#[allow(clippy::disallowed_methods)]
pub fn final_prune_from_candidates<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    candidates_per_node: &[Vec<(u32, f32)>],
    max_degree: usize,
    metric: Metric,
    alpha: f32,
    saturate: bool,
) -> Vec<Vec<u32>> {
    // Dimension-specialized f32 distance — fastest for the many-to-many occlusion loop.
    // f32 precompute converts each candidate once; the occlusion loop then uses pure f32 FMA
    // (~55 reuses per candidate). Native f16 would do F16C conversion on every reuse.
    let dist_fn = <f32 as DistanceProvider<f32>>::distance_comparer(metric, Some(ndims));

    // Thread-local f32 buffer to avoid per-node allocation.
    thread_local! {
        static PRUNE_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    candidates_per_node
        .par_iter()
        .enumerate()
        .map(|(node_id, candidates)| {
            if candidates.is_empty() {
                return Vec::new();
            }

            let nc = candidates.len();

            PRUNE_BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                // node_f32 (ndims) + cand_f32 (nc * ndims)
                let total = (nc + 1) * ndims;
                if buf.len() < total {
                    buf.resize(total, 0.0);
                }

                // Convert node + all candidates to f32 once.
                let (node_f32, cand_f32) = buf[..total].split_at_mut(ndims);
                T::as_f32_into(&data[node_id * ndims..(node_id + 1) * ndims], node_f32)
                    .expect("f32 conversion");

                let mut fresh_dist_xz = vec![0.0f32; nc];
                {
                    let _timer =
                        crate::profile::PhaseTimer::start("final_prune/convert_and_dist_xz");
                    for (ci, &(id, _)) in candidates.iter().enumerate() {
                        let dst = &mut cand_f32[ci * ndims..(ci + 1) * ndims];
                        T::as_f32_into(&data[id as usize * ndims..(id as usize + 1) * ndims], dst)
                            .expect("f32 conversion");
                        fresh_dist_xz[ci] = dist_fn.call(node_f32, dst);
                    }
                }

                // Sort by fresh distance.
                let mut order: Vec<usize> = (0..nc).collect();
                {
                    let _timer = crate::profile::PhaseTimer::start("final_prune/sort");
                    order.sort_unstable_by(|&a, &b| {
                        fresh_dist_xz[a]
                            .partial_cmp(&fresh_dist_xz[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                // Greedy occlusion — prune ALL nodes for diversity.
                const UNVISITED: u8 = 0;
                const SELECTED: u8 = 1;
                const OCCLUDED: u8 = 2;
                let mut state = vec![UNVISITED; nc];
                let mut selected: Vec<u32> = Vec::with_capacity(max_degree.min(nc));

                {
                    let _timer = crate::profile::PhaseTimer::start("final_prune/occlusion_loop");
                    for oi in 0..nc {
                        let i = order[oi];
                        if selected.len() >= max_degree {
                            break;
                        }
                        if state[i] != UNVISITED {
                            continue;
                        }

                        selected.push(candidates[i].0);
                        state[i] = SELECTED;

                        let y_f32 = &cand_f32[i * ndims..(i + 1) * ndims];

                        for oj in (oi + 1)..nc {
                            let j = order[oj];
                            if state[j] != UNVISITED {
                                continue;
                            }
                            let dist_x_z = fresh_dist_xz[j];
                            let z_f32 = &cand_f32[j * ndims..(j + 1) * ndims];
                            let dist_y_z = dist_fn.call(y_f32, z_f32);

                            if alpha * dist_y_z < dist_x_z {
                                state[j] = OCCLUDED;
                            }
                        }
                    }
                }

                // Saturation: fill remaining degree slots in fresh distance order.
                {
                    let _timer = crate::profile::PhaseTimer::start("final_prune/saturate");
                    if saturate && selected.len() < max_degree {
                        for oi in 0..nc {
                            let i = order[oi];
                            if selected.len() >= max_degree {
                                break;
                            }
                            if state[i] != SELECTED {
                                selected.push(candidates[i].0);
                            }
                        }
                    }
                }

                selected
            }) // close PRUNE_BUF.with
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_data(npoints: usize, ndims: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..npoints * ndims)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect()
    }

    #[test]
    fn test_build_small() {
        let npoints = 100;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config).unwrap();

        assert_eq!(graph.npoints, npoints);
        assert!(graph.avg_degree() > 0.0);
        assert!(graph.num_isolated() < npoints);
    }

    #[test]
    fn test_build_data_length_mismatch() {
        let data = vec![0.0f32; 10];
        let config = PiPNNConfig::default();

        let result = build(&data, 5, 3, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, PiPNNError::DataLengthMismatch { .. }));
    }

    #[test]
    fn test_search_basic() {
        let npoints = 200;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 4,
            max_degree: 32,
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config).unwrap();

        let query = &data[0..ndims];
        let results = graph.search(&data, query, 10, 50);

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_recall() {
        use crate::leaf_build::brute_force_knn;

        let npoints = 500;
        let ndims = 16;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 128,
            c_min: 32,
            k: 4,
            max_degree: 32,
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config).unwrap();

        let k = 10;
        let search_l = 100;
        let num_queries = 20;

        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..ndims)
                .map(|_| rng.random_range(-1.0f32..1.0f32))
                .collect();

            let approx = graph.search(&data, &query, k, search_l);
            let exact = brute_force_knn(&data, ndims, npoints, &query, k);

            let exact_set: std::collections::HashSet<usize> =
                exact.iter().map(|&(id, _)| id).collect();
            let recall = approx
                .iter()
                .filter(|&&(id, _)| exact_set.contains(&id))
                .count() as f64
                / k as f64;

            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        eprintln!("Average recall@{}: {:.4}", k, avg_recall);

        assert!(avg_recall > 0.2, "recall too low: {:.4}", avg_recall);
    }

    #[test]
    fn test_config_validate() {
        let config = PiPNNConfig::default();
        assert!(config.validate().is_ok());

        let bad = PiPNNConfig {
            c_max: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            c_min: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            c_min: 2048,
            c_max: 1024,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            p_samp: 0.0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            p_samp: 1.5,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            fanout: vec![],
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            fanout: vec![0],
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            num_hash_planes: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            num_hash_planes: 17,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_config_validate_failures() {
        // max_degree = 0
        let bad = PiPNNConfig {
            max_degree: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        // k = 0
        let bad = PiPNNConfig {
            k: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        // replicas = 0
        let bad = PiPNNConfig {
            replicas: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        // l_max = 0
        let bad = PiPNNConfig {
            l_max: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        // p_samp exactly 1.0 is valid
        let ok = PiPNNConfig {
            p_samp: 1.0,
            ..Default::default()
        };
        assert!(ok.validate().is_ok());

        // num_hash_planes = 1 (boundary) is valid
        let ok = PiPNNConfig {
            num_hash_planes: 1,
            ..Default::default()
        };
        assert!(ok.validate().is_ok());

        // num_hash_planes = 16 (boundary) is valid
        let ok = PiPNNConfig {
            num_hash_planes: 16,
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn test_build_cosine() {
        let npoints = 100;
        let ndims = 8;
        // Generate random data and normalize each vector for cosine.
        let mut data = generate_random_data(npoints, ndims, 42);
        for i in 0..npoints {
            let row = &mut data[i * ndims..(i + 1) * ndims];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            metric: diskann_vector::distance::Metric::Cosine,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert!(matches!(graph.metric, Metric::Cosine));
        assert_eq!(graph.npoints, npoints);
        assert!(graph.avg_degree() > 0.0);
    }

    /// Train SQ parameters from data. Test-only helper.
    fn train_sq_params(data: &[f32], npoints: usize, ndims: usize) -> SQParams {
        use diskann_quantization::scalar::train::ScalarQuantizationParameters;
        use diskann_utils::views::MatrixView;

        let data_matrix = MatrixView::try_from(data, npoints, ndims)
            .expect("data length must equal npoints * ndims");
        let quantizer = ScalarQuantizationParameters::default().train(data_matrix);
        let shift = quantizer.shift().to_vec();
        let scale = quantizer.scale();
        let inverse_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };
        SQParams {
            shift,
            inverse_scale,
        }
    }

    #[test]
    fn test_build_with_sq() {
        let npoints = 100;
        let ndims = 64; // must be multiple of 64 for u64 alignment in quantize
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };

        let sq_params = train_sq_params(&data, npoints, ndims);

        let graph = super::build_with_sq(&data, npoints, ndims, &config, &sq_params).unwrap();
        assert_eq!(graph.npoints, npoints);
        assert!(graph.avg_degree() > 0.0);
    }

    #[test]
    fn test_build_typed_f32() {
        let npoints = 60;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };

        let graph_direct = build(&data, npoints, ndims, &config).unwrap();
        let graph_typed = build_typed::<f32>(&data, npoints, ndims, &config).unwrap();

        // Both should produce the same npoints and medoid.
        assert_eq!(graph_direct.npoints, graph_typed.npoints);
        assert_eq!(graph_direct.medoid, graph_typed.medoid);
    }

    #[test]
    fn test_save_graph_format() {
        let npoints = 50;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config).unwrap();

        let dir = std::env::temp_dir().join("pipnn_test_save_graph");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_graph.bin");
        graph.save_graph(&path).unwrap();

        // Read back and verify the header.
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= 24, "file too small: {} bytes", bytes.len());

        // First 8 bytes: u64 LE file size.
        let file_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(file_size as usize, bytes.len(), "header file_size mismatch");

        // Bytes 8..12: u32 LE max degree.
        let max_deg = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(max_deg as usize, graph.max_degree());

        // Bytes 12..16: u32 LE start point (medoid).
        let start_pt = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(start_pt as usize, graph.medoid);

        // Clean up.
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_medoid_is_valid() {
        let npoints = 100;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert!(
            graph.medoid < npoints,
            "medoid {} is out of range [0, {})",
            graph.medoid,
            npoints
        );
    }

    #[test]
    fn test_graph_connectivity() {
        // With sufficient replicas and params, no nodes should be isolated.
        let npoints = 200;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 4,
            max_degree: 32,
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config).unwrap();

        // With these settings no node should be completely isolated.
        assert_eq!(
            graph.num_isolated(),
            0,
            "found {} isolated nodes with replicas=2",
            graph.num_isolated()
        );
    }

    #[test]
    fn test_build_zero_npoints() {
        let data: Vec<f32> = vec![];
        let config = PiPNNConfig::default();
        let result = build(&data, 0, 8, &config);
        assert!(result.is_err(), "npoints=0 should error");
    }

    #[test]
    fn test_build_zero_ndims() {
        let data: Vec<f32> = vec![];
        let config = PiPNNConfig::default();
        let result = build(&data, 10, 0, &config);
        assert!(result.is_err(), "ndims=0 should error");
    }

    #[test]
    fn test_build_single_point() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 1,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, 1, 4, &config).unwrap();
        assert_eq!(graph.npoints, 1, "should have 1 point");
        assert_eq!(
            graph.adjacency[0].len(),
            0,
            "single point should have 0 edges"
        );
    }

    #[test]
    fn test_build_two_points() {
        let data = vec![0.0f32, 0.0, 1.0, 0.0];
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 1,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, 2, 2, &config).unwrap();
        assert_eq!(graph.npoints, 2, "should have 2 points");
        // With 2 points, they should connect to each other.
        let total_edges: usize = graph.adjacency.iter().map(|a| a.len()).sum();
        assert!(
            total_edges > 0,
            "two points should have at least one edge between them"
        );
    }

    #[test]
    fn test_build_duplicate_points() {
        // All identical points; build should still succeed.
        let npoints = 20;
        let ndims = 4;
        let data = vec![1.0f32; npoints * ndims];
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 4,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert_eq!(
            graph.npoints, npoints,
            "should build successfully with duplicate points"
        );
    }

    #[test]
    fn test_build_very_small_k() {
        let npoints = 50;
        let ndims = 4;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 1,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert_eq!(graph.npoints, npoints, "k=1 should produce valid graph");
        assert!(
            graph.avg_degree() > 0.0,
            "k=1 should still produce some edges"
        );
    }

    #[test]
    fn test_build_k_larger_than_leaf() {
        // k > c_max should still work (clamped inside extract_knn).
        let npoints = 50;
        let ndims = 4;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 100, // larger than c_max
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert_eq!(
            graph.npoints, npoints,
            "k > c_max should still produce valid graph"
        );
    }

    #[test]
    fn test_search_empty_graph() {
        let graph = PiPNNGraph {
            adjacency: vec![],
            npoints: 0,
            ndims: 4,
            medoid: 0,
            metric: Metric::L2,
            build_stats: Default::default(),
        };
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let results = graph.search(&[], &query, 5, 10);
        assert!(
            results.is_empty(),
            "search on empty graph should return empty results"
        );
    }

    #[test]
    fn test_search_k_larger_than_npoints() {
        let npoints = 10;
        let ndims = 4;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 4,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        let query = &data[0..ndims];
        // Request more neighbors than points exist.
        let results = graph.search(&data, query, 100, 200);
        assert!(
            results.len() <= npoints,
            "should not return more results than npoints, got {}",
            results.len()
        );
    }

    #[test]
    fn test_search_with_self_query() {
        let npoints = 100;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 4,
            max_degree: 32,
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        // Query with the medoid point itself.
        let medoid = graph.medoid;
        let query = &data[medoid * ndims..(medoid + 1) * ndims];
        let results = graph.search(&data, query, 5, 50);
        assert!(
            !results.is_empty(),
            "search should return at least one result"
        );
        assert_eq!(
            results[0].0, medoid,
            "searching with a data point should find itself first"
        );
        assert!(
            results[0].1 < 1e-6,
            "self-distance should be near zero, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_search_different_l_values() {
        use crate::leaf_build::brute_force_knn;

        let npoints = 300;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 4,
            max_degree: 32,
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();

        let k = 10;
        let query = &data[0..ndims];
        let exact = brute_force_knn(&data, ndims, npoints, query, k);
        let exact_set: std::collections::HashSet<usize> = exact.iter().map(|&(id, _)| id).collect();

        // Compare recall for small L vs large L.
        let results_small_l = graph.search(&data, query, k, k);
        let recall_small: f64 = results_small_l
            .iter()
            .filter(|&&(id, _)| exact_set.contains(&id))
            .count() as f64
            / k as f64;

        let results_large_l = graph.search(&data, query, k, 200);
        let recall_large: f64 = results_large_l
            .iter()
            .filter(|&&(id, _)| exact_set.contains(&id))
            .count() as f64
            / k as f64;

        assert!(
            recall_large >= recall_small,
            "larger L ({:.4}) should give recall >= smaller L ({:.4})",
            recall_large,
            recall_small
        );
    }

    #[test]
    fn test_build_with_sq_wrong_shift_dims() {
        let npoints = 50;
        let ndims = 64;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        // Shift length != ndims.
        let sq_params = SQParams {
            shift: vec![0.0f32; ndims + 5], // wrong length
            inverse_scale: 1.0,
        };
        let result = build_with_sq(&data, npoints, ndims, &config, &sq_params);
        assert!(
            result.is_err(),
            "shift length != ndims should produce an error"
        );
        assert!(
            matches!(result.unwrap_err(), PiPNNError::DimensionMismatch { .. }),
            "should be a DimensionMismatch error"
        );
    }

    #[test]
    fn test_build_with_sq_produces_connected_graph() {
        let npoints = 100;
        let ndims = 64;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 4,
            max_degree: 32,
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };
        let sq_params = train_sq_params(&data, npoints, ndims);
        let graph = build_with_sq(&data, npoints, ndims, &config, &sq_params).unwrap();
        assert_eq!(
            graph.num_isolated(), 0,
            "build_with_sq should produce a connected graph with sufficient replicas, found {} isolated nodes",
            graph.num_isolated()
        );
    }

    #[test]
    fn test_build_typed_data_length_mismatch() {
        let data = vec![1.0f32; 30]; // 30 elements
        let config = PiPNNConfig::default();
        // npoints=5, ndims=8 expects 40 elements but data has 30.
        let result = build_typed::<f32>(&data, 5, 8, &config);
        assert!(
            result.is_err(),
            "data length mismatch should produce an error"
        );
    }

    #[test]
    fn test_save_graph_single_node() {
        let graph = PiPNNGraph {
            adjacency: vec![vec![]],
            npoints: 1,
            ndims: 4,
            medoid: 0,
            metric: Metric::L2,
            build_stats: Default::default(),
        };
        let dir = std::env::temp_dir().join("pipnn_test_save_single");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single.bin");
        graph.save_graph(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= 24, "file too small for single node graph");
        let file_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(
            file_size as usize,
            bytes.len(),
            "header file_size mismatch for single node"
        );

        // Max degree should be 0 for single node with no edges.
        let max_deg = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(
            max_deg, 0,
            "single node with no edges should have max_degree=0"
        );

        // Read back neighbor count for the single node.
        let num_neighbors = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        assert_eq!(num_neighbors, 0, "single node should have 0 neighbors");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_save_graph_large() {
        let npoints = 1000;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 128,
            c_min: 32,
            k: 4,
            max_degree: 32,
            replicas: 1,
            l_max: 64,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();

        let dir = std::env::temp_dir().join("pipnn_test_save_large");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large.bin");
        graph.save_graph(&path).unwrap();

        // Read back and verify we can parse all adjacency lists.
        let bytes = std::fs::read(&path).unwrap();
        let file_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(
            file_size as usize,
            bytes.len(),
            "header file_size mismatch for large graph"
        );

        let mut offset = 24usize;
        let mut total_parsed_nodes = 0usize;
        while offset < bytes.len() {
            let num_neighbors =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            for _ in 0..num_neighbors {
                let neighbor =
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                assert!(
                    neighbor < npoints,
                    "neighbor index {} out of range for node {}",
                    neighbor,
                    total_parsed_nodes
                );
                offset += 4;
            }
            total_parsed_nodes += 1;
        }
        assert_eq!(
            total_parsed_nodes,
            npoints + 1, // +1 for frozen start point
            "expected to parse {} nodes ({}+1 frozen) but got {}",
            npoints + 1,
            npoints,
            total_parsed_nodes
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_config_c_min_greater_than_c_max() {
        let config = PiPNNConfig {
            c_min: 2048,
            c_max: 1024,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "c_min > c_max should fail validation"
        );
    }

    #[test]
    fn test_config_empty_fanout() {
        let config = PiPNNConfig {
            fanout: vec![],
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "empty fanout should fail validation"
        );
    }

    #[test]
    fn test_config_zero_fanout_element() {
        let config = PiPNNConfig {
            fanout: vec![5, 0, 2],
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "fanout containing 0 should fail validation"
        );
    }

    #[test]
    fn test_config_p_samp_zero() {
        let config = PiPNNConfig {
            p_samp: 0.0,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "p_samp=0.0 should fail validation"
        );
    }

    #[test]
    fn test_config_p_samp_negative() {
        let config = PiPNNConfig {
            p_samp: -0.5,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "p_samp < 0 should fail validation"
        );
    }

    #[test]
    fn test_config_hash_planes_zero() {
        let config = PiPNNConfig {
            num_hash_planes: 0,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "num_hash_planes=0 should fail validation"
        );
    }

    #[test]
    fn test_config_hash_planes_17() {
        let config = PiPNNConfig {
            num_hash_planes: 17,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "num_hash_planes=17 (> 16) should fail validation"
        );
    }

    #[test]
    fn test_final_prune_reduces_degree() {
        let npoints = 200;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        // Build without final prune, then build with, and compare max degree.
        let config_no_prune = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 6,
            max_degree: 16,
            replicas: 2,
            l_max: 64,
            final_prune: false,
            ..Default::default()
        };
        let config_with_prune = PiPNNConfig {
            final_prune: true,
            ..config_no_prune.clone()
        };

        let graph_no = build(&data, npoints, ndims, &config_no_prune).unwrap();
        let graph_yes = build(&data, npoints, ndims, &config_with_prune).unwrap();

        // Final prune should not increase max degree beyond max_degree.
        assert!(
            graph_yes.max_degree() <= config_with_prune.max_degree,
            "final_prune max_degree {} > config max_degree {}",
            graph_yes.max_degree(),
            config_with_prune.max_degree
        );

        // Both should be valid graphs.
        assert!(graph_no.avg_degree() > 0.0);
        assert!(graph_yes.avg_degree() > 0.0);
    }

    #[test]
    fn test_final_prune_from_candidates_diversity() {
        // 4 points: 0=(0,0), 1=(1,0), 2=(0,1), 3=(0.1,0) -- point 3 is occluded by 1.
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.1, 0.0];
        let candidates = vec![
            // Node 0's candidates sorted by distance: 3 (close, same direction as 1), 1 (far along x), 2 (far along y)
            vec![(3, 0.01f32), (1, 1.0f32), (2, 1.0f32)],
            vec![],
            vec![],
            vec![],
        ];

        let result = final_prune_from_candidates(&data, 2, &candidates, 2, Metric::L2, 1.2, true);
        let node0 = &result[0];
        // With alpha=1.2, point 3 should be selected first (closest).
        // Point 1 might be pruned because dist(3,1) * 1.2 < dist(0,1).
        // Point 2 should survive (different direction).
        assert!(!node0.is_empty());
        assert!(node0.len() <= 2, "should respect max_degree=2");
        // Node 0 should keep at least one neighbor.
        assert!(
            node0.contains(&3),
            "closest candidate should always be selected"
        );
    }

    #[test]
    fn test_final_prune_from_candidates_empty() {
        let data: Vec<f32> = vec![0.0; 8];
        let candidates: Vec<Vec<(u32, f32)>> = vec![vec![], vec![], vec![], vec![]];
        let result = final_prune_from_candidates(&data, 2, &candidates, 10, Metric::L2, 1.2, true);
        assert!(result.iter().all(|adj| adj.is_empty()));
    }

    #[test]
    fn test_final_prune_from_candidates_single_candidate() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];
        let candidates = vec![vec![(1, 1.0f32)], vec![(0, 1.0f32)]];
        let result = final_prune_from_candidates(&data, 2, &candidates, 10, Metric::L2, 1.2, true);
        assert_eq!(result[0], vec![1]);
        assert_eq!(result[1], vec![0]);
    }

    #[test]
    fn test_final_prune_alpha_effect() {
        // Higher alpha = less aggressive pruning = more edges retained.
        let npoints = 200;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config_aggressive = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 6,
            max_degree: 16,
            replicas: 2,
            l_max: 64,
            final_prune: true,
            alpha: 1.0,
            ..Default::default()
        };
        let config_relaxed = PiPNNConfig {
            alpha: 2.0,
            ..config_aggressive.clone()
        };

        let graph_aggressive = build(&data, npoints, ndims, &config_aggressive).unwrap();
        let graph_relaxed = build(&data, npoints, ndims, &config_relaxed).unwrap();

        // Relaxed alpha should yield denser graph (more edges survive pruning).
        assert!(
            graph_relaxed.avg_degree() >= graph_aggressive.avg_degree(),
            "alpha=2.0 ({:.1}) should produce >= degree than alpha=1.0 ({:.1})",
            graph_relaxed.avg_degree(),
            graph_aggressive.avg_degree()
        );
    }

    #[test]
    fn test_build_final_prune_vs_no_prune_recall() {
        // Both should produce searchable graphs with reasonable recall.
        let npoints = 500;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config_no_prune = PiPNNConfig {
            c_max: 128,
            c_min: 32,
            k: 3,
            max_degree: 32,
            replicas: 1,
            l_max: 64,
            final_prune: false,
            ..Default::default()
        };
        let config_prune = PiPNNConfig {
            l_max: 64,
            final_prune: true,
            ..config_no_prune.clone()
        };

        let graph_no = build(&data, npoints, ndims, &config_no_prune).unwrap();
        let graph_yes = build(&data, npoints, ndims, &config_prune).unwrap();

        // Both should have non-trivial degree.
        assert!(graph_no.avg_degree() > 1.0);
        assert!(graph_yes.avg_degree() > 1.0);

        // Final prune should produce sparser graph.
        assert!(
            graph_yes.avg_degree() <= graph_no.avg_degree(),
            "pruned ({:.1}) should be <= unpruned ({:.1})",
            graph_yes.avg_degree(),
            graph_no.avg_degree()
        );

        // Both should be searchable.
        let query = &data[0..ndims];
        let r1 = crate::leaf_build::brute_force_knn(&data, ndims, npoints, query, 10);
        let s_no = graph_no.search(&data, query, 10, 50);
        let s_yes = graph_yes.search(&data, query, 10, 50);
        assert!(!s_no.is_empty(), "no_prune search should return results");
        assert!(!s_yes.is_empty(), "prune search should return results");

        // Both should find the nearest neighbor (query is point 0).
        assert_eq!(s_no[0].0, r1[0].0, "no_prune should find NN");
        assert_eq!(s_yes[0].0, r1[0].0, "prune should find NN");
    }

    #[test]
    fn test_build_cosine_normalized() {
        let npoints = 100;
        let ndims = 8;
        let mut data = generate_random_data(npoints, ndims, 42);
        // Normalize all vectors.
        for i in 0..npoints {
            let row = &mut data[i * ndims..(i + 1) * ndims];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            metric: Metric::CosineNormalized,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert!(graph.avg_degree() > 0.0);
        assert_eq!(graph.metric, Metric::CosineNormalized);

        // Search should work with cosine metric.
        let query = &data[0..ndims];
        let results = graph.search(&data, query, 5, 20);
        assert!(!results.is_empty());
        // First result should be the query point itself.
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_config_validate_inner_product_rejected() {
        let config = PiPNNConfig {
            metric: Metric::InnerProduct,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_alpha_infinity() {
        let config = PiPNNConfig {
            alpha: f32::INFINITY,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_p_samp_nan() {
        let config = PiPNNConfig {
            p_samp: f64::NAN,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipnn_graph_stats() {
        let npoints = 100;
        let ndims = 4;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();

        assert_eq!(graph.npoints, npoints);
        assert_eq!(graph.ndims, ndims);
        assert!(graph.medoid < npoints);
        assert!(graph.max_degree() <= config.max_degree);
        assert!(graph.avg_degree() > 0.0);
        assert!(graph.avg_degree() <= config.max_degree as f64);
        // num_isolated should be 0 for a well-connected graph.
        assert_eq!(
            graph.num_isolated(),
            0,
            "graph should have no isolated nodes"
        );
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = PiPNNConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: PiPNNConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config.c_max, deserialized.c_max);
        assert_eq!(config.k, deserialized.k);
        assert_eq!(config.max_degree, deserialized.max_degree);
        assert!((config.alpha - deserialized.alpha).abs() < 1e-6);
    }
}
