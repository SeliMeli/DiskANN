/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf building: GEMM-based all-pairs distance computation and bi-directed k-NN extraction.
//!
//! For each leaf partition (bounded by C_max, typically 1024-2048):
//! 1. Compute all-pairs distance matrix via GEMM
//!    For L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a.b)
//!    The dot product matrix A * A^T is computed as a GEMM operation.
//! 2. Extract k nearest neighbors per point using partial sort
//! 3. Create bi-directed edges (both forward and reverse k-NN)

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use diskann::utils::VectorRepr;
use diskann_vector::distance::{DistanceProvider, Metric, SquaredL2};
use diskann_vector::PureDistanceFunction;

/// Env switch: `PIPNN_POINT_CACHE=1` opts into the per-thread point cache.
/// Default OFF — the May 2026 experiment showed the cache plateaus around 10%
/// hit rate on Enron 1M (fanout=[8,3]) regardless of size because the 24
/// leaves containing each point are scattered across the entire leaf list.
/// Read once and cached so the loop has no syscall overhead.
fn point_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PIPNN_POINT_CACHE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("on")
        )
    })
}

/// Global counters for cache effectiveness. Each thread folds its local
/// `point_cache.hits / .misses` into these when its buffers are released.
pub static POINT_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub static POINT_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Per-thread direct-mapped point cache used during leaf gather.
///
/// Each point appears in up to `fanout[0] * fanout[1] * ...` leaves (e.g. 24
/// with fanout=[8,3]). Without caching, each appearance triggers a fresh DRAM
/// read of `ndims` floats. With a direct-mapped cache + leaf sort that groups
/// overlapping leaves on the same thread, repeated lookups hit L2/L3 instead.
///
/// Layout:
/// - `keys[slot]` = u32::MAX (empty) or the point ID currently in `slot`.
/// - `data[slot*ndims .. slot*ndims+ndims]` = the f32 vector for that point.
///
/// Conflict misses are handled by simple overwrite (no associativity). With a
/// good hash mixing function and capacity ≥ leaf size, collisions inside a
/// single leaf are rare; across leaves, eviction is the goal anyway.
pub struct PointCache {
    pub data: Vec<f32>,
    pub keys: Vec<u32>,
    pub ndims: usize,
    pub hits: u64,
    pub misses: u64,
}

/// Default 2^11 = 2048 slots. At ndims=384 (Enron), 2048 * 384 * 4 = 3 MB per
/// thread — overflows L2 (1 MB) but stays in shared L3 (Cascade Lake ~32 MB,
/// EPYC 7763 ~32 MB/CCX). At 16 threads × 3 MB = 48 MB total, contention is
/// shared so this is acceptable. Override with PIPNN_CACHE_LOG2=<n>.
const POINT_CACHE_LOG2_DEFAULT: u32 = 11;
const POINT_CACHE_EMPTY: u32 = u32::MAX;

fn point_cache_log2() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PIPNN_CACHE_LOG2")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|n| (3..=20).contains(n))
            .unwrap_or(POINT_CACHE_LOG2_DEFAULT)
    })
}

fn point_cache_size() -> usize {
    1usize << point_cache_log2()
}

fn point_cache_mask() -> usize {
    point_cache_size() - 1
}

impl Default for PointCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PointCache {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            keys: Vec::new(),
            ndims: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Allocate cache storage for `ndims`-vectors. Re-init if `ndims` changed.
    pub fn ensure_capacity(&mut self, ndims: usize) {
        let sz = point_cache_size();
        if self.ndims != ndims {
            self.data.clear();
            self.data.resize(sz * ndims, 0.0);
            self.keys.clear();
            self.keys.resize(sz, POINT_CACHE_EMPTY);
            self.ndims = ndims;
        } else if self.keys.len() < sz {
            self.data.resize(sz * ndims, 0.0);
            self.keys.resize(sz, POINT_CACHE_EMPTY);
        }
    }

    /// Reset all slots to empty, keep allocations. Called between builds so a
    /// stale entry from a prior dataset can't collide with the new one.
    pub fn clear(&mut self) {
        for k in self.keys.iter_mut() {
            *k = POINT_CACHE_EMPTY;
        }
        self.hits = 0;
        self.misses = 0;
    }

    /// Free cache memory entirely (e.g. on thread release).
    pub fn release(&mut self) {
        self.data = Vec::new();
        self.keys = Vec::new();
        self.ndims = 0;
        self.hits = 0;
        self.misses = 0;
    }

    /// Fibonacci hashing mixes sequential point IDs across slots.
    #[inline(always)]
    fn slot(key: u32) -> usize {
        let h = (key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h as usize) & point_cache_mask()
    }
}

/// Thread-local reusable buffers for leaf building.
/// Avoids repeated allocation/deallocation of large matrices.
pub struct LeafBuffers {
    pub local_data: Vec<f32>,
    pub norms_sq: Vec<f32>,
    pub dot_matrix: Vec<f32>,
    pub dist_matrix: Vec<f32>,
    pub seen: Vec<bool>,
    pub point_cache: PointCache,
}

impl Default for LeafBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl LeafBuffers {
    pub fn new() -> Self {
        Self {
            local_data: Vec::new(),
            norms_sq: Vec::new(),
            dot_matrix: Vec::new(),
            dist_matrix: Vec::new(),
            seen: Vec::new(),
            point_cache: PointCache::new(),
        }
    }

    /// Ensure all buffers are large enough for a leaf of size n x ndims.
    fn ensure_capacity(&mut self, n: usize, ndims: usize) {
        let nd = n * ndims;
        let nn = n * n;
        if self.local_data.len() < nd {
            self.local_data.resize(nd, 0.0);
        }
        if self.norms_sq.len() < n {
            self.norms_sq.resize(n, 0.0);
        }
        if self.dot_matrix.len() < nn {
            self.dot_matrix.resize(nn, 0.0);
        }
        if self.dist_matrix.len() < nn {
            self.dist_matrix.resize(nn, 0.0);
        }
        if self.seen.len() < nn {
            self.seen.resize(nn, false);
        }
    }
}

/// Thread-local reusable buffers for quantized leaf building.
struct QuantLeafBuffers {
    local_u64: Vec<u64>,
    dist_matrix: Vec<f32>,
    seen: Vec<bool>,
}

impl QuantLeafBuffers {
    fn new() -> Self {
        Self {
            local_u64: Vec::new(),
            dist_matrix: Vec::new(),
            seen: Vec::new(),
        }
    }
}

thread_local! {
    /// Thread-local reusable buffers for leaf building. Public so builder can
    /// batch multiple leaves per TLS access (amortizes the `with()` overhead).
    pub static LEAF_BUFFERS: RefCell<LeafBuffers> = RefCell::new(LeafBuffers::new());
    static QUANT_BUFFERS: RefCell<QuantLeafBuffers> = RefCell::new(QuantLeafBuffers::new());
}

/// Release thread-local leaf build buffers on the calling thread.
///
/// After leaf building is complete, these buffers pin pages in glibc's
/// per-thread arenas, preventing `malloc_trim` from returning freed
/// reservoir memory to the OS. Calling this from each rayon thread
/// helps glibc reclaim arena pages (best-effort: depends on rayon work-stealing touching all workers).
pub fn release_thread_buffers() {
    LEAF_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        bufs.local_data = Vec::new();
        bufs.norms_sq = Vec::new();
        bufs.dot_matrix = Vec::new();
        bufs.dist_matrix = Vec::new();
        bufs.seen = Vec::new();
        if bufs.point_cache.hits != 0 || bufs.point_cache.misses != 0 {
            POINT_CACHE_HITS.fetch_add(bufs.point_cache.hits, Ordering::Relaxed);
            POINT_CACHE_MISSES.fetch_add(bufs.point_cache.misses, Ordering::Relaxed);
        }
        bufs.point_cache.release();
    });
    QUANT_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        bufs.local_u64 = Vec::new();
        bufs.dist_matrix = Vec::new();
        bufs.seen = Vec::new();
    });
}

/// An edge produced by leaf building: (source, destination, distance).
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub distance: f32,
}

/// Extract k nearest neighbors for each point from the distance matrix.
///
/// For small k (≤3), uses a direct linear scan tracking top-k — O(n) per row
/// with minimal overhead. For larger k, uses index-based quickselect.
fn extract_knn(dist_matrix: &[f32], n: usize, k: usize) -> Vec<(usize, usize, f32)> {
    if n <= 1 || k == 0 {
        return Vec::new();
    }
    let actual_k = k.min(n - 1);
    if actual_k <= 3 {
        return extract_knn_small(dist_matrix, n, actual_k);
    }
    extract_knn_general(dist_matrix, n, actual_k)
}

/// Specialized extraction for k ≤ 3: AVX-512 SIMD scan tracking top-k per row.
/// Processes 16 distances per cycle, only drops to scalar for the rare lanes
/// that beat the current threshold. Same pattern as the partition SIMD top-k.
fn extract_knn_small(dist_matrix: &[f32], n: usize, k: usize) -> Vec<(usize, usize, f32)> {
    debug_assert!(k <= 3 && k < n);
    let mut edges = Vec::with_capacity(n * k);

    for i in 0..n {
        let row = &dist_matrix[i * n..(i + 1) * n];
        let mut top: [(u32, f32); 3] = [(u32::MAX, f32::MAX); 3];
        let threshold_idx = k - 1;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            use std::arch::x86_64::*;
            let chunks = n / 16;
            unsafe {
                // Set self-distance slot to MAX so it's never selected.
                // (dist_matrix diagonal is already MAX, but be safe.)
                for chunk in 0..chunks {
                    let base = chunk * 16;
                    let thresh = _mm512_set1_ps(top[threshold_idx].1);
                    let dists = _mm512_loadu_ps(row.as_ptr().add(base));
                    let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(dists, thresh);
                    if mask != 0 {
                        let mut d_arr = [0.0f32; 16];
                        _mm512_storeu_ps(d_arr.as_mut_ptr(), dists);
                        let mut m = mask;
                        while m != 0 {
                            let lane = m.trailing_zeros() as usize;
                            m &= m - 1;
                            let j = base + lane;
                            if j == i {
                                continue;
                            }
                            let d = d_arr[lane];
                            if d < top[threshold_idx].1 {
                                top[threshold_idx] = (j as u32, d);
                                for t in (1..k).rev() {
                                    if top[t].1 < top[t - 1].1 {
                                        top.swap(t, t - 1);
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                // Remainder
                for j in (chunks * 16)..n {
                    if j == i {
                        continue;
                    }
                    let d = *row.get_unchecked(j);
                    if d < top[threshold_idx].1 {
                        top[threshold_idx] = (j as u32, d);
                        for t in (1..k).rev() {
                            if top[t].1 < top[t - 1].1 {
                                top.swap(t, t - 1);
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        {
            use std::arch::x86_64::*;
            let chunks = n / 8;
            unsafe {
                for chunk in 0..chunks {
                    let base = chunk * 8;
                    let thresh = _mm256_set1_ps(top[threshold_idx].1);
                    let dists = _mm256_loadu_ps(row.as_ptr().add(base));
                    let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(dists, thresh));
                    if mask != 0 {
                        let mut d_arr = [0.0f32; 8];
                        _mm256_storeu_ps(d_arr.as_mut_ptr(), dists);
                        let mut m = mask as u32;
                        while m != 0 {
                            let lane = m.trailing_zeros() as usize;
                            m &= m - 1;
                            let j = base + lane;
                            if j == i {
                                continue;
                            }
                            let d = d_arr[lane];
                            if d < top[threshold_idx].1 {
                                top[threshold_idx] = (j as u32, d);
                                for t in (1..k).rev() {
                                    if top[t].1 < top[t - 1].1 {
                                        top.swap(t, t - 1);
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                for j in (chunks * 8)..n {
                    if j == i {
                        continue;
                    }
                    let d = *row.get_unchecked(j);
                    if d < top[threshold_idx].1 {
                        top[threshold_idx] = (j as u32, d);
                        for t in (1..k).rev() {
                            if top[t].1 < top[t - 1].1 {
                                top.swap(t, t - 1);
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = unsafe { *row.get_unchecked(j) };
                if d < top[threshold_idx].1 {
                    top[threshold_idx] = (j as u32, d);
                    for t in (1..k).rev() {
                        if top[t].1 < top[t - 1].1 {
                            top.swap(t, t - 1);
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        for t in 0..k {
            edges.push((i, top[t].0 as usize, top[t].1));
        }
    }

    edges
}

/// General extraction using quickselect on an index array.
fn extract_knn_general(dist_matrix: &[f32], n: usize, k: usize) -> Vec<(usize, usize, f32)> {
    let mut edges = Vec::with_capacity(n * k);
    let mut indices: Vec<u32> = (0..n as u32).collect();

    for i in 0..n {
        let row = &dist_matrix[i * n..(i + 1) * n];

        for j in 0..n {
            unsafe {
                *indices.get_unchecked_mut(j) = j as u32;
            }
        }

        if k < n {
            indices.select_nth_unstable_by(k - 1, |&a, &b| {
                let da = unsafe { *row.get_unchecked(a as usize) };
                let db = unsafe { *row.get_unchecked(b as usize) };
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for idx in 0..k {
            let j = unsafe { *indices.get_unchecked(idx) } as usize;
            edges.push((i, j, row[j]));
        }
    }

    edges
}

/// Build a leaf partition: compute all-pairs distances and extract bi-directed k-NN edges.
///
/// Returns edges as (global_src, global_dst, distance).
pub fn build_leaf<T: VectorRepr>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
    k: usize,
    metric: diskann_vector::distance::Metric,
) -> Vec<Edge> {
    let n = indices.len();
    if n <= 1 {
        return Vec::new();
    }

    LEAF_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        build_leaf_with_buffers(data, ndims, indices, k, metric, &mut bufs)
    })
}

/// Build a leaf using caller-provided buffers, bypassing thread-local access.
/// Use this when processing multiple leaves in a batch to amortize TLS overhead.
pub fn build_leaf_with_buffers<T: VectorRepr>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
    k: usize,
    metric: diskann_vector::distance::Metric,
    bufs: &mut LeafBuffers,
) -> Vec<Edge> {
    let n = indices.len();
    {
        let _timer = crate::profile::PhaseTimer::start("leaf_build/ensure_capacity");
        bufs.ensure_capacity(n, ndims);
    }

    // Gather rows for this leaf into `bufs.local_data`, converting T -> f32.
    // Each point may appear in many leaves (fanout-product replicas), so a
    // direct-mapped per-thread point cache short-circuits repeat reads.
    // PIPNN_POINT_CACHE=0 disables it for A/B tests.
    {
        let _timer = crate::profile::PhaseTimer::start("leaf_build/conv");
        if point_cache_enabled() {
            bufs.point_cache.ensure_capacity(ndims);
            let LeafBuffers {
                local_data,
                point_cache,
                ..
            } = &mut *bufs;
            let local_data = &mut local_data[..n * ndims];
            for (i, &idx) in indices.iter().enumerate() {
                let key = idx as u32;
                let slot = PointCache::slot(key);
                let dst_off = i * ndims;
                let dst = &mut local_data[dst_off..dst_off + ndims];
                // SAFETY: slot is masked into `keys`/`data` capacity.
                let hit = unsafe { *point_cache.keys.get_unchecked(slot) } == key;
                if hit {
                    let src_off = slot * ndims;
                    let src = unsafe { point_cache.data.get_unchecked(src_off..src_off + ndims) };
                    dst.copy_from_slice(src);
                    point_cache.hits += 1;
                } else {
                    let src = &data[idx * ndims..(idx + 1) * ndims];
                    T::as_f32_into(src, dst).expect("f32 conversion");
                    let src_off = slot * ndims;
                    unsafe {
                        *point_cache.keys.get_unchecked_mut(slot) = key;
                        let cdst = point_cache.data.get_unchecked_mut(src_off..src_off + ndims);
                        cdst.copy_from_slice(dst);
                    }
                    point_cache.misses += 1;
                }
            }
        } else {
            let local_data = &mut bufs.local_data[..n * ndims];
            for (i, &idx) in indices.iter().enumerate() {
                let src = &data[idx * ndims..(idx + 1) * ndims];
                let dst = &mut local_data[i * ndims..(i + 1) * ndims];
                T::as_f32_into(src, dst).expect("f32 conversion");
            }
        }
    }

    let local_data = &bufs.local_data[..n * ndims];
    if !matches!(metric, Metric::CosineNormalized | Metric::InnerProduct) {
        let norms_sq = &mut bufs.norms_sq[..n];
        {
            let _timer = crate::profile::PhaseTimer::start("leaf_build/norms");
            for i in 0..n {
                let row = &local_data[i * ndims..(i + 1) * ndims];
                let mut norm = 0.0f32;
                for &v in row.iter() {
                    norm += v * v;
                }
                norms_sq[i] = norm;
            }
        }
    }

    // GEMM: dots = local_data * local_data^T
    let dot_matrix = &mut bufs.dot_matrix[..n * n];
    {
        let _timer = crate::profile::PhaseTimer::start("leaf_build/gemm");
        crate::gemm::sgemm_aat(local_data, n, ndims, dot_matrix);
    }

    let norms_sq = &bufs.norms_sq[..n];

    // Convert to distance matrix using the target metric.
    use diskann_vector::distance::Metric;
    let dist_matrix = {
        let _timer = crate::profile::PhaseTimer::start("leaf_build/dist");
        match metric {
            Metric::CosineNormalized => {
                // Pre-normalized: dist = 1 - dot(a, b)
                for i in 0..n {
                    let row = &mut dot_matrix[i * n..(i + 1) * n];
                    for val in row.iter_mut() {
                        *val = (1.0 - *val).max(0.0);
                    }
                    row[i] = f32::MAX;
                }
                &mut bufs.dot_matrix[..n * n]
            }
            Metric::Cosine => {
                // Unnormalized: dist = 1 - dot(a,b)/(||a||*||b||)
                let dist = &mut bufs.dist_matrix[..n * n];
                for i in 0..n {
                    let ni_sqrt = norms_sq[i].sqrt();
                    for j in 0..n {
                        let denom = ni_sqrt * norms_sq[j].sqrt();
                        let cos_sim = if denom > 0.0 {
                            dot_matrix[i * n + j] / denom
                        } else {
                            0.0
                        };
                        dist[i * n + j] = (1.0 - cos_sim).max(0.0);
                    }
                    dist[i * n + i] = f32::MAX;
                }
                dist
            }
            Metric::L2 => {
                let dist = &mut bufs.dist_matrix[..n * n];
                for i in 0..n {
                    let ni = norms_sq[i];
                    for j in 0..n {
                        dist[i * n + j] = (ni + norms_sq[j] - 2.0 * dot_matrix[i * n + j]).max(0.0);
                    }
                    dist[i * n + i] = f32::MAX;
                }
                dist
            }
            Metric::InnerProduct => {
                for i in 0..n {
                    let row = &mut dot_matrix[i * n..(i + 1) * n];
                    for val in row.iter_mut() {
                        *val = -*val;
                    }
                    row[i] = f32::MAX;
                }
                &mut bufs.dot_matrix[..n * n]
            }
        }
    };

    let local_edges = {
        let _timer = crate::profile::PhaseTimer::start("leaf_build/knn");
        extract_knn(dist_matrix, n, k)
    };
    let seen = &mut bufs.seen[..n * n];
    {
        let _timer = crate::profile::PhaseTimer::start("leaf_build/seen_fill");
        seen.fill(false);
    }
    {
        let _timer = crate::profile::PhaseTimer::start("leaf_build/bidir");
        make_bidirected_edges(&local_edges, dist_matrix, n, indices, seen)
    }
}

/// Build a leaf using direct pairwise SIMD distance — no GEMM, no f32 conversion.
///
/// Computes distances directly on native T (f16) data using DiskANN's SIMD distance
/// functions. Eliminates: f16→f32 gather, norm computation, n×n GEMM, distance matrix
/// conversion, and separate extract_knn pass. Fuses distance computation with top-k
/// tracking for each point.
///
/// This approach is faster than GEMM for low-dimensional data (d≤128) because:
/// - No f16→f32 conversion (6.8% of build)
/// - No n×n matrix allocation (64KB per 128-point leaf)
/// - Distance + top-k fused in one pass per row
/// - SIMD distance on f16 uses F16C internally (same throughput as f32 GEMM)
pub fn build_leaf_direct<T: VectorRepr + diskann_vector::distance::DistanceProvider<T>>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
    k: usize,
    metric: Metric,
    dist_fn: &diskann_vector::distance::Distance<T, T>,
) -> Vec<Edge> {
    let n = indices.len();
    if n <= 1 {
        return Vec::new();
    }
    let actual_k = k.min(n - 1);

    // Compute top-k nearest for each point using direct distance.
    let mut local_edges: Vec<(usize, usize, f32)> = Vec::with_capacity(n * actual_k);
    let mut top: [(u32, f32); 3] = [(u32::MAX, f32::MAX); 3];

    for i in 0..n {
        let a = &data[indices[i] * ndims..(indices[i] + 1) * ndims];

        // Reset top-k.
        for t in top[..actual_k].iter_mut() {
            *t = (u32::MAX, f32::MAX);
        }
        let threshold_idx = actual_k - 1;

        for j in 0..n {
            if j == i {
                continue;
            }
            let b = &data[indices[j] * ndims..(indices[j] + 1) * ndims];
            let d = dist_fn.call(a, b);
            if d < top[threshold_idx].1 {
                top[threshold_idx] = (j as u32, d);
                for t in (1..actual_k).rev() {
                    if top[t].1 < top[t - 1].1 {
                        top.swap(t, t - 1);
                    } else {
                        break;
                    }
                }
            }
        }

        for t in 0..actual_k {
            local_edges.push((i, top[t].0 as usize, top[t].1));
        }
    }

    // Create bidirected edges. For symmetric metrics, we need the reverse distance.
    // Recompute for the reverse edge (cheaper than storing n×n matrix).
    let mut edges = Vec::with_capacity(local_edges.len() * 2);
    // Use a simple HashSet-like dedup via sorted insert to a small buffer.
    // For n≤256 and k≤3, the edge count is small enough for a flat scan.
    let mut seen_set: Vec<(usize, usize)> = Vec::with_capacity(local_edges.len() * 2);

    for &(src_local, dst_local, dist) in &local_edges {
        let src_global = indices[src_local];
        let dst_global = indices[dst_local];

        if !seen_set.contains(&(src_local, dst_local)) {
            seen_set.push((src_local, dst_local));
            edges.push(Edge {
                src: src_global,
                dst: dst_global,
                distance: dist,
            });
        }
        if !seen_set.contains(&(dst_local, src_local)) {
            seen_set.push((dst_local, src_local));
            // Compute reverse distance.
            let b = &data[dst_global * ndims..(dst_global + 1) * ndims];
            let a = &data[src_global * ndims..(src_global + 1) * ndims];
            let rev_dist = dist_fn.call(b, a);
            edges.push(Edge {
                src: dst_global,
                dst: src_global,
                distance: rev_dist,
            });
        }
    }

    edges
}

/// Build a leaf using 1-bit quantized vectors with Hamming distance.
pub fn build_leaf_quantized(
    qdata: &crate::quantize::QuantizedData,
    indices: &[usize],
    k: usize,
) -> Vec<Edge> {
    let n = indices.len();
    if n <= 1 {
        return Vec::new();
    }

    QUANT_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        let u64s = qdata.u64s_per_vec();
        let nn = n * n;

        // Ensure buffers are large enough, reusing across leaves.
        if bufs.local_u64.len() < n * u64s {
            bufs.local_u64.resize(n * u64s, 0);
        }
        if bufs.dist_matrix.len() < nn {
            bufs.dist_matrix.resize(nn, 0.0);
        }
        if bufs.seen.len() < nn {
            bufs.seen.resize(nn, false);
        }

        // Destructure for simultaneous mutable borrows.
        let QuantLeafBuffers {
            local_u64,
            dist_matrix,
            seen,
        } = &mut *bufs;

        // Gather contiguous u64 data.
        let local = &mut local_u64[..n * u64s];
        for (i, &idx) in indices.iter().enumerate() {
            local[i * u64s..(i + 1) * u64s].copy_from_slice(qdata.get_u64(idx));
        }

        // Compute all-pairs Hamming distance in-place.
        let dist = &mut dist_matrix[..nn];
        let local_ptr = local.as_ptr();
        let dist_ptr = dist.as_mut_ptr();
        for i in 0..n {
            // SAFETY: `i` is in 0..n, so `i * n + i` is within the n*n-element dist buffer.
            unsafe {
                *dist_ptr.add(i * n + i) = f32::MAX;
            }
            // SAFETY: `i * u64s` is within the `n * u64s`-element local buffer.
            let a_base = unsafe { local_ptr.add(i * u64s) };
            for j in (i + 1)..n {
                // SAFETY: `j * u64s` is within the `n * u64s`-element local buffer.
                let b_base = unsafe { local_ptr.add(j * u64s) };
                let mut h = 0u32;
                for k_idx in 0..u64s {
                    // SAFETY: `k_idx` is in 0..u64s, so `a_base.add(k_idx)` and
                    // `b_base.add(k_idx)` are within their respective vector slices.
                    unsafe {
                        h += (*a_base.add(k_idx) ^ *b_base.add(k_idx)).count_ones();
                    }
                }
                let d = h as f32;
                // SAFETY: `i < j < n`, so both `i * n + j` and `j * n + i` are
                // within the n*n-element dist buffer.
                unsafe {
                    *dist_ptr.add(i * n + j) = d;
                    *dist_ptr.add(j * n + i) = d;
                }
            }
        }

        let local_edges = extract_knn(dist, n, k);
        let seen = &mut seen[..nn];
        seen.fill(false);
        make_bidirected_edges(&local_edges, dist, n, indices, seen)
    })
}

/// Convert k-NN edges to bi-directed global edges, deduplicating via a seen buffer.
/// For symmetric metrics, dist(a,b) == dist(b,a) but we use the matrix lookup
/// for the reverse edge to stay correct for any future asymmetric metric.
fn make_bidirected_edges(
    local_edges: &[(usize, usize, f32)],
    dist_matrix: &[f32],
    n: usize,
    indices: &[usize],
    seen: &mut [bool],
) -> Vec<Edge> {
    let mut global_edges = Vec::with_capacity(local_edges.len() * 2);
    for &(src, dst, dist) in local_edges {
        if !seen[src * n + dst] {
            seen[src * n + dst] = true;
            global_edges.push(Edge {
                src: indices[src],
                dst: indices[dst],
                distance: dist,
            });
        }
        if !seen[dst * n + src] {
            seen[dst * n + src] = true;
            global_edges.push(Edge {
                src: indices[dst],
                dst: indices[src],
                distance: dist_matrix[dst * n + src],
            });
        }
    }
    global_edges
}

/// Fill `bufs.dist_matrix[..n*n]` with the leaf's all-pairs distance matrix via GEMM
/// (gather → norms → A·Aᵀ → metric conversion), diagonal set to `f32::MAX`. Mirrors the
/// distance computation in [`build_leaf_with_buffers`] without the k-NN extraction.
fn gemm_fill_dist_matrix<T: VectorRepr>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
    metric: Metric,
    bufs: &mut LeafBuffers,
) {
    let n = indices.len();
    {
        let local = &mut bufs.local_data[..n * ndims];
        for (i, &idx) in indices.iter().enumerate() {
            T::as_f32_into(
                &data[idx * ndims..(idx + 1) * ndims],
                &mut local[i * ndims..(i + 1) * ndims],
            )
            .expect("f32 conversion");
        }
    }
    if !matches!(metric, Metric::CosineNormalized | Metric::InnerProduct) {
        let LeafBuffers {
            local_data,
            norms_sq,
            ..
        } = &mut *bufs;
        let local = &local_data[..n * ndims];
        for i in 0..n {
            let mut s = 0.0f32;
            for &v in &local[i * ndims..(i + 1) * ndims] {
                s += v * v;
            }
            norms_sq[i] = s;
        }
    }
    {
        let LeafBuffers {
            local_data,
            dot_matrix,
            ..
        } = &mut *bufs;
        crate::gemm::sgemm_aat(&local_data[..n * ndims], n, ndims, &mut dot_matrix[..n * n]);
    }
    let LeafBuffers {
        norms_sq,
        dot_matrix,
        dist_matrix,
        ..
    } = &mut *bufs;
    let norms = &norms_sq[..n];
    let dots = &dot_matrix[..n * n];
    let dm = &mut dist_matrix[..n * n];
    match metric {
        Metric::CosineNormalized => {
            for i in 0..n {
                for j in 0..n {
                    dm[i * n + j] = (1.0 - dots[i * n + j]).max(0.0);
                }
                dm[i * n + i] = f32::MAX;
            }
        }
        Metric::Cosine => {
            for i in 0..n {
                let ni = norms[i].sqrt();
                for j in 0..n {
                    let den = ni * norms[j].sqrt();
                    let cs = if den > 0.0 { dots[i * n + j] / den } else { 0.0 };
                    dm[i * n + j] = (1.0 - cs).max(0.0);
                }
                dm[i * n + i] = f32::MAX;
            }
        }
        Metric::L2 => {
            for i in 0..n {
                let ni = norms[i];
                for j in 0..n {
                    dm[i * n + j] = (ni + norms[j] - 2.0 * dots[i * n + j]).max(0.0);
                }
                dm[i * n + i] = f32::MAX;
            }
        }
        Metric::InnerProduct => {
            for i in 0..n {
                for j in 0..n {
                    dm[i * n + j] = -dots[i * n + j];
                }
                dm[i * n + i] = f32::MAX;
            }
        }
    }
}

/// Per-point RobustPrune over a precomputed leaf distance matrix `dm` (n×n, row-major,
/// diagonal = `f32::MAX`). For each leaf point, the candidate set is either every other
/// leaf-mate (`leaf_k = None`, Exp 1) or its `leaf_k` nearest (`Some(k)`, Exp 2); that set
/// is occlusion-pruned to `max_degree` via [`crate::prune::robust_prune_occlude`] using
/// matrix lookups. Emits directed `Edge { src=point, dst=kept_neighbor }`.
fn prune_leaf_from_matrix(
    dm: &[f32],
    n: usize,
    indices: &[usize],
    leaf_k: Option<usize>,
    max_degree: usize,
    alpha: f32,
    leaf_prune: bool,
) -> Vec<Edge> {
    let mut edges = Vec::with_capacity(n * max_degree.min(n));
    let mut cand_local: Vec<u32> = Vec::with_capacity(n);
    let mut node_dists: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let row = &dm[i * n..(i + 1) * n];
        cand_local.clear();
        cand_local.extend((0..n as u32).filter(|&j| j as usize != i));
        if let Some(k) = leaf_k {
            let k = k.min(cand_local.len());
            if k > 0 && k < cand_local.len() {
                cand_local.select_nth_unstable_by(k - 1, |&a, &b| {
                    row[a as usize]
                        .partial_cmp(&row[b as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                cand_local.truncate(k);
            }
        }
        let nc = cand_local.len();
        if nc == 0 {
            continue;
        }
        node_dists.clear();
        node_dists.extend(cand_local.iter().map(|&j| row[j as usize]));
        if leaf_prune {
            // RobustPrune in the leaf (Exp 1 / Exp 2).
            let selected = crate::prune::robust_prune_occlude(
                nc,
                &node_dists[..nc],
                max_degree,
                alpha,
                false,
                |a, b| dm[cand_local[a] as usize * n + cand_local[b] as usize],
            );
            for s in selected {
                let lj = cand_local[s as usize] as usize;
                edges.push(Edge {
                    src: indices[i],
                    dst: indices[lj],
                    distance: node_dists[s as usize],
                });
            }
        } else {
            // No leaf prune (paper's bi-directed-kNN-style control): emit the top-k
            // candidates directly and let the single global final RobustPrune decide.
            for (ci, &lj) in cand_local.iter().enumerate() {
                edges.push(Edge {
                    src: indices[i],
                    dst: indices[lj as usize],
                    distance: node_dists[ci],
                });
            }
        }
    }
    edges
}

/// Exp 1 leaf builder: all-to-all RobustPrune, **no GEMM**.
///
/// Candidate set per point = every other leaf-mate. The n×n leaf distance matrix is
/// built with direct pairwise SIMD distance (no GEMM, no top-k extraction); each point
/// is then occlusion-pruned to `max_degree`. Symmetry/merge across leaves is handled
/// downstream by the candidate accumulator + final RobustPrune.
pub fn build_leaf_robust_no_gemm<T: VectorRepr>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
    max_degree: usize,
    metric: Metric,
    alpha: f32,
    bufs: &mut LeafBuffers,
) -> Vec<Edge> {
    let n = indices.len();
    if n <= 1 {
        return Vec::new();
    }
    bufs.ensure_capacity(n, ndims);
    {
        let local = &mut bufs.local_data[..n * ndims];
        for (i, &idx) in indices.iter().enumerate() {
            T::as_f32_into(
                &data[idx * ndims..(idx + 1) * ndims],
                &mut local[i * ndims..(i + 1) * ndims],
            )
            .expect("f32 conversion");
        }
    }
    let dist_fn = <f32 as DistanceProvider<f32>>::distance_comparer(metric, Some(ndims));
    {
        let LeafBuffers {
            local_data,
            dist_matrix,
            ..
        } = &mut *bufs;
        let local = &local_data[..n * ndims];
        let dm = &mut dist_matrix[..n * n];
        for i in 0..n {
            dm[i * n + i] = f32::MAX;
            let a = &local[i * ndims..(i + 1) * ndims];
            for j in (i + 1)..n {
                let d = dist_fn.call(a, &local[j * ndims..(j + 1) * ndims]);
                dm[i * n + j] = d;
                dm[j * n + i] = d;
            }
        }
    }
    let dm = &bufs.dist_matrix[..n * n];
    prune_leaf_from_matrix(dm, n, indices, None, max_degree, alpha, true)
}

/// Exp 2 leaf builder: GEMM all-pairs → top-`leaf_k` candidates → (optionally) RobustPrune
/// to `max_degree`. `leaf_prune=true` = Exp 2 (RobustPrune in leaf); `leaf_prune=false` =
/// the paper's bi-directed-kNN-style control (keep top-k, prune only at the global merge).
#[allow(clippy::too_many_arguments)]
pub fn build_leaf_gemm_topk_robust<T: VectorRepr>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
    leaf_k: usize,
    max_degree: usize,
    metric: Metric,
    alpha: f32,
    leaf_prune: bool,
    bufs: &mut LeafBuffers,
) -> Vec<Edge> {
    let n = indices.len();
    if n <= 1 {
        return Vec::new();
    }
    bufs.ensure_capacity(n, ndims);
    gemm_fill_dist_matrix(data, ndims, indices, metric, bufs);
    let dm = &bufs.dist_matrix[..n * n];
    prune_leaf_from_matrix(dm, n, indices, Some(leaf_k), max_degree, alpha, leaf_prune)
}

#[cfg(test)]
mod robust_leaf_tests {
    use super::*;
    use diskann_vector::distance::Metric;

    fn invariants(edges: &[Edge], indices: &[usize], max_degree: usize) {
        use std::collections::HashMap;
        let leaf: std::collections::HashSet<usize> = indices.iter().copied().collect();
        let mut deg: HashMap<usize, usize> = HashMap::new();
        for e in edges {
            assert_ne!(e.src, e.dst, "no self loops");
            assert!(leaf.contains(&e.src) && leaf.contains(&e.dst), "edges within leaf");
            *deg.entry(e.src).or_insert(0) += 1;
        }
        for (_s, d) in deg {
            assert!(d <= max_degree, "degree {} exceeds max_degree {}", d, max_degree);
        }
    }

    #[test]
    fn test_leaf_robust_no_gemm_invariants_and_diversity() {
        // 5 points: 0=(0,0),1=(1,0),2=(0,1),3=(0.05,0) near-dup of 1,4=(5,5) far.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.05, 0.0, 5.0, 5.0];
        let indices = vec![0usize, 1, 2, 3, 4];
        let mut bufs = LeafBuffers::new();
        let edges = build_leaf_robust_no_gemm(&data, 2, &indices, 8, Metric::L2, 1.0, &mut bufs);
        assert!(!edges.is_empty());
        invariants(&edges, &indices, 8);
        // From point 1=(1,0) at alpha=1.0: 3=(0.05,0) is far closer to 1 than 0/2 are,
        // so the directed edges from 1 should include 3 (its nearest). Sanity: 1 has edges.
        assert!(edges.iter().any(|e| e.src == 1), "point 1 should emit edges");
    }

    #[test]
    fn test_leaf_gemm_topk_robust_invariants() {
        // 12 random-ish points in 3D; top-k=5, max_degree 3.
        let mut data = Vec::new();
        for i in 0..12 {
            data.extend_from_slice(&[(i as f32) * 0.3, ((i * 7) % 5) as f32, (i as f32).sin()]);
        }
        let indices: Vec<usize> = (0..12).collect();
        let mut bufs = LeafBuffers::new();
        let edges = build_leaf_gemm_topk_robust(&data, 3, &indices, 5, 3, Metric::L2, 1.2, true, &mut bufs);
        assert!(!edges.is_empty());
        invariants(&edges, &indices, 3);
    }

    #[test]
    fn test_leaf_robust_no_gemm_matches_topk_when_k_covers_leaf() {
        // With leaf_k >= n-1, Exp2's candidate set == all leaf-mates == Exp1's set, so
        // the two builders must produce identical edge sets (same matrix source aside).
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 0.5];
        let indices = vec![0usize, 1, 2, 3, 4];
        let mut b1 = LeafBuffers::new();
        let mut b2 = LeafBuffers::new();
        let e1 = build_leaf_robust_no_gemm(&data, 2, &indices, 4, Metric::L2, 1.2, &mut b1);
        let e2 = build_leaf_gemm_topk_robust(&data, 2, &indices, 100, 4, Metric::L2, 1.2, true, &mut b2);
        let s1: std::collections::HashSet<(usize, usize)> = e1.iter().map(|e| (e.src, e.dst)).collect();
        let s2: std::collections::HashSet<(usize, usize)> = e2.iter().map(|e| (e.src, e.dst)).collect();
        assert_eq!(s1, s2, "all-pairs (no-gemm) and full-k (gemm) prune must agree");
    }
}

/// Brute-force search the dataset using L2 distance.
///
/// Returns the `k` nearest neighbor indices and distances for the query.
pub fn brute_force_knn(
    data: &[f32],
    ndims: usize,
    npoints: usize,
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut dists: Vec<(usize, f32)> = (0..npoints)
        .map(|i| {
            let point = &data[i * ndims..(i + 1) * ndims];
            let dist = SquaredL2::evaluate(point, query);
            (i, dist)
        })
        .collect();

    let actual_k = k.min(npoints);
    if actual_k > 0 && actual_k < dists.len() {
        dists.select_nth_unstable_by(actual_k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        dists.truncate(actual_k);
    }
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_vector::distance::{DistanceProvider, Metric};

    #[test]
    fn test_gemm_aat() {
        // 2x3 matrix:
        // [1 2 3]
        // [4 5 6]
        // A * A^T should be:
        // [14 32]
        // [32 77]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 4];
        crate::gemm::sgemm_aat(&a, 2, 3, &mut result);

        assert!((result[0] - 14.0).abs() < 1e-6);
        assert!((result[1] - 32.0).abs() < 1e-6);
        assert!((result[2] - 32.0).abs() < 1e-6);
        assert!((result[3] - 77.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_l2() {
        let dist_fn = <f32 as DistanceProvider<f32>>::distance_comparer(Metric::L2, Some(2));
        let p0 = [0.0f32, 0.0];
        let p1 = [1.0f32, 0.0];
        let p2 = [0.0f32, 1.0];
        // dist(0,1) = 1
        assert!((dist_fn.call(&p0, &p1) - 1.0).abs() < 1e-6);
        // dist(0,2) = 1
        assert!((dist_fn.call(&p0, &p2) - 1.0).abs() < 1e-6);
        // dist(1,2) = 2
        assert!((dist_fn.call(&p1, &p2) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_leaf() {
        let data = vec![
            0.0, 0.0, // point 0
            1.0, 0.0, // point 1
            0.0, 1.0, // point 2
            1.0, 1.0, // point 3
        ];
        let indices = vec![0, 1, 2, 3];

        let edges = build_leaf(&data, 2, &indices, 2, Metric::L2);

        assert!(!edges.is_empty());

        for edge in &edges {
            assert!(edge.src < 4);
            assert!(edge.dst < 4);
            assert!(edge.src != edge.dst);
            assert!(edge.distance >= 0.0);
        }
    }

    #[test]
    fn test_extract_knn() {
        let dist = vec![f32::MAX, 1.0, 4.0, 1.0, f32::MAX, 1.0, 4.0, 1.0, f32::MAX];
        let edges = extract_knn(&dist, 3, 1);

        assert_eq!(edges.len(), 3);

        let p0_edges: Vec<_> = edges.iter().filter(|e| e.0 == 0).collect();
        assert_eq!(p0_edges.len(), 1);
        assert_eq!(p0_edges[0].1, 1);

        let p2_edges: Vec<_> = edges.iter().filter(|e| e.0 == 2).collect();
        assert_eq!(p2_edges.len(), 1);
        assert_eq!(p2_edges[0].1, 1);
    }

    #[test]
    fn test_brute_force_knn() {
        let data = vec![
            0.0, 0.0, // point 0
            1.0, 0.0, // point 1
            0.0, 1.0, // point 2
            1.0, 1.0, // point 3
        ];
        let query = vec![0.1, 0.1];
        let results = brute_force_knn(&data, 2, 4, &query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_build_leaf_cosine() {
        // Verify that cosine distance path works correctly with normalized vectors.
        let mut data = vec![
            1.0, 0.0, // point 0: along x
            0.0, 1.0, // point 1: along y
            0.707, 0.707, // point 2: 45 degrees
            -1.0, 0.0, // point 3: negative x
        ];
        // Normalize all vectors.
        for i in 0..4 {
            let row = &mut data[i * 2..(i + 1) * 2];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        let indices = vec![0, 1, 2, 3];
        let edges = build_leaf(&data, 2, &indices, 2, Metric::CosineNormalized);

        assert!(!edges.is_empty(), "cosine leaf should produce edges");

        for edge in &edges {
            assert!(edge.src < 4);
            assert!(edge.dst < 4);
            assert_ne!(edge.src, edge.dst);
            // Cosine distance for normalized vectors is in [0, 2].
            assert!(edge.distance >= 0.0, "negative cosine distance");
        }
    }

    #[test]
    fn test_build_leaf_cosine_normalized_matches_unit_norm_l2_neighbors() {
        // CosineNormalized assumes unit vectors. On unit vectors, squared L2
        // and 1-dot induce the same ordering, so the directed neighbor set
        // should match the L2 path even if the optimized cosine-normalized path
        // skips explicit norm computation.
        let mut data = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.8, 0.6, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        for row in data.chunks_exact_mut(3) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in row.iter_mut() {
                *v /= norm;
            }
        }

        let indices = vec![0, 1, 2, 3, 4];
        let cosine_edges = build_leaf(&data, 3, &indices, 2, Metric::CosineNormalized);
        let l2_edges = build_leaf(&data, 3, &indices, 2, Metric::L2);

        let cosine_set: std::collections::HashSet<(usize, usize)> =
            cosine_edges.iter().map(|e| (e.src, e.dst)).collect();
        let l2_set: std::collections::HashSet<(usize, usize)> =
            l2_edges.iter().map(|e| (e.src, e.dst)).collect();

        assert_eq!(cosine_set, l2_set);
    }

    #[test]
    fn test_build_leaf_quantized() {
        // Build a leaf using quantized data and verify basic correctness.
        let ndims = 64;
        let npoints = 10;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let (shift, inverse_scale) = {
            use diskann_quantization::scalar::train::ScalarQuantizationParameters;
            use diskann_utils::views::MatrixView;
            let dm = MatrixView::try_from(data.as_slice(), npoints, ndims).unwrap();
            let q = ScalarQuantizationParameters::default().train(dm);
            let s = q.scale();
            (q.shift().to_vec(), if s == 0.0 { 1.0 } else { 1.0 / s })
        };
        let qdata = crate::quantize::quantize_1bit(&data, npoints, ndims, &shift, inverse_scale);
        let indices: Vec<usize> = (0..npoints).collect();
        let edges = build_leaf_quantized(&qdata, &indices, 3);

        assert!(!edges.is_empty(), "quantized leaf should produce edges");

        for edge in &edges {
            assert!(edge.src < npoints, "src {} out of range", edge.src);
            assert!(edge.dst < npoints, "dst {} out of range", edge.dst);
            assert_ne!(edge.src, edge.dst);
            assert!(edge.distance >= 0.0);
        }
    }

    #[test]
    fn test_build_leaf_single_point() {
        // A leaf with 1 point should produce no edges.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let indices = vec![0];
        let edges = build_leaf(&data, 4, &indices, 3, Metric::L2);
        assert!(
            edges.is_empty(),
            "single point leaf should produce 0 edges, got {}",
            edges.len()
        );
    }

    #[test]
    fn test_build_leaf_two_points() {
        // A leaf with 2 points should produce bidirectional edges.
        let data = vec![0.0f32, 0.0, 1.0, 0.0];
        let indices = vec![0, 1];
        let edges = build_leaf(&data, 2, &indices, 3, Metric::L2);
        assert!(!edges.is_empty(), "two point leaf should produce edges");

        // Should have both directions: 0->1 and 1->0.
        let has_0_to_1 = edges.iter().any(|e| e.src == 0 && e.dst == 1);
        let has_1_to_0 = edges.iter().any(|e| e.src == 1 && e.dst == 0);
        assert!(has_0_to_1, "should have edge 0 -> 1");
        assert!(has_1_to_0, "should have edge 1 -> 0");
    }

    #[test]
    fn test_build_leaf_k_equals_n() {
        // k >= n, every point should connect to every other.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];
        let n = indices.len();
        // k = n means each point gets n-1 nearest neighbors = all others.
        let edges = build_leaf(&data, 2, &indices, n, Metric::L2);

        // Collect directed edges.
        let edge_set: std::collections::HashSet<(usize, usize)> =
            edges.iter().map(|e| (e.src, e.dst)).collect();

        // Every pair (i, j) with i != j should be present.
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        edge_set.contains(&(i, j)),
                        "k >= n: edge ({} -> {}) should exist",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_build_leaf_with_buffers_reuse() {
        // Call build_leaf_with_buffers twice and verify buffers are reused.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];
        let mut bufs = LeafBuffers::new();

        let edges1 = build_leaf_with_buffers(&data, 2, &indices, 2, Metric::L2, &mut bufs);
        assert!(!edges1.is_empty(), "first call should produce edges");

        // Verify buffers are allocated.
        assert!(
            !bufs.local_data.is_empty(),
            "buffers should be allocated after first call"
        );

        // Second call with same data should still work.
        let edges2 = build_leaf_with_buffers(&data, 2, &indices, 2, Metric::L2, &mut bufs);
        assert_eq!(
            edges1.len(),
            edges2.len(),
            "same input should produce same number of edges with reused buffers"
        );
    }

    #[test]
    fn test_extract_knn_k_larger_than_n() {
        // k > n-1 should be clamped.
        let dist = vec![f32::MAX, 1.0, 1.0, f32::MAX];
        let edges = extract_knn(&dist, 2, 100); // k=100 but only 2 points
        assert_eq!(
            edges.len(),
            2,
            "k > n-1 should be clamped, each point gets 1 neighbor, total 2 edges"
        );
    }

    #[test]
    fn test_brute_force_knn_single_point() {
        let data = vec![5.0f32, 10.0];
        let query = vec![5.0, 10.0];
        let results = brute_force_knn(&data, 2, 1, &query, 5);
        assert_eq!(
            results.len(),
            1,
            "brute force on 1 point should return 1 result"
        );
        assert_eq!(results[0].0, 0, "should return the only point (index 0)");
        assert!(
            results[0].1 < 1e-6,
            "distance to identical query should be near zero"
        );
    }

    #[test]
    fn test_brute_force_knn_identity() {
        // query = data point, first result should be self with distance 0.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let query = vec![1.0, 0.0]; // same as point 1
        let results = brute_force_knn(&data, 2, 4, &query, 3);
        assert_eq!(
            results[0].0, 1,
            "query identical to point 1 should find it first"
        );
        assert!(
            results[0].1 < 1e-6,
            "self-distance should be 0, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_edge_symmetry() {
        // Verify that build_leaf produces bi-directed edges:
        // if (a -> b) exists, then (b -> a) should also exist.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let indices = vec![0, 1, 2, 3, 4];
        let edges = build_leaf(&data, 2, &indices, 2, Metric::L2);

        // Collect directed edges as a set.
        let edge_set: std::collections::HashSet<(usize, usize)> =
            edges.iter().map(|e| (e.src, e.dst)).collect();

        // For every edge (a, b), (b, a) should also exist.
        for edge in &edges {
            assert!(
                edge_set.contains(&(edge.dst, edge.src)),
                "edge ({} -> {}) exists but reverse ({} -> {}) does not",
                edge.src,
                edge.dst,
                edge.dst,
                edge.src
            );
        }
    }

    #[test]
    fn test_build_leaf_cosine_unnormalized() {
        // Cosine (unnormalized) path: distance = 1 - dot(a,b)/(|a|*|b|).
        // Vectors with different norms but same direction should have distance ~0.
        let data = vec![
            1.0, 0.0, // point 0: unit x
            3.0, 0.0, // point 1: 3x in same direction
            0.0, 1.0, // point 2: unit y (orthogonal)
            1.0, 1.0, // point 3: 45 degrees
        ];
        let indices = vec![0, 1, 2, 3];
        let edges = build_leaf(&data, 2, &indices, 2, Metric::Cosine);

        assert!(!edges.is_empty());
        // Points 0 and 1 are co-linear — cosine distance should be ~0.
        let e01 = edges.iter().find(|e| e.src == 0 && e.dst == 1);
        assert!(e01.is_some(), "co-linear points should be neighbors");
        assert!(
            e01.unwrap().distance < 0.01,
            "cosine dist between co-linear should be ~0, got {}",
            e01.unwrap().distance
        );
    }

    #[test]
    fn test_build_leaf_inner_product() {
        // InnerProduct: distance = -dot(a,b). Lower (more negative) = closer.
        let data = vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // dot with self = 2, dot with (1,0) = 1
        ];
        let indices = vec![0, 1, 2];
        let edges = build_leaf(&data, 2, &indices, 1, Metric::InnerProduct);
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_build_leaf_large_k_clamped() {
        // k=1000 on 5 points should produce all-pairs edges (clamped to n-1=4).
        let data = vec![0.0f32; 5 * 4];
        let indices = vec![0, 1, 2, 3, 4];
        let edges = build_leaf(&data, 4, &indices, 1000, Metric::L2);
        let edge_set: std::collections::HashSet<(usize, usize)> =
            edges.iter().map(|e| (e.src, e.dst)).collect();
        // All pairs should exist.
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert!(
                        edge_set.contains(&(i, j)),
                        "all-pairs edge ({}, {}) missing",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_build_leaf_distances_nonnegative() {
        // All distance metrics should produce non-negative distances.
        let data = vec![-1.5, 2.3, 0.1, 0.7, -0.4, 1.9, 1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2];
        for metric in [Metric::L2, Metric::Cosine, Metric::CosineNormalized] {
            let edges = build_leaf(&data, 3, &indices, 2, metric);
            for e in &edges {
                assert!(
                    e.distance >= 0.0,
                    "{:?}: negative distance {} for ({},{})",
                    metric,
                    e.distance,
                    e.src,
                    e.dst
                );
            }
        }
    }

    #[test]
    fn test_extract_knn_k_zero() {
        let dist = vec![f32::MAX, 1.0, 1.0, f32::MAX];
        let edges = extract_knn(&dist, 2, 0);
        assert!(edges.is_empty(), "k=0 should return no edges");
    }

    #[test]
    fn test_build_leaf_buffer_reuse_different_sizes() {
        // First call with large leaf, second with small — buffers should handle both.
        let data_large = vec![1.0f32; 20 * 4];
        let indices_large: Vec<usize> = (0..20).collect();
        let edges1 = build_leaf(&data_large, 4, &indices_large, 2, Metric::L2);
        assert!(!edges1.is_empty());

        // Second call with smaller leaf on same thread — should reuse thread-local buffers.
        let data_small = vec![1.0f32; 4 * 4];
        let indices_small: Vec<usize> = (0..4).collect();
        let edges2 = build_leaf(&data_small, 4, &indices_small, 2, Metric::L2);
        assert!(
            !edges2.is_empty(),
            "small leaf after large should work with reused buffers"
        );
    }
}
