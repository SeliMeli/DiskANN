/*
 * Leaf-build micro-benchmark: times each sub-step of build_leaf_with_buffers
 * for various leaf sizes to diagnose why c_max=1024 is 6x slower than c_max=256
 * on 384d CosineNormalized data instead of the expected 3.5x.
 *
 * Usage: cargo run --release -p diskann-pipnn --example leaf_profile -- <enron_fp16.bin>
 */

use half::f16;
use std::time::Instant;

use diskann::utils::VectorRepr;
use diskann_pipnn::gemm::sgemm_aat;
use diskann_pipnn::leaf_build::LeafBuffers;

fn load_fp16_bin(path: &str) -> (Vec<f16>, usize, usize) {
    let bytes = std::fs::read(path).expect("failed to read file");
    let npoints = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data: Vec<f16> = bytes[8..]
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(data.len(), npoints * ndims);
    (data, npoints, ndims)
}

/// Replicates extract_knn from leaf_build.rs (private fn).
fn extract_knn(dist_matrix: &[f32], n: usize, k: usize) -> Vec<(usize, usize, f32)> {
    if n <= 1 || k == 0 {
        return Vec::new();
    }
    let actual_k = k.min(n - 1);
    let mut edges = Vec::with_capacity(n * actual_k);
    let mut indices: Vec<u32> = (0..n as u32).collect();

    for i in 0..n {
        let row = &dist_matrix[i * n..(i + 1) * n];
        for j in 0..n {
            unsafe {
                *indices.get_unchecked_mut(j) = j as u32;
            }
        }
        if actual_k < n {
            indices.select_nth_unstable_by(actual_k - 1, |&a, &b| {
                let da = unsafe { *row.get_unchecked(a as usize) };
                let db = unsafe { *row.get_unchecked(b as usize) };
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        for idx in 0..actual_k {
            let j = unsafe { *indices.get_unchecked(idx) } as usize;
            edges.push((i, j, row[j]));
        }
    }
    edges
}

/// Replicates make_bidirected_edges from leaf_build.rs (private fn).
fn make_bidirected_edges(
    local_edges: &[(usize, usize, f32)],
    dist_matrix: &[f32],
    n: usize,
    indices: &[usize],
    seen: &mut [bool],
) -> Vec<(usize, usize, f32)> {
    let mut global_edges = Vec::with_capacity(local_edges.len() * 2);
    for &(src, dst, dist) in local_edges {
        if !seen[src * n + dst] {
            seen[src * n + dst] = true;
            global_edges.push((indices[src], indices[dst], dist));
        }
        if !seen[dst * n + src] {
            seen[dst * n + src] = true;
            global_edges.push((indices[dst], indices[src], dist_matrix[dst * n + src]));
        }
    }
    global_edges
}

/// All timing results for a single leaf size, in microseconds.
#[derive(Default, Clone)]
struct Timings {
    conv_us: f64,
    norms_us: f64,
    gemm_us: f64,
    dist_us: f64,
    knn_us: f64,
    seen_us: f64,
    bidir_us: f64,
    total_us: f64,
}

fn profile_leaf(
    data: &[f16],
    ndims: usize,
    n: usize,
    k: usize,
    bufs: &mut LeafBuffers,
) -> Timings {
    // Use first n consecutive indices.
    let indices: Vec<usize> = (0..n).collect();

    let t_total = Instant::now();

    // --- Step (a): f16 -> f32 conversion ---
    resize_bufs(bufs, n, ndims);
    let local_data = &mut bufs.local_data[..n * ndims];
    let t0 = Instant::now();
    for (i, &idx) in indices.iter().enumerate() {
        let src = &data[idx * ndims..(idx + 1) * ndims];
        let dst = &mut local_data[i * ndims..(i + 1) * ndims];
        f16::as_f32_into(src, dst).expect("f32 conversion");
    }
    let conv_us = t0.elapsed().as_secs_f64() * 1e6;

    // --- Step (b): Norms computation ---
    let norms_sq = &mut bufs.norms_sq[..n];
    let t1 = Instant::now();
    for i in 0..n {
        let row = &local_data[i * ndims..(i + 1) * ndims];
        let mut norm = 0.0f32;
        for &v in row.iter() {
            norm += v * v;
        }
        norms_sq[i] = norm;
    }
    let norms_us = t1.elapsed().as_secs_f64() * 1e6;

    // --- Step (c): GEMM (sgemm_aat) ---
    let dot_matrix = &mut bufs.dot_matrix[..n * n];
    let t2 = Instant::now();
    sgemm_aat(local_data, n, ndims, dot_matrix);
    let gemm_us = t2.elapsed().as_secs_f64() * 1e6;

    // --- Step (d): Distance matrix conversion (CosineNormalized: 1 - dot) ---
    let t3 = Instant::now();
    for i in 0..n {
        let row = &mut dot_matrix[i * n..(i + 1) * n];
        for val in row.iter_mut() {
            *val = (1.0 - *val).max(0.0);
        }
        row[i] = f32::MAX;
    }
    let dist_us = t3.elapsed().as_secs_f64() * 1e6;

    // dist_matrix is now dot_matrix (in-place for CosineNormalized).
    let dist_matrix = &bufs.dot_matrix[..n * n];

    // --- Step (e): extract_knn (partial sort) ---
    let t4 = Instant::now();
    let local_edges = extract_knn(dist_matrix, n, k);
    let knn_us = t4.elapsed().as_secs_f64() * 1e6;

    // --- Step (f): seen.fill(false) ---
    let seen = &mut bufs.seen[..n * n];
    let t5 = Instant::now();
    seen.fill(false);
    let seen_us = t5.elapsed().as_secs_f64() * 1e6;

    // --- Step (g): make_bidirected_edges ---
    let t6 = Instant::now();
    let _global_edges = make_bidirected_edges(&local_edges, dist_matrix, n, &indices, seen);
    let bidir_us = t6.elapsed().as_secs_f64() * 1e6;

    let total_us = t_total.elapsed().as_secs_f64() * 1e6;

    Timings {
        conv_us,
        norms_us,
        gemm_us,
        dist_us,
        knn_us,
        seen_us,
        bidir_us,
        total_us,
    }
}

/// Resize LeafBuffers to fit n points of ndims dimensions.
/// LeafBuffers::ensure_capacity is private, but all fields are pub,
/// so we replicate the logic as a free function.
fn resize_bufs(bufs: &mut LeafBuffers, n: usize, ndims: usize) {
    let nd = n * ndims;
    let nn = n * n;
    if bufs.local_data.len() < nd {
        bufs.local_data.resize(nd, 0.0);
    }
    if bufs.norms_sq.len() < n {
        bufs.norms_sq.resize(n, 0.0);
    }
    if bufs.dot_matrix.len() < nn {
        bufs.dot_matrix.resize(nn, 0.0);
    }
    if bufs.dist_matrix.len() < nn {
        bufs.dist_matrix.resize(nn, 0.0);
    }
    if bufs.seen.len() < nn {
        bufs.seen.resize(nn, false);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: leaf_profile <enron_fp16.bin>");

    println!("Loading {}...", path);
    let (data, npoints, ndims) = load_fp16_bin(path);
    println!("Loaded: {} points x {} dims\n", npoints, ndims);

    let sizes = [64, 128, 256, 512, 1024];
    let k = 2; // leaf_k used in real builds
    let iters = 10;

    // Verify we have enough data points.
    let max_size = *sizes.last().unwrap();
    assert!(
        npoints >= max_size,
        "dataset has {} points but need at least {}",
        npoints,
        max_size
    );

    // Pre-allocate buffers for the largest size (reused across all sizes like the real code).
    let mut bufs = LeafBuffers::new();
    resize_bufs(&mut bufs, max_size, ndims);

    // Warmup: run the largest size once to warm caches and JIT.
    let _ = profile_leaf(&data, ndims, max_size, k, &mut bufs);

    // Collect all timings first, then print tables.
    let mut all_timings: Vec<(usize, Timings)> = Vec::new();
    for &size in &sizes {
        let mut avg = Timings::default();

        for _ in 0..iters {
            let t = profile_leaf(&data, ndims, size, k, &mut bufs);
            avg.conv_us += t.conv_us;
            avg.norms_us += t.norms_us;
            avg.gemm_us += t.gemm_us;
            avg.dist_us += t.dist_us;
            avg.knn_us += t.knn_us;
            avg.seen_us += t.seen_us;
            avg.bidir_us += t.bidir_us;
            avg.total_us += t.total_us;
        }

        let d = iters as f64;
        avg.conv_us /= d;
        avg.norms_us /= d;
        avg.gemm_us /= d;
        avg.dist_us /= d;
        avg.knn_us /= d;
        avg.seen_us /= d;
        avg.bidir_us /= d;
        avg.total_us /= d;
        all_timings.push((size, avg));
    }

    // Print absolute timings table.
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Size", "Conv", "Norms", "GEMM", "Dist", "KNN", "Seen", "Bidir", "Total"
    );
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "(us)", "(us)", "(us)", "(us)", "(us)", "(us)", "(us)", "(us)"
    );
    println!("{}", "-".repeat(96));

    for (size, avg) in &all_timings {
        println!(
            "{:>6} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            size,
            avg.conv_us,
            avg.norms_us,
            avg.gemm_us,
            avg.dist_us,
            avg.knn_us,
            avg.seen_us,
            avg.bidir_us,
            avg.total_us,
        );
    }

    // Print ratios relative to size=256.
    println!("\n--- Ratios relative to size=256 ---");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Size", "Conv", "Norms", "GEMM", "Dist", "KNN", "Seen", "Bidir", "Total"
    );
    println!("{}", "-".repeat(96));

    // Find the 256 baseline.
    let base = all_timings
        .iter()
        .find(|(s, _)| *s == 256)
        .map(|(_, t)| t.clone())
        .expect("size 256 must be in the list");

    for (size, t) in &all_timings {
        let ratio = |val: f64, base_val: f64| -> String {
            if base_val > 0.0 {
                format!("{:.2}x", val / base_val)
            } else {
                "N/A".to_string()
            }
        };
        println!(
            "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            size,
            ratio(t.conv_us, base.conv_us),
            ratio(t.norms_us, base.norms_us),
            ratio(t.gemm_us, base.gemm_us),
            ratio(t.dist_us, base.dist_us),
            ratio(t.knn_us, base.knn_us),
            ratio(t.seen_us, base.seen_us),
            ratio(t.bidir_us, base.bidir_us),
            ratio(t.total_us, base.total_us),
        );
    }
}
