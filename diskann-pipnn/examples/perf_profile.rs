//! Single-run profiling binary for `perf stat` / `perf record`.
//!
//! Runs partition and leaf_build each once, with a pause between phases
//! to allow targeted profiling. Env vars control behavior:
//!
//! - `DISKANN_DATASETS` — path to dataset directory (default: ./datasets)
//! - `PIPNN_PROFILE_PHASE` — which phase to run: "partition", "leaf_build", "both" (default)
//! - `PIPNN_PROFILE_LOAD` — which load: "enron_1m" (default), "bigann_1m", etc.
//!
//! Usage:
//! ```sh
//! # Build release with debug symbols:
//! cargo build -p diskann-pipnn --example perf_profile --release
//!
//! # Run with perf stat (hardware counters):
//! perf stat -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,\
//!   L1-dcache-loads,L1-dcache-load-misses \
//!   ./target/release/examples/perf_profile
//!
//! # Run with perf record (call graph):
//! perf record -g -F 997 ./target/release/examples/perf_profile
//! perf report --no-children
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use bytemuck;
use diskann_pipnn::partition::PartitionConfig;
use diskann_pipnn::{leaf_build, partition};
use diskann_vector::distance::Metric;
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;

fn main() {
    let phase = std::env::var("PIPNN_PROFILE_PHASE").unwrap_or_else(|_| "both".into());
    let load_name = std::env::var("PIPNN_PROFILE_LOAD").unwrap_or_else(|_| "enron_1m".into());
    let dataset_dir =
        std::env::var("DISKANN_DATASETS").unwrap_or_else(|_| "datasets".into());

    let (filename, npoints, ndims, metric) = match load_name.as_str() {
        "enron_1m" => ("enron_fp16.bin", 1_087_932usize, 384usize, Metric::CosineNormalized),
        "bigann_1m" => ("bigann_1m_fp16.bin", 1_000_000, 128, Metric::L2),
        "bigann_10m" => ("bigann_10m_fp16.bin", 10_000_000, 128, Metric::L2),
        "enron_10m" => ("enron_10m_fp16.bin", 10_000_000, 384, Metric::CosineNormalized),
        _ => {
            eprintln!("Unknown load: {load_name}");
            std::process::exit(1);
        }
    };

    let path = format!("{dataset_dir}/{filename}");
    eprintln!("Loading {path} ({npoints} x {ndims})...");
    let file = std::fs::File::open(&path).expect("open dataset");
    let mmap = unsafe { Mmap::map(&file).expect("mmap") };
    let data: &[f16] = bytemuck::cast_slice(&mmap[8..8 + npoints * ndims * 2]);

    let c_max: usize = std::env::var("PIPNN_CMAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let p_samp: f64 = std::env::var("PIPNN_PSAMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.005);
    let f0: usize = std::env::var("PIPNN_F0")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let f1: usize = std::env::var("PIPNN_F1")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let pcfg = PartitionConfig {
        c_max,
        c_min: 16,
        p_samp,
        fanout: vec![f0, f1],
        metric,
        leader_cap: 1000,
    };
    let leaf_k: usize = std::env::var("PIPNN_LEAFK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    eprintln!("c_max={c_max} p_samp={p_samp}");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .expect("pool");

    if phase == "qpartition" {
        // Test: sign-bit quantize, then Hamming-based partition.
        eprintln!("=== QUANTIZED PARTITION ===");
        let ndims_aligned = ((ndims + 63) / 64) * 64;
        let bytes_per_vec = ndims_aligned / 8;
        let u64s_per_vec = bytes_per_vec / 8;

        let t0 = Instant::now();
        // Simple sign-bit quantization (no training): bit = (x > 0) ? 1 : 0
        // For CosineNormalized data, this is SimHash.
        let mut bits = vec![0u8; npoints * bytes_per_vec];
        pool.install(|| {
            bits.par_chunks_mut(bytes_per_vec)
                .enumerate()
                .for_each(|(i, out)| {
                    let src = &data[i * ndims..(i + 1) * ndims];
                    for d in 0..ndims {
                        if src[d].to_f32() > 0.0 {
                            out[d / 8] |= 1 << (d % 8);
                        }
                    }
                });
        });
        let quant_secs = t0.elapsed().as_secs_f64();
        eprintln!("Sign-bit quantization: {quant_secs:.3}s");

        let qdata = diskann_pipnn::quantize::QuantizedData::from_raw(
            bits, bytes_per_vec, ndims, npoints,
        );

        let t1 = Instant::now();
        let leaves = pool.install(|| partition::partition_quantized(&qdata, npoints, &pcfg, 42));
        let part_secs = t1.elapsed().as_secs_f64();
        let total_pts: usize = leaves.iter().map(|l| l.indices.len()).sum();
        eprintln!(
            "Quantized partition: {:.3}s, {} leaves, avg_size={:.1}, overlap={:.2}x",
            part_secs,
            leaves.len(),
            total_pts as f64 / leaves.len().max(1) as f64,
            total_pts as f64 / npoints as f64,
        );
        eprintln!("Total (quant+partition): {:.3}s", quant_secs + part_secs);

        // Also run leaf_build on the quantized partition leaves.
        let edges = AtomicUsize::new(0);
        let t2 = Instant::now();
        #[allow(clippy::disallowed_methods)]
        pool.install(|| {
            leaves.par_chunks(64).for_each(|chunk| {
                leaf_build::LEAF_BUFFERS.with(|cell| {
                    let mut bufs = cell.borrow_mut();
                    for leaf in chunk {
                        let indices_usize: Vec<usize> =
                            leaf.indices.iter().map(|&i| i as usize).collect();
                        let e = leaf_build::build_leaf_with_buffers(
                            data, ndims, &indices_usize, leaf_k, metric, &mut bufs,
                        );
                        edges.fetch_add(e.len(), Ordering::Relaxed);
                    }
                });
            });
        });
        let leaf_secs = t2.elapsed().as_secs_f64();
        eprintln!(
            "Leaf build: {:.3}s, {} edges",
            leaf_secs,
            edges.load(Ordering::Relaxed)
        );
        eprintln!("Total (quant+partition+leaf): {:.3}s", quant_secs + part_secs + leaf_secs);
    } else if phase == "partition" || phase == "both" {
        eprintln!("=== PARTITION ===");
        let t0 = Instant::now();
        let leaves = pool.install(|| partition::partition(data, ndims, npoints, &pcfg, 42));
        let secs = t0.elapsed().as_secs_f64();
        let total_pts: usize = leaves.iter().map(|l| l.indices.len()).sum();
        eprintln!(
            "Partition: {:.3}s, {} leaves, avg_size={:.1}, overlap={:.2}x",
            secs,
            leaves.len(),
            total_pts as f64 / leaves.len().max(1) as f64,
            total_pts as f64 / npoints as f64,
        );

        if phase == "both" {
            eprintln!("=== LEAF BUILD ===");
            let edges = AtomicUsize::new(0);
            let t0 = Instant::now();
            #[allow(clippy::disallowed_methods)]
            pool.install(|| {
                leaves.par_iter().for_each(|leaf| {
                    let indices_usize: Vec<usize> =
                        leaf.indices.iter().map(|&i| i as usize).collect();
                    let e = leaf_build::build_leaf(data, ndims, &indices_usize, leaf_k, metric);
                    edges.fetch_add(e.len(), Ordering::Relaxed);
                });
            });
            let secs = t0.elapsed().as_secs_f64();
            eprintln!(
                "Leaf build: {:.3}s, {} edges",
                secs,
                edges.load(Ordering::Relaxed)
            );
        }
    } else if phase == "leaf_build" {
        eprintln!("Building partition fixture for leaf_build...");
        let leaves = pool.install(|| partition::partition(data, ndims, npoints, &pcfg, 42));
        eprintln!("Partition done ({} leaves). Starting leaf build profiling...", leaves.len());

        let edges = AtomicUsize::new(0);
        let t0 = Instant::now();
        // Use par_chunks + build_leaf_with_buffers to test TLS batching.
        #[allow(clippy::disallowed_methods)]
        pool.install(|| {
            leaves.par_chunks(64).for_each(|chunk| {
                leaf_build::LEAF_BUFFERS.with(|cell| {
                    let mut bufs = cell.borrow_mut();
                    for leaf in chunk {
                        let indices_usize: Vec<usize> =
                            leaf.indices.iter().map(|&i| i as usize).collect();
                        let e = leaf_build::build_leaf_with_buffers(
                            data, ndims, &indices_usize, leaf_k, metric, &mut bufs,
                        );
                        edges.fetch_add(e.len(), Ordering::Relaxed);
                    }
                });
            });
        });
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "Leaf build: {:.3}s, {} edges",
            secs,
            edges.load(Ordering::Relaxed)
        );
    }
}
