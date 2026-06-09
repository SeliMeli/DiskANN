/*
 * Scaling benchmark: measures each PiPNN phase independently at various thread counts.
 * Usage: cargo run --release -p diskann-pipnn --example scaling_bench -- <enron_fp16.bin>
 */

use std::time::Instant;
use diskann_pipnn::partition::PartitionConfig;
use half::f16;

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

fn bench_partition(data: &[f16], ndims: usize, npoints: usize, config: &PartitionConfig, num_threads: usize) -> (f64, usize) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    let t = Instant::now();
    let leaves = pool.install(|| {
        diskann_pipnn::partition::partition(data, ndims, npoints, config, 42)
    });
    (t.elapsed().as_secs_f64(), leaves.len())
}

fn bench_leaf_build(data: &[f16], ndims: usize, leaves: &[diskann_pipnn::partition::Leaf], k: usize, metric: diskann_vector::distance::Metric, num_threads: usize) -> (f64, usize) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    let total_edges = std::sync::atomic::AtomicUsize::new(0);
    let t = Instant::now();
    pool.install(|| {
        use rayon::prelude::*;
        leaves.par_iter().for_each(|leaf| {
            let indices_usize: Vec<usize> = leaf.indices.iter().map(|&i| i as usize).collect();
            let edges = diskann_pipnn::leaf_build::build_leaf(data, ndims, &indices_usize, k, metric);
            total_edges.fetch_add(edges.len(), std::sync::atomic::Ordering::Relaxed);
        });
    });
    (t.elapsed().as_secs_f64(), total_edges.load(std::sync::atomic::Ordering::Relaxed))
}

fn bench_hash_prune_insert(data: &[f16], ndims: usize, leaves: &[diskann_pipnn::partition::Leaf], npoints: usize, k: usize, metric: diskann_vector::distance::Metric, num_threads: usize) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

    // Create HashPrune with sketches.
    let hash_prune = pool.install(|| {
        let sketches = diskann_pipnn::hash_prune::LshSketches::new(data, npoints, ndims, 12, 42);
        diskann_pipnn::hash_prune::HashPrune::from_sketches(sketches, npoints, 64, 64)
    });

    // Bench only the insert phase.
    let t = Instant::now();
    pool.install(|| {
        use rayon::prelude::*;
        leaves.par_iter().for_each(|leaf| {
            let indices_usize: Vec<usize> = leaf.indices.iter().map(|&i| i as usize).collect();
            let edges = diskann_pipnn::leaf_build::build_leaf(data, ndims, &indices_usize, k, metric);
            hash_prune.add_edges_batched(&edges);
        });
    });
    t.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: scaling_bench <enron_fp16.bin>");

    println!("Loading {}...", path);
    let (data, npoints, ndims) = load_fp16_bin(path);
    println!("Loaded: {} points x {} dims\n", npoints, ndims);

    let config = PartitionConfig {
        c_max: 256,
        c_min: 128,
        p_samp: 0.01,
        fanout: vec![8, 3],
        metric: diskann_vector::distance::Metric::CosineNormalized,
        leader_cap: 1000,
    };
    let seed = 42u64;

    let thread_counts = [1, 2, 4, 8, 12, 16];

    // Phase 1: Partition scaling
    println!("=== Partition Scaling ===");
    println!("{:>8} {:>10} {:>8}", "Threads", "Time(s)", "Leaves");
    for &t in &thread_counts {
        let (secs, leaves) = bench_partition(&data, ndims, npoints, &config, t);
        println!("{:>8} {:>10.3} {:>8}", t, secs, leaves);
    }

    // Get leaves for subsequent benchmarks (use 16 threads).
    let pool16 = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
    let leaves = pool16.install(|| {
        diskann_pipnn::partition::partition(&data, ndims, npoints, &config, 42)
    });
    println!("\nUsing {} leaves for leaf_build/hash_prune benchmarks\n", leaves.len());

    // Phase 2: Leaf build scaling (without hash_prune)
    println!("=== Leaf Build Scaling (no hash_prune) ===");
    println!("{:>8} {:>10} {:>10}", "Threads", "Time(s)", "Edges");
    for &t in &thread_counts {
        let (secs, edges) = bench_leaf_build(&data, ndims, &leaves, 2, config.metric, t);
        println!("{:>8} {:>10.3} {:>10}", t, secs, edges);
    }

    // Phase 3: Leaf build + hash_prune insert scaling
    println!("\n=== Leaf Build + HashPrune Insert Scaling ===");
    println!("{:>8} {:>10}", "Threads", "Time(s)");
    for &t in &thread_counts {
        let secs = bench_hash_prune_insert(&data, ndims, &leaves, npoints, 2, config.metric, t);
        println!("{:>8} {:>10.3}", t, secs);
    }
}
