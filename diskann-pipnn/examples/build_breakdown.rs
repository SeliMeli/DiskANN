/*
 * Full build breakdown: c_max=256 vs 1024 with HashPrune.
 * Measures: partition, leaf_build (GEMM-only vs full), hash_prune insert, graph extract.
 */
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use diskann::utils::VectorRepr;
use diskann_pipnn::partition::PartitionConfig;
use half::f16;
use rayon::prelude::*;

fn load_fp16_bin(path: &str) -> (Vec<f16>, usize, usize) {
    let bytes = std::fs::read(path).expect("failed to read file");
    let npoints = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data: Vec<f16> = bytes[8..].chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]])).collect();
    (data, npoints, ndims)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: build_breakdown <enron_fp16.bin>");
    let (data, npoints, ndims) = load_fp16_bin(path);
    println!("Loaded: {} points x {} dims\n", npoints, ndims);

    let metric = diskann_vector::distance::Metric::CosineNormalized;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();

    for c_max in [256, 1024] {
        let config = PartitionConfig {
            c_max, c_min: c_max / 2, p_samp: 0.01, fanout: vec![8, 3], metric, leader_cap: 1000,
        };

        println!("=== c_max={} ===", c_max);

        // Partition
        let t = Instant::now();
        let leaves = pool.install(|| {
            diskann_pipnn::partition::partition(&data, ndims, npoints, &config, 42)
        });
        let part_s = t.elapsed().as_secs_f64();
        println!("  Partition:    {:.3}s  ({} leaves)", part_s, leaves.len());

        // Leaf build only (no hash_prune)
        let edges_count = AtomicUsize::new(0);
        let t = Instant::now();
        pool.install(|| {
            leaves.par_iter().for_each(|leaf| {
                let idx: Vec<usize> = leaf.indices.iter().map(|&i| i as usize).collect();
                let e = diskann_pipnn::leaf_build::build_leaf(&data, ndims, &idx, 2, metric);
                edges_count.fetch_add(e.len(), Ordering::Relaxed);
            });
        });
        let leaf_s = t.elapsed().as_secs_f64();
        let edges = edges_count.load(Ordering::Relaxed);
        println!("  Leaf build:   {:.3}s  ({} edges)", leaf_s, edges);

        // Leaf build + hash_prune insert
        let sketches = pool.install(|| {
            diskann_pipnn::hash_prune::LshSketches::new(&data, npoints, ndims, 14, 42)
        });
        let hash_prune = diskann_pipnn::hash_prune::HashPrune::from_sketches(sketches, npoints, 64, 64);

        let t = Instant::now();
        pool.install(|| {
            leaves.par_iter().for_each(|leaf| {
                let idx: Vec<usize> = leaf.indices.iter().map(|&i| i as usize).collect();
                let e = diskann_pipnn::leaf_build::build_leaf(&data, ndims, &idx, 2, metric);
                hash_prune.add_edges_batched(&e);
            });
        });
        let leaf_hp_s = t.elapsed().as_secs_f64();
        let hp_overhead = leaf_hp_s - leaf_s;
        println!("  Leaf+HP:      {:.3}s  (HP overhead: {:.3}s)", leaf_hp_s, hp_overhead);

        // Graph extract
        let t = Instant::now();
        let _graph = pool.install(|| hash_prune.extract_graph());
        let extract_s = t.elapsed().as_secs_f64();
        println!("  Extract:      {:.3}s", extract_s);
        println!("  Total:        {:.3}s\n", part_s + leaf_hp_s + extract_s);
    }
}
