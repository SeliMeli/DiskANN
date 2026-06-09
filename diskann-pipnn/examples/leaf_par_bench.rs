/*
 * Bench: serialized parallel GEMM for large leaves (mutex + Par::Rayon).
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
    let path = args.get(1).expect("usage: leaf_par_bench <enron_fp16.bin>");
    let (data, npoints, ndims) = load_fp16_bin(path);
    println!("Loaded: {} points x {} dims", npoints, ndims);

    let metric = diskann_vector::distance::Metric::CosineNormalized;
    let config = PartitionConfig {
        c_max: 1024, c_min: 512, p_samp: 0.01, fanout: vec![8, 3], metric, leader_cap: 1000,
    };

    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let leaves = pool8.install(|| {
        diskann_pipnn::partition::partition(&data, ndims, npoints, &config, 42)
    });
    let large = leaves.iter().filter(|l| l.indices.len() >= 512).count();
    println!("{} leaves, {} large(>=512)\n", leaves.len(), large);

    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();

    // Current: build_leaf now has mutex+parallel GEMM for n>=512
    let edges = AtomicUsize::new(0);
    let t = Instant::now();
    pool.install(|| {
        leaves.par_iter().for_each(|leaf| {
            let idx: Vec<usize> = leaf.indices.iter().map(|&i| i as usize).collect();
            let e = diskann_pipnn::leaf_build::build_leaf(&data, ndims, &idx, 2, metric);
            edges.fetch_add(e.len(), Ordering::Relaxed);
        });
    });
    println!("Mutex + par GEMM(n>=512): {:.3}s  ({} edges)", t.elapsed().as_secs_f64(), edges.load(Ordering::Relaxed));
}
