/*
 * Benchmark: partition v1 (GEMM) vs v2 (point-by-point SIMD) on Enron 1M.
 *
 * Usage: cargo run --release -p diskann-pipnn --example partition_bench -- <path_to_enron_fp16.bin>
 */

use std::time::Instant;

use diskann::utils::VectorRepr;
use diskann_pipnn::partition::PartitionConfig;
use half::f16;

fn load_fp16_bin(path: &str) -> (Vec<f16>, usize, usize) {
    let bytes = std::fs::read(path).expect("failed to read file");
    // DiskANN binary format: 4 bytes npoints (u32) + 4 bytes ndims (u32) + data
    let npoints = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data_bytes = &bytes[8..];
    assert_eq!(
        data_bytes.len(),
        npoints * ndims * 2,
        "file size mismatch: expected {} bytes, got {}",
        npoints * ndims * 2,
        data_bytes.len()
    );
    let data: Vec<f16> = data_bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect();
    (data, npoints, ndims)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: partition_bench <path_to_enron_fp16.bin>");

    println!("Loading {}...", path);
    let (data, npoints, ndims) = load_fp16_bin(path);
    println!("Loaded: {} points x {} dims", npoints, ndims);

    let config = PartitionConfig {
        c_max: 256,
        c_min: 128,
        p_samp: 0.05,
        fanout: vec![8, 3],
        metric: diskann_vector::distance::Metric::CosineNormalized,
        leader_cap: 1000,
    };

    let seed = 42u64;
    let num_threads = 16;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    for p_samp in [0.005, 0.01, 0.05, 0.1, 0.2] {
        let cfg = PartitionConfig {
            p_samp,
            ..config.clone()
        };
        println!("\n=== p_samp={} ===", p_samp);
        let t = Instant::now();
        let leaves = pool.install(|| {
            diskann_pipnn::partition::partition(&data, ndims, npoints, &cfg, seed)
        });
        let elapsed = t.elapsed().as_secs_f64();
        let total_pts: usize = leaves.iter().map(|l| l.indices.len()).sum();
        println!(
            "  Total: {:.3}s  Leaves: {}  Overlap: {:.2}x",
            elapsed, leaves.len(), total_pts as f64 / npoints as f64
        );
    }
}
