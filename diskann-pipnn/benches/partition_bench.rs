/*
 * Benchmark: RBC partition on real Enron 1M dataset.
 *
 * Run with:
 *   cargo bench -p diskann-pipnn --bench partition_bench
 *
 * Requires Enron dataset at the path below (or set ENRON_DATA env var).
 */

use criterion::{criterion_group, criterion_main, Criterion};
use diskann_pipnn::partition::{self, PartitionConfig};
use half::f16;
use std::time::Duration;

const DEFAULT_DATA_PATH: &str = "/tmp/normalized_dim_384_vector_fp16_1087932_vectors.bin";

fn load_dataset() -> (Vec<f16>, usize, usize) {
    let path = std::env::var("ENRON_DATA").unwrap_or_else(|_| DEFAULT_DATA_PATH.to_string());
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("Cannot read {}: {}", path, e));
    let npoints = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data: Vec<f16> = bytes[8..]
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(data.len(), npoints * ndims);
    eprintln!("Loaded {}x{} fp16 from {}", npoints, ndims, path);
    (data, npoints, ndims)
}

fn make_config() -> PartitionConfig {
    PartitionConfig {
        c_max: 256,
        c_min: 16,
        p_samp: 0.005,
        fanout: vec![8, 3],
        metric: diskann_vector::distance::Metric::CosineNormalized,
        leader_cap: 1000,
    }
}

fn bench_partition(c: &mut Criterion) {
    let (data, npoints, ndims) = load_dataset();
    let config = make_config();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("partition_enron_1m");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function("partition", |b| {
        b.iter(|| {
            pool.install(|| partition::partition(&data, ndims, npoints, &config, 42))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_partition);
criterion_main!(benches);
