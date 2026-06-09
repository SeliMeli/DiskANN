use diskann_pipnn::partition::{self, PartitionConfig};
use half::f16;
use std::time::Instant;

const DATA_PATH: &str = "/tmp/normalized_dim_384_vector_fp16_1087932_vectors.bin";

fn main() {
    let bytes = std::fs::read(DATA_PATH).expect("cannot read data");
    let npoints = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data: Vec<f16> = bytes[8..]
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect();
    println!("Loaded {}x{} fp16", npoints, ndims);

    let config = PartitionConfig {
        c_max: 256,
        c_min: 16,
        p_samp: 0.005,
        fanout: vec![8, 3],
        metric: diskann_vector::distance::Metric::CosineNormalized,
        leader_cap: 1000,
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();

    let t = Instant::now();
    let leaves = pool.install(|| partition::partition(&data, ndims, npoints, &config, 42));
    let elapsed = t.elapsed().as_secs_f64();
    println!("partition: {:.3}s, {} leaves", elapsed, leaves.len());
}
