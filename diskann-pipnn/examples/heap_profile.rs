//! dhat-rs heap profiler for PiPNN build.
//!
//! Run:
//!   cargo run --release -p diskann-pipnn --example heap_profile --features heap-profiling
//!
//! Produces `dhat-heap.json` — view at https://nnethercote.github.io/dh_view/dh_view.html

#[cfg(feature = "heap-profiling")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use diskann_pipnn::PiPNNConfig;
use half::f16;

fn load_fp16_bin_as_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).expect("failed to read file");
    let npoints = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    // Convert fp16 -> f32 ONCE at load time. Eliminates per-leaf as_f32_into during build.
    let data: Vec<f32> = bytes[8..]
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    assert_eq!(data.len(), npoints * ndims);
    (data, npoints, ndims)
}

fn main() {
    #[cfg(feature = "heap-profiling")]
    let _profiler = dhat::Profiler::new_heap();

    // CLI: heap_profile <data_path> [--c_max N] [--k N] [--l_max N] [--hp N]
    //                   [--threads N] [--fanout 10,3] [--final_prune] [--metric l2|cosine_normalized]
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "/tmp/enron_10m_fp16.bin".to_string());

    let mut c_max: usize = 256;
    let mut c_min: usize = 16;
    let mut leaf_k: usize = 3;
    let mut l_max: usize = 64;
    let mut hp: usize = 14;
    let mut nthreads: usize = 16;
    let mut fanout: Vec<usize> = vec![10, 3];
    let mut final_prune: bool = false;
    let mut metric = diskann_vector::distance::Metric::CosineNormalized;
    let mut p_samp: f64 = 0.005;

    while let Some(a) = args.next() {
        let mut val = || args.next().expect("missing arg value");
        match a.as_str() {
            "--c_max" => c_max = val().parse().unwrap(),
            "--c_min" => c_min = val().parse().unwrap(),
            "--k" => leaf_k = val().parse().unwrap(),
            "--l_max" => l_max = val().parse().unwrap(),
            "--hp" => hp = val().parse().unwrap(),
            "--threads" => nthreads = val().parse().unwrap(),
            "--fanout" => fanout = val().split(',').map(|s| s.parse().unwrap()).collect(),
            "--p_samp" => p_samp = val().parse().unwrap(),
            "--final_prune" => final_prune = true,
            "--metric" => metric = val().parse().unwrap(),
            other => panic!("unknown arg: {other}"),
        }
    }

    eprintln!("Loading {} (converting fp16 → f32 at load)...", path);
    let (data, npoints, ndims) = load_fp16_bin_as_f32(&path);
    eprintln!("Loaded: {} x {} f32 ({} MB)", npoints, ndims, data.len() * 4 / (1024 * 1024));
    eprintln!("Config: c_max={c_max} c_min={c_min} k={leaf_k} l_max={l_max} hp={hp} fanout={fanout:?} \
               final_prune={final_prune} metric={metric:?} threads={nthreads}");

    let config = PiPNNConfig {
        c_max,
        c_min,
        p_samp,
        fanout,
        k: leaf_k,
        max_degree: 64,
        replicas: 1,
        l_max,
        num_hash_planes: hp,
        metric,
        final_prune,
        alpha: 1.2,
        num_threads: nthreads,
        leader_cap: 1000,
        saturate_after_prune: true,
    };

    eprintln!("Building PiPNN index...");
    let graph = diskann_pipnn::builder::build_typed(&data, npoints, ndims, &config)
        .expect("build failed");

    eprintln!(
        "Done. {} nodes, avg_degree={:.1}",
        graph.npoints,
        graph.avg_degree()
    );
    eprintln!("{}", graph.build_stats);
}
