//! In-memory PiPNN build benchmark for BigANN 10M **u8** (AMX-int8 GEMM path).
//!
//! Times the pure in-memory PiPNN build on raw u8 vectors. The faer baseline
//! converts u8 -> f32 before the GEMM; the MKL build feeds raw u8 to AMX-int8.
//!
//! Usage:
//!   cargo run --release -p diskann-pipnn --example inmem_build_bench_u8 -- \
//!     <path-to-u8bin> [npoints] [nthreads]

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::time::Instant;

use diskann_pipnn::builder::build_typed;
use diskann_pipnn::{PiPNNBuildContext, PiPNNConfig};
use diskann_vector::distance::Metric;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Read a `.u8bin` (8-byte header: npoints u32, ndims u32, then raw u8 body).
fn read_u8_dataset(path: &str, limit_points: Option<usize>) -> (Vec<u8>, usize, usize) {
    let mut file = File::open(path).expect("open dataset");
    let mut hdr = [0u8; 8];
    file.read_exact(&mut hdr).expect("read header");
    let npoints_file = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let npoints = limit_points.unwrap_or(npoints_file).min(npoints_file);
    let nelems = npoints * ndims;

    let mut data = vec![0u8; nelems];
    file.seek(SeekFrom::Start(8)).unwrap();
    file.read_exact(&mut data).expect("read body");

    println!(
        "loaded {}/{} points × {} dims ({} MB, u8)",
        npoints,
        npoints_file,
        ndims,
        nelems >> 20
    );
    (data, npoints, ndims)
}

fn read_rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest.trim().split_whitespace().next().unwrap().parse().unwrap_or(0);
        }
    }
    0
}

fn main() {
    let _ = tracing_subscriber::fmt().try_init();

    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/home/weiyaoluo/datasets/bigann/bigann_10m.u8bin".to_string());
    let limit_points = args.get(2).and_then(|s| s.parse::<usize>().ok());
    let nthreads = args.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(48);

    let (data, npoints, ndims) = read_u8_dataset(&path, limit_points);

    // BigANN 10M config used for the before/after comparison (algo knobs only;
    // metric/threads/max_degree live in PiPNNBuildContext).
    let config = PiPNNConfig {
        num_hash_planes: 14,
        c_max: 512,
        c_min: 128,
        p_samp: 0.1,
        fanout: vec![8, 3],
        k: 3,
        replicas: 1,
        l_max: 128,
        final_prune: false,
        alpha: 1.2,
        leader_cap: 1000,
        saturate_after_prune: true,
        ..PiPNNConfig::default()
    };
    println!("config: {:?}  threads={}", config, nthreads);
    let ctx = PiPNNBuildContext::new(config, NonZeroUsize::new(64).unwrap(), Metric::L2, nthreads)
        .expect("ctx");
    let rss_pre = read_rss_kb();
    println!("RSS before build: {} MB", rss_pre / 1024);

    let t0 = Instant::now();
    let graph = build_typed(&data, npoints, ndims, &ctx).expect("build");
    let dt = t0.elapsed();

    let rss_post = read_rss_kb();
    println!(
        "BUILD_WALL={:.3}s  RSS_peak~{} MB  npoints={}  avg_degree={:.4}",
        dt.as_secs_f64(),
        rss_post / 1024,
        graph.npoints,
        graph.avg_degree()
    );
    println!("stats: {:?}", graph.build_stats);
}
