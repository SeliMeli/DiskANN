/*
 * Bench: sequential vs parallel leaf GEMM (faer Par::Seq vs Par::Rayon).
 * Usage: cargo run --release -p diskann-pipnn --example leaf_gemm_bench -- <enron_fp16.bin>
 */
use std::time::Instant;
use diskann::utils::VectorRepr;
use half::f16;

fn load_fp16_bin(path: &str) -> (Vec<f16>, usize, usize) {
    let bytes = std::fs::read(path).expect("failed to read file");
    let npoints = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data: Vec<f16> = bytes[8..]
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect();
    (data, npoints, ndims)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: leaf_gemm_bench <enron_fp16.bin>");

    let (data, _npoints, ndims) = load_fp16_bin(path);
    println!("ndims={}\n", ndims);

    // Use 16 threads for parallel GEMM
    let _pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();

    println!("{:>6} {:>10} {:>10} {:>8}", "Size", "Seq(us)", "Par(us)", "Speedup");
    println!("{}", "-".repeat(40));

    for n in [64, 128, 256, 512, 1024] {
        let mut local_data = vec![0.0f32; n * ndims];
        for i in 0..n {
            let src = &data[i * ndims..(i + 1) * ndims];
            f16::as_f32_into(src, &mut local_data[i * ndims..(i + 1) * ndims]).unwrap();
        }

        let mut dot_seq = vec![0.0f32; n * n];
        let mut dot_par = vec![0.0f32; n * n];

        let iters = if n <= 256 { 100 } else if n <= 512 { 30 } else { 10 };

        // Warmup
        diskann_pipnn::gemm::sgemm_aat(&local_data, n, ndims, &mut dot_seq);
        diskann_pipnn::gemm::sgemm_aat_par(&local_data, n, ndims, &mut dot_par);

        // Bench sequential
        let t = Instant::now();
        for _ in 0..iters {
            diskann_pipnn::gemm::sgemm_aat(&local_data, n, ndims, &mut dot_seq);
        }
        let seq_us = t.elapsed().as_micros() as f64 / iters as f64;

        // Bench parallel
        let t = Instant::now();
        for _ in 0..iters {
            diskann_pipnn::gemm::sgemm_aat_par(&local_data, n, ndims, &mut dot_par);
        }
        let par_us = t.elapsed().as_micros() as f64 / iters as f64;

        println!("{:>6} {:>10.0} {:>10.0} {:>8.2}x", n, seq_us, par_us, seq_us / par_us);

        // Verify correctness
        let max_diff: f32 = dot_seq.iter().zip(dot_par.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "n={} max_diff={}", n, max_diff);
    }
}
