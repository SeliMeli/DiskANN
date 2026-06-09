/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Microbench: BigANN u8 partition assignment kernels.
//!
//! Compares the current partition-style tile pipeline:
//!
//! ```text
//! gather u8 -> f32 tile -> f32 GEMM against leaders -> fused row top-k
//! ```
//!
//! against direct point/leader kernels:
//!
//! ```text
//! u8 SIMD L2(point, leader) -> immediate top-k, no dot matrix write
//! u8 SIMD L2(point, 4 leaders) -> immediate top-k, reuses point loads
//! ```
//!
//! Default input is `datasets/bigann/bigann_10m.u8bin`.
//!
//! Example:
//!
//! ```sh
//! cargo run --release -p diskann-pipnn --example bigann_assign_microbench
//! cargo run --release -p diskann-pipnn --example bigann_assign_microbench -- \
//!   --points 100000 --leaders 1000 --fanout 10 --threads 16 --iters 5
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use diskann_pipnn::gemm::sgemm_abt;
use memmap2::Mmap;
use rayon::prelude::*;

const DEFAULT_PATH: &str = "datasets/bigann/bigann_10m.u8bin";

#[derive(Debug, Clone)]
struct Args {
    path: String,
    points: usize,
    leaders: usize,
    fanout: usize,
    threads: usize,
    iters: usize,
    mb: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            path: DEFAULT_PATH.to_string(),
            points: 100_000,
            leaders: 1_000,
            fanout: 10,
            threads: 16,
            iters: 5,
            mb: 128,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else {
            eprintln!("missing value for {flag}");
            std::process::exit(2);
        };
        match flag.as_str() {
            "--path" => args.path = value,
            "--points" => args.points = value.parse().expect("--points must be usize"),
            "--leaders" => args.leaders = value.parse().expect("--leaders must be usize"),
            "--fanout" => args.fanout = value.parse().expect("--fanout must be usize"),
            "--threads" => args.threads = value.parse().expect("--threads must be usize"),
            "--iters" => args.iters = value.parse().expect("--iters must be usize"),
            "--mb" => args.mb = value.parse().expect("--mb must be usize"),
            _ => {
                eprintln!("unknown flag: {flag}");
                std::process::exit(2);
            }
        }
    }
    args
}

struct BigAnnMmap {
    _file: std::fs::File,
    mmap: Mmap,
    npoints: usize,
    ndims: usize,
}

impl BigAnnMmap {
    fn open(path: &str) -> Self {
        let file = std::fs::File::open(path).expect("open BigANN file");
        let mmap = unsafe { Mmap::map(&file).expect("mmap BigANN file") };
        assert!(mmap.len() >= 8, "file too small");
        let npoints = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
        let ndims = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let expected = 8 + npoints * ndims;
        assert_eq!(
            mmap.len(),
            expected,
            "unexpected BigANN file length: expected {expected}, got {}",
            mmap.len()
        );
        Self {
            _file: file,
            mmap,
            npoints,
            ndims,
        }
    }

    fn data(&self) -> &[u8] {
        &self.mmap[8..]
    }
}

#[inline]
fn insert_top_u32(top: &mut [(u32, u32)], id: u32, dist: u32) {
    let last = top.len() - 1;
    if dist < top[last].1 {
        top[last] = (id, dist);
        let mut i = last;
        while i > 0 && top[i].1 < top[i - 1].1 {
            top.swap(i, i - 1);
            i -= 1;
        }
    }
}

#[inline]
fn insert_top_f32(top: &mut [(u32, f32)], id: u32, dist: f32) {
    let last = top.len() - 1;
    if dist < top[last].1 {
        top[last] = (id, dist);
        let mut i = last;
        while i > 0 && top[i].1 < top[i - 1].1 {
            top.swap(i, i - 1);
            i -= 1;
        }
    }
}

#[inline]
fn l2_u8_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as i32 - y as i32;
            (d * d) as u32
        })
        .sum()
}

#[inline]
fn l2_u8(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { l2_u8_avx2(a, b) };
        }
    }
    l2_u8_scalar(a, b)
}

#[inline]
fn l2_u8x4(a: &[u8], b0: &[u8], b1: &[u8], b2: &[u8], b3: &[u8]) -> [u32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { l2_u8x4_avx2(a, b0, b1, b2, b3) };
        }
    }
    [
        l2_u8_scalar(a, b0),
        l2_u8_scalar(a, b1),
        l2_u8_scalar(a, b2),
        l2_u8_scalar(a, b3),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_u8_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_si256();
    let zero = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= a.len() {
        let av = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let bv = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        let a_lo = _mm256_unpacklo_epi8(av, zero);
        let a_hi = _mm256_unpackhi_epi8(av, zero);
        let b_lo = _mm256_unpacklo_epi8(bv, zero);
        let b_hi = _mm256_unpackhi_epi8(bv, zero);

        let d_lo = _mm256_sub_epi16(a_lo, b_lo);
        let d_hi = _mm256_sub_epi16(a_hi, b_hi);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_lo, d_lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_hi, d_hi));
        i += 32;
    }

    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut sum = lanes.iter().map(|&v| v as u32).sum::<u32>();
    while i < a.len() {
        let d = a[i] as i32 - b[i] as i32;
        sum += (d * d) as u32;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_u8x4_avx2(a: &[u8], b0: &[u8], b1: &[u8], b2: &[u8], b3: &[u8]) -> [u32; 4] {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();
    let mut i = 0usize;

    while i + 32 <= a.len() {
        let av = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let a_lo = _mm256_unpacklo_epi8(av, zero);
        let a_hi = _mm256_unpackhi_epi8(av, zero);

        let bv0 = _mm256_loadu_si256(b0.as_ptr().add(i) as *const __m256i);
        let bv1 = _mm256_loadu_si256(b1.as_ptr().add(i) as *const __m256i);
        let bv2 = _mm256_loadu_si256(b2.as_ptr().add(i) as *const __m256i);
        let bv3 = _mm256_loadu_si256(b3.as_ptr().add(i) as *const __m256i);

        let b0_lo = _mm256_unpacklo_epi8(bv0, zero);
        let b0_hi = _mm256_unpackhi_epi8(bv0, zero);
        let b1_lo = _mm256_unpacklo_epi8(bv1, zero);
        let b1_hi = _mm256_unpackhi_epi8(bv1, zero);
        let b2_lo = _mm256_unpacklo_epi8(bv2, zero);
        let b2_hi = _mm256_unpackhi_epi8(bv2, zero);
        let b3_lo = _mm256_unpacklo_epi8(bv3, zero);
        let b3_hi = _mm256_unpackhi_epi8(bv3, zero);

        let d0_lo = _mm256_sub_epi16(a_lo, b0_lo);
        let d0_hi = _mm256_sub_epi16(a_hi, b0_hi);
        let d1_lo = _mm256_sub_epi16(a_lo, b1_lo);
        let d1_hi = _mm256_sub_epi16(a_hi, b1_hi);
        let d2_lo = _mm256_sub_epi16(a_lo, b2_lo);
        let d2_hi = _mm256_sub_epi16(a_hi, b2_hi);
        let d3_lo = _mm256_sub_epi16(a_lo, b3_lo);
        let d3_hi = _mm256_sub_epi16(a_hi, b3_hi);

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0_lo, d0_lo));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0_hi, d0_hi));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1_lo, d1_lo));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1_hi, d1_hi));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d2_lo, d2_lo));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d2_hi, d2_hi));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(d3_lo, d3_lo));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(d3_hi, d3_hi));

        i += 32;
    }

    #[inline]
    unsafe fn reduce(v: std::arch::x86_64::__m256i) -> u32 {
        use std::arch::x86_64::*;
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v);
        lanes.iter().map(|&x| x as u32).sum::<u32>()
    }

    let mut out = [reduce(acc0), reduce(acc1), reduce(acc2), reduce(acc3)];
    while i < a.len() {
        let ax = a[i] as i32;
        let d0 = ax - b0[i] as i32;
        let d1 = ax - b1[i] as i32;
        let d2 = ax - b2[i] as i32;
        let d3 = ax - b3[i] as i32;
        out[0] += (d0 * d0) as u32;
        out[1] += (d1 * d1) as u32;
        out[2] += (d2 * d2) as u32;
        out[3] += (d3 * d3) as u32;
        i += 1;
    }
    out
}

fn make_ids(npoints: usize, count: usize, offset: usize) -> Vec<u32> {
    assert!(count <= npoints, "count must fit in dataset");
    let stride = (npoints / count.max(1)).max(1);
    (0..count)
        .map(|i| ((offset + i * stride) % npoints) as u32)
        .collect()
}

fn assign_direct_simd(
    data: &[u8],
    ndims: usize,
    points: &[u32],
    leaders: &[u32],
    fanout: usize,
    assignments: &mut [u32],
) {
    let k = fanout.min(leaders.len());
    assert!(k <= 16, "fanout > 16 not supported by this microbench");
    assignments
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(pi, out)| {
            let pidx = points[pi] as usize;
            let p = &data[pidx * ndims..(pidx + 1) * ndims];
            let mut top = [(u32::MAX, u32::MAX); 16];
            for (li, &leader) in leaders.iter().enumerate() {
                let lidx = leader as usize;
                let l = &data[lidx * ndims..(lidx + 1) * ndims];
                let dist = l2_u8(p, l);
                insert_top_u32(&mut top[..k], li as u32, dist);
            }
            for i in 0..k {
                out[i] = top[i].0;
            }
        });
}

fn assign_direct_simd_x4(
    data: &[u8],
    ndims: usize,
    points: &[u32],
    leaders: &[u32],
    fanout: usize,
    assignments: &mut [u32],
) {
    let k = fanout.min(leaders.len());
    assert!(k <= 16, "fanout > 16 not supported by this microbench");
    assignments
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(pi, out)| {
            let pidx = points[pi] as usize;
            let p = &data[pidx * ndims..(pidx + 1) * ndims];
            let mut top = [(u32::MAX, u32::MAX); 16];

            let mut li = 0usize;
            while li + 4 <= leaders.len() {
                let l0 = leaders[li] as usize;
                let l1 = leaders[li + 1] as usize;
                let l2 = leaders[li + 2] as usize;
                let l3 = leaders[li + 3] as usize;
                let d = l2_u8x4(
                    p,
                    &data[l0 * ndims..(l0 + 1) * ndims],
                    &data[l1 * ndims..(l1 + 1) * ndims],
                    &data[l2 * ndims..(l2 + 1) * ndims],
                    &data[l3 * ndims..(l3 + 1) * ndims],
                );
                insert_top_u32(&mut top[..k], li as u32, d[0]);
                insert_top_u32(&mut top[..k], (li + 1) as u32, d[1]);
                insert_top_u32(&mut top[..k], (li + 2) as u32, d[2]);
                insert_top_u32(&mut top[..k], (li + 3) as u32, d[3]);
                li += 4;
            }

            while li < leaders.len() {
                let lidx = leaders[li] as usize;
                let l = &data[lidx * ndims..(lidx + 1) * ndims];
                insert_top_u32(&mut top[..k], li as u32, l2_u8(p, l));
                li += 1;
            }

            for i in 0..k {
                out[i] = top[i].0;
            }
        });
}

fn assign_gemm_tile(
    data: &[u8],
    ndims: usize,
    points: &[u32],
    leaders: &[u32],
    fanout: usize,
    mb: usize,
    assignments: &mut [u32],
) {
    let np = points.len();
    let nl = leaders.len();
    let k = fanout.min(nl);
    assert!(k <= 16, "fanout > 16 not supported by this microbench");

    let mut l_data = vec![0.0f32; nl * ndims];
    let mut l_norms = vec![0.0f32; nl];
    for (li, &leader) in leaders.iter().enumerate() {
        let src = &data[leader as usize * ndims..(leader as usize + 1) * ndims];
        let dst = &mut l_data[li * ndims..(li + 1) * ndims];
        let mut norm = 0.0f32;
        for (d, &x) in dst.iter_mut().zip(src.iter()) {
            let xf = x as f32;
            *d = xf;
            norm += xf * xf;
        }
        l_norms[li] = norm;
    }

    assignments
        .par_chunks_mut(mb * k)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * mb;
            let rows = (row_start + mb).min(np) - row_start;
            let mut p_data = vec![0.0f32; rows * ndims];
            let mut p_norms = vec![0.0f32; rows];
            let mut dots = vec![0.0f32; rows * nl];

            for r in 0..rows {
                let pidx = points[row_start + r] as usize;
                let src = &data[pidx * ndims..(pidx + 1) * ndims];
                let dst = &mut p_data[r * ndims..(r + 1) * ndims];
                let mut norm = 0.0f32;
                for (d, &x) in dst.iter_mut().zip(src.iter()) {
                    let xf = x as f32;
                    *d = xf;
                    norm += xf * xf;
                }
                p_norms[r] = norm;
            }

            sgemm_abt(&p_data, rows, ndims, &l_data, nl, &mut dots);

            for r in 0..rows {
                let dot_row = &dots[r * nl..(r + 1) * nl];
                let mut top = [(u32::MAX, f32::MAX); 16];
                for j in 0..nl {
                    let dist = p_norms[r] + l_norms[j] - 2.0 * dot_row[j];
                    insert_top_f32(&mut top[..k], j as u32, dist);
                }
                let out = &mut out_chunk[r * k..(r + 1) * k];
                for i in 0..k {
                    out[i] = top[i].0;
                }
            }
        });
}

fn scalar_expected(
    data: &[u8],
    ndims: usize,
    point: u32,
    leaders: &[u32],
    fanout: usize,
) -> [u32; 16] {
    let k = fanout.min(leaders.len());
    let pidx = point as usize;
    let p = &data[pidx * ndims..(pidx + 1) * ndims];
    let mut top = [(u32::MAX, u32::MAX); 16];
    for (li, &leader) in leaders.iter().enumerate() {
        let lidx = leader as usize;
        let l = &data[lidx * ndims..(lidx + 1) * ndims];
        insert_top_u32(&mut top[..k], li as u32, l2_u8_scalar(p, l));
    }
    let mut out = [u32::MAX; 16];
    for i in 0..k {
        out[i] = top[i].0;
    }
    out
}

fn verify(
    data: &[u8],
    ndims: usize,
    points: &[u32],
    leaders: &[u32],
    fanout: usize,
    direct: &[u32],
    gemm: &[u32],
) {
    let k = fanout.min(leaders.len());
    let rows = points.len().min(128);
    for r in 0..rows {
        let expected = scalar_expected(data, ndims, points[r], leaders, fanout);
        let direct_row = &direct[r * k..(r + 1) * k];
        let gemm_row = &gemm[r * k..(r + 1) * k];
        assert_eq!(direct_row, &expected[..k], "direct mismatch at row {r}");
        assert_eq!(gemm_row, &expected[..k], "gemm mismatch at row {r}");
    }
}

fn measure<F: FnMut()>(iters: usize, mut f: F) -> (Duration, Duration, Duration) {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    samples.sort_unstable();
    let total = samples.iter().copied().sum::<Duration>();
    (samples[0], samples[iters / 2], total / iters as u32)
}

fn checksum(xs: &[u32]) -> u64 {
    xs.iter()
        .fold(0u64, |acc, &x| acc.wrapping_mul(1_000_003).wrapping_add(x as u64))
}

fn main() {
    let args = parse_args();
    assert!(args.fanout > 0 && args.fanout <= 16);
    assert!(args.iters > 0);
    assert!(args.mb > 0);

    let bigann = BigAnnMmap::open(&args.path);
    let npoints = bigann.npoints;
    let ndims = bigann.ndims;
    assert!(args.points <= npoints);
    assert!(args.leaders <= npoints);

    let points = make_ids(npoints, args.points, 0);
    let leaders = make_ids(npoints, args.leaders, npoints / 3);
    let k = args.fanout.min(args.leaders);
    let mut direct = vec![0u32; args.points * k];
    let mut direct_x4 = vec![0u32; args.points * k];
    let mut gemm = vec![0u32; args.points * k];

    println!("BigANN assign microbench");
    println!("  path:    {}", args.path);
    println!("  data:    {} x {} u8", npoints, ndims);
    println!("  points:  {}", args.points);
    println!("  leaders: {}", args.leaders);
    println!("  fanout:  {}", k);
    println!("  threads: {}", args.threads);
    println!("  iters:   {}", args.iters);
    println!("  gemm mb: {}", args.mb);
    #[cfg(target_arch = "x86_64")]
    println!("  avx2:    {}", std::is_x86_feature_detected!("avx2"));
    println!();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("build rayon pool");

    pool.install(|| {
        assign_direct_simd(bigann.data(), ndims, &points, &leaders, k, &mut direct);
        assign_direct_simd_x4(
            bigann.data(),
            ndims,
            &points,
            &leaders,
            k,
            &mut direct_x4,
        );
        assign_gemm_tile(
            bigann.data(),
            ndims,
            &points,
            &leaders,
            k,
            args.mb,
            &mut gemm,
        );
    });
    verify(bigann.data(), ndims, &points, &leaders, k, &direct, &gemm);
    assert_eq!(
        &direct_x4[..points.len().min(128) * k],
        &direct[..points.len().min(128) * k],
        "direct x4 mismatch"
    );
    println!("correctness: first {} rows match scalar top-k", points.len().min(128));
    println!();

    let (direct_min, direct_med, direct_mean) = pool.install(|| {
        measure(args.iters, || {
            assign_direct_simd(bigann.data(), ndims, &points, &leaders, k, &mut direct);
            black_box(checksum(&direct));
        })
    });

    let (gemm_min, gemm_med, gemm_mean) = pool.install(|| {
        measure(args.iters, || {
            assign_gemm_tile(
                bigann.data(),
                ndims,
                &points,
                &leaders,
                k,
                args.mb,
                &mut gemm,
            );
            black_box(checksum(&gemm));
        })
    });

    let (x4_min, x4_med, x4_mean) = pool.install(|| {
        measure(args.iters, || {
            assign_direct_simd_x4(
                bigann.data(),
                ndims,
                &points,
                &leaders,
                k,
                &mut direct_x4,
            );
            black_box(checksum(&direct_x4));
        })
    });

    println!(
        "direct SIMD+topk: min/med/mean = {:>8.3}/{:>8.3}/{:>8.3} ms",
        direct_min.as_secs_f64() * 1e3,
        direct_med.as_secs_f64() * 1e3,
        direct_mean.as_secs_f64() * 1e3,
    );
    println!(
        "gemm tile+topk:   min/med/mean = {:>8.3}/{:>8.3}/{:>8.3} ms",
        gemm_min.as_secs_f64() * 1e3,
        gemm_med.as_secs_f64() * 1e3,
        gemm_mean.as_secs_f64() * 1e3,
    );
    println!(
        "direct x4+topk:   min/med/mean = {:>8.3}/{:>8.3}/{:>8.3} ms",
        x4_min.as_secs_f64() * 1e3,
        x4_med.as_secs_f64() * 1e3,
        x4_mean.as_secs_f64() * 1e3,
    );
    println!(
        "speedup by min:   direct/gemm = {:.3}x  gemm/direct = {:.3}x",
        gemm_min.as_secs_f64() / direct_min.as_secs_f64(),
        direct_min.as_secs_f64() / gemm_min.as_secs_f64(),
    );
}
