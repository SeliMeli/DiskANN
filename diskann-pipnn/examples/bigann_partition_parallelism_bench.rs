/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Microbench: Rayon topology for BigANN partition assignment.
//!
//! This isolates scheduling choices for the GEMM-tile + fused-topk assignment
//! path used by partitioning. It simulates multiple independent partition work
//! items and compares:
//!
//! - all partitions parallel, inner tiles parallel
//! - all partitions parallel, inner tiles sequential
//! - limited partition wave, inner tiles parallel
//! - partitions sequential, inner tiles parallel
//! - level-wide flat tile queue, each tile uses sequential GEMM
//!
//! Example:
//!
//! ```sh
//! cargo run --release -p diskann-pipnn --example bigann_partition_parallelism_bench -- \
//!   --partitions 16 --partition-points 50000 --leaders 500 --threads 16 --iters 3
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
    partitions: usize,
    partition_points: usize,
    leaders: usize,
    fanout: usize,
    outer_width: usize,
    threads: usize,
    iters: usize,
    mb: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            path: DEFAULT_PATH.to_string(),
            partitions: 16,
            partition_points: 50_000,
            leaders: 500,
            fanout: 10,
            outer_width: 4,
            threads: 16,
            iters: 3,
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
            "--partitions" => args.partitions = value.parse().expect("--partitions"),
            "--partition-points" => args.partition_points = value.parse().expect("--partition-points"),
            "--leaders" => args.leaders = value.parse().expect("--leaders"),
            "--fanout" => args.fanout = value.parse().expect("--fanout"),
            "--outer-width" => args.outer_width = value.parse().expect("--outer-width"),
            "--threads" => args.threads = value.parse().expect("--threads"),
            "--iters" => args.iters = value.parse().expect("--iters"),
            "--mb" => args.mb = value.parse().expect("--mb"),
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
        assert_eq!(mmap.len(), 8 + npoints * ndims, "bad BigANN file length");
        Self { _file: file, mmap, npoints, ndims }
    }

    fn data(&self) -> &[u8] {
        &self.mmap[8..]
    }
}

#[derive(Clone)]
struct WorkItem {
    points: Vec<u32>,
    leaders: Vec<u32>,
}

struct PreparedPartition<'a> {
    points: &'a [u32],
    l_data: Vec<f32>,
    l_norms: Vec<f32>,
    nl: usize,
}

#[derive(Clone, Copy)]
struct TileJob {
    partition: usize,
    row_start: usize,
    rows: usize,
}

fn make_work(args: &Args, npoints: usize) -> Vec<WorkItem> {
    assert!(args.partition_points <= npoints);
    assert!(args.leaders <= npoints);
    let point_stride = (npoints / (args.partition_points * args.partitions).max(1)).max(1);
    let leader_stride = (npoints / (args.leaders * args.partitions).max(1)).max(1);
    (0..args.partitions)
        .map(|p| {
            let point_base = p * 1_000_003;
            let leader_base = npoints / 3 + p * 917_611;
            let points = (0..args.partition_points)
                .map(|i| ((point_base + i * point_stride) % npoints) as u32)
                .collect();
            let leaders = (0..args.leaders)
                .map(|i| ((leader_base + i * leader_stride) % npoints) as u32)
                .collect();
            WorkItem { points, leaders }
        })
        .collect()
}

#[inline]
fn insert_top(top: &mut [(u32, f32)], id: u32, dist: f32) {
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

fn assign_gemm_tile(
    data: &[u8],
    ndims: usize,
    work: &WorkItem,
    fanout: usize,
    mb: usize,
    inner_parallel: bool,
    assignments: &mut [u32],
) {
    let np = work.points.len();
    let nl = work.leaders.len();
    let k = fanout.min(nl);
    assert!(k <= 16, "fanout > 16 not supported");

    let mut l_data = vec![0.0f32; nl * ndims];
    let mut l_norms = vec![0.0f32; nl];
    for (li, &leader) in work.leaders.iter().enumerate() {
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

    let chunk_body = |chunk_idx: usize, out_chunk: &mut [u32]| {
        let row_start = chunk_idx * mb;
        let rows = (row_start + mb).min(np) - row_start;
        let mut p_data = vec![0.0f32; rows * ndims];
        let mut p_norms = vec![0.0f32; rows];
        let mut dots = vec![0.0f32; rows * nl];

        for r in 0..rows {
            let pidx = work.points[row_start + r] as usize;
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
                insert_top(&mut top[..k], j as u32, dist);
            }
            let out = &mut out_chunk[r * k..(r + 1) * k];
            for i in 0..k {
                out[i] = top[i].0;
            }
        }
    };

    if inner_parallel {
        assignments
            .par_chunks_mut(mb * k)
            .enumerate()
            .for_each(|(idx, chunk)| chunk_body(idx, chunk));
    } else {
        for (idx, chunk) in assignments.chunks_mut(mb * k).enumerate() {
            chunk_body(idx, chunk);
        }
    }
}

fn prepare_partitions<'a>(data: &[u8], ndims: usize, work: &'a [WorkItem]) -> Vec<PreparedPartition<'a>> {
    work.iter()
        .map(|w| {
            let nl = w.leaders.len();
            let mut l_data = vec![0.0f32; nl * ndims];
            let mut l_norms = vec![0.0f32; nl];
            for (li, &leader) in w.leaders.iter().enumerate() {
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
            PreparedPartition { points: &w.points, l_data, l_norms, nl }
        })
        .collect()
}

fn make_tile_jobs(prepared: &[PreparedPartition<'_>], mb: usize) -> Vec<TileJob> {
    let mut jobs = Vec::new();
    for (partition, p) in prepared.iter().enumerate() {
        let mut row_start = 0usize;
        while row_start < p.points.len() {
            let rows = (p.points.len() - row_start).min(mb);
            jobs.push(TileJob { partition, row_start, rows });
            row_start += rows;
        }
    }
    jobs
}

fn run_flat_tile_queue(
    data: &[u8],
    ndims: usize,
    work: &[WorkItem],
    args: &Args,
    buffers: &mut [Vec<u32>],
) {
    let prepared = prepare_partitions(data, ndims, work);
    let jobs = make_tile_jobs(&prepared, args.mb);
    let k = args.fanout.min(args.leaders);

    // Avoid aliasing `buffers` mutably across parallel jobs by writing each
    // partition's assignments into a raw address. Tile jobs are disjoint row
    // ranges by construction.
    let ptrs: Vec<usize> = buffers.iter_mut().map(|b| b.as_mut_ptr() as usize).collect();

    jobs.par_iter().for_each(|job| {
        let part = &prepared[job.partition];
        let nl = part.nl;
        let k = k.min(nl);
        let rows = job.rows;
        let mut p_data = vec![0.0f32; rows * ndims];
        let mut p_norms = vec![0.0f32; rows];
        let mut dots = vec![0.0f32; rows * nl];

        for r in 0..rows {
            let pidx = part.points[job.row_start + r] as usize;
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

        sgemm_abt(&p_data, rows, ndims, &part.l_data, nl, &mut dots);

        let base = job.row_start * k;
        let out_ptr = ptrs[job.partition] as *mut u32;
        for r in 0..rows {
            let dot_row = &dots[r * nl..(r + 1) * nl];
            let mut top = [(u32::MAX, f32::MAX); 16];
            for j in 0..nl {
                let dist = p_norms[r] + part.l_norms[j] - 2.0 * dot_row[j];
                insert_top(&mut top[..k], j as u32, dist);
            }
            unsafe {
                let out = std::slice::from_raw_parts_mut(out_ptr.add(base + r * k), k);
                for i in 0..k {
                    out[i] = top[i].0;
                }
            }
        }
    });
}

fn checksum(buffers: &[Vec<u32>]) -> u64 {
    buffers.iter().flat_map(|v| v.iter()).fold(0u64, |acc, &x| {
        acc.wrapping_mul(1_000_003).wrapping_add(x as u64)
    })
}

fn run_all_outer_parallel(
    data: &[u8],
    ndims: usize,
    work: &[WorkItem],
    args: &Args,
    inner_parallel: bool,
    buffers: &mut [Vec<u32>],
) {
    work.par_iter()
        .zip(buffers.par_iter_mut())
        .for_each(|(w, out)| {
            assign_gemm_tile(data, ndims, w, args.fanout, args.mb, inner_parallel, out);
        });
}

fn run_limited_outer_parallel(
    data: &[u8],
    ndims: usize,
    work: &[WorkItem],
    args: &Args,
    buffers: &mut [Vec<u32>],
) {
    let width = args.outer_width.max(1);
    for (work_chunk, out_chunk) in work.chunks(width).zip(buffers.chunks_mut(width)) {
        work_chunk
            .par_iter()
            .zip(out_chunk.par_iter_mut())
            .for_each(|(w, out)| {
                assign_gemm_tile(data, ndims, w, args.fanout, args.mb, true, out);
            });
    }
}

fn run_outer_seq_inner_parallel(
    data: &[u8],
    ndims: usize,
    work: &[WorkItem],
    args: &Args,
    buffers: &mut [Vec<u32>],
) {
    for (w, out) in work.iter().zip(buffers.iter_mut()) {
        assign_gemm_tile(data, ndims, w, args.fanout, args.mb, true, out);
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

fn print_result(label: &str, result: (Duration, Duration, Duration)) {
    println!(
        "{label:<38} min/med/mean = {:>8.3}/{:>8.3}/{:>8.3} ms",
        result.0.as_secs_f64() * 1e3,
        result.1.as_secs_f64() * 1e3,
        result.2.as_secs_f64() * 1e3,
    );
}

fn main() {
    let args = parse_args();
    assert!(args.partitions > 0);
    assert!(args.partition_points > 0);
    assert!(args.leaders > 0);
    assert!(args.fanout > 0 && args.fanout <= 16);
    assert!(args.iters > 0);
    assert!(args.mb > 0);

    let bigann = BigAnnMmap::open(&args.path);
    let work = make_work(&args, bigann.npoints);
    let k = args.fanout.min(args.leaders);
    let mut buffers = vec![vec![0u32; args.partition_points * k]; args.partitions];

    println!("BigANN partition parallelism bench");
    println!("  path:             {}", args.path);
    println!("  data:             {} x {} u8", bigann.npoints, bigann.ndims);
    println!("  partitions:       {}", args.partitions);
    println!("  partition_points: {}", args.partition_points);
    println!("  leaders:          {}", args.leaders);
    println!("  fanout:           {}", k);
    println!("  outer_width:      {}", args.outer_width);
    println!("  threads:          {}", args.threads);
    println!("  iters:            {}", args.iters);
    println!("  mb:               {}", args.mb);
    println!();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("build rayon pool");

    pool.install(|| {
        run_all_outer_parallel(bigann.data(), bigann.ndims, &work, &args, false, &mut buffers);
        black_box(checksum(&buffers));
    });

    let outer_par_inner_par = pool.install(|| {
        measure(args.iters, || {
            run_all_outer_parallel(bigann.data(), bigann.ndims, &work, &args, true, &mut buffers);
            black_box(checksum(&buffers));
        })
    });
    let outer_par_inner_seq = pool.install(|| {
        measure(args.iters, || {
            run_all_outer_parallel(bigann.data(), bigann.ndims, &work, &args, false, &mut buffers);
            black_box(checksum(&buffers));
        })
    });
    let limited_outer_inner_par = pool.install(|| {
        measure(args.iters, || {
            run_limited_outer_parallel(bigann.data(), bigann.ndims, &work, &args, &mut buffers);
            black_box(checksum(&buffers));
        })
    });
    let outer_seq_inner_par = pool.install(|| {
        measure(args.iters, || {
            run_outer_seq_inner_parallel(bigann.data(), bigann.ndims, &work, &args, &mut buffers);
            black_box(checksum(&buffers));
        })
    });
    let flat_tile_queue = pool.install(|| {
        measure(args.iters, || {
            run_flat_tile_queue(bigann.data(), bigann.ndims, &work, &args, &mut buffers);
            black_box(checksum(&buffers));
        })
    });

    print_result("all partitions par + inner tiles par", outer_par_inner_par);
    print_result("all partitions par + inner tiles seq", outer_par_inner_seq);
    print_result("limited partitions par + inner par", limited_outer_inner_par);
    print_result("partitions seq + inner tiles par", outer_seq_inner_par);
    print_result("flat level tile queue", flat_tile_queue);
}
