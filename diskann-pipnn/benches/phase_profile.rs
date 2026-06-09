/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sub-step profiler for the 3 PiPNN phases.
//!
//! Requires the `bench-profiling` Cargo feature. Runs partition → leaf_build →
//! final_prune on a single chosen load and prints a per-phase stdout table of
//! sub-step timings accumulated via `crate::profile::PhaseTimer`.
//!
//! Unlike the criterion benches (which measure wall-clock of each phase as a
//! black box), this profiler answers "where is the time going inside a phase."
//!
//! Run:
//! ```sh
//! cargo bench -p diskann-pipnn --bench phase_profile \
//!     --features bench-profiling -- --load bigann_1m
//! cargo bench -p diskann-pipnn --bench phase_profile \
//!     --features bench-profiling -- --load enron_1m
//! ```

#[path = "common.rs"]
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use diskann_pipnn::builder::final_prune_from_candidates;
use diskann_pipnn::{leaf_build, partition, profile};
use rayon::prelude::*;

fn parse_args() -> common::Load {
    let args: Vec<String> = std::env::args().collect();
    let mut load = common::Load::BigAnn1M;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--load" => {
                i += 1;
                if let Some(l) = args.get(i).and_then(|s| common::Load::from_name(s)) {
                    load = l;
                } else {
                    eprintln!(
                        "Unknown --load value. Choices: bigann_1m, bigann_10m, enron_1m, enron_10m"
                    );
                    std::process::exit(2);
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: phase_profile [--load <bigann_1m|bigann_10m|enron_1m|enron_10m>]"
                );
                std::process::exit(0);
            }
            // criterion may pass its own args; ignore unknowns so
            // `cargo bench` without explicit args still works
            _ => {}
        }
        i += 1;
    }
    load
}

fn print_snapshot(label: &str, wall: Duration, snapshot: &[(&'static str, Duration)]) {
    println!("=== {} ({:.3}s wall) ===", label, wall.as_secs_f64());
    if snapshot.is_empty() {
        println!("  (no sub-step timings — build with --features bench-profiling)");
        return;
    }
    let total_ns: u128 = snapshot.iter().map(|(_, d)| d.as_nanos()).sum();
    println!(
        "  {:<32} {:>10} {:>8}",
        "sub-step", "time(s)", "%"
    );
    for (name, d) in snapshot {
        let secs = d.as_secs_f64();
        let pct = if total_ns > 0 {
            100.0 * d.as_nanos() as f64 / total_ns as f64
        } else {
            0.0
        };
        println!("  {:<32} {:>10.3} {:>7.1}%", name, secs, pct);
    }
    let sum = Duration::from_nanos(total_ns as u64);
    println!(
        "  {:<32} {:>10.3} {:>7.1}%",
        "TOTAL (summed sub-steps)",
        sum.as_secs_f64(),
        100.0
    );
    println!();
}

fn main() {
    let load = parse_args();
    let Some(dataset) = common::dataset_for(load) else {
        std::process::exit(1);
    };
    let cfg = load.pipnn_config();
    let ndims = load.ndims();
    let npoints = load.npoints();
    let data = dataset.data();
    let pool = common::pool();

    println!(
        "Dataset: {} ({} x {}, {:?})",
        load.name(),
        npoints,
        ndims,
        load.metric()
    );
    println!(
        "Config:  c_max={} c_min={} fanout={:?} leaf_k={} l_max={} hp={} p_samp={}",
        cfg.c_max, cfg.c_min, cfg.fanout, cfg.k, cfg.l_max, cfg.num_hash_planes, cfg.p_samp
    );
    println!();

    // ---- Phase 1: Partition ----
    profile::reset();
    let t0 = Instant::now();
    let leaves = pool.install(|| {
        partition::partition(data, ndims, npoints, &load.partition_config(), 42)
    });
    let wall = t0.elapsed();
    let snap = profile::take_snapshot();
    print_snapshot(
        &format!("Partition (RBC) — {} leaves", leaves.len()),
        wall,
        &snap,
    );

    // ---- Phase 2: Leaf build ----
    profile::reset();
    let edges = AtomicUsize::new(0);
    let t0 = Instant::now();
    // Allow: runs inside `pool.install(|| ...)` so rayon work stays on the bench pool.
    #[allow(clippy::disallowed_methods)]
    pool.install(|| {
        leaves.par_iter().for_each(|leaf| {
            let indices_usize: Vec<usize> = leaf.indices.iter().map(|&i| i as usize).collect();
            let e = leaf_build::build_leaf(data, ndims, &indices_usize, cfg.k, cfg.metric);
            edges.fetch_add(e.len(), Ordering::Relaxed);
        });
    });
    let wall = t0.elapsed();
    let snap = profile::take_snapshot();
    print_snapshot(
        &format!(
            "Leaf build — {} edges across {} leaves",
            edges.load(Ordering::Relaxed),
            leaves.len()
        ),
        wall,
        &snap,
    );

    // ---- Phase 3: Final prune ----
    // Use the disk-cached candidate fixture (builds once per config, seconds on
    // re-runs). This avoids re-running hash_prune inside the profiler.
    //
    // Reset AFTER candidates_for(): on a cache miss it rebuilds partition +
    // leaf_build internally, and those sub-step timers would otherwise pollute
    // the final_prune snapshot.
    let Some(candidates) = common::candidates_for(load) else {
        eprintln!(
            "[{}] could not build/load candidate fixture; skipping final_prune phase",
            load.name()
        );
        return;
    };
    profile::reset();
    let t0 = Instant::now();
    let pruned = pool.install(|| {
        final_prune_from_candidates(
            data,
            ndims,
            candidates.as_slice(),
            cfg.max_degree,
            cfg.metric,
            cfg.alpha,
            true,
        )
    });
    let wall = t0.elapsed();
    let snap = profile::take_snapshot();

    let avg_in: f64 = candidates.iter().map(|v| v.len()).sum::<usize>() as f64 / candidates.len() as f64;
    let avg_out: f64 = pruned.iter().map(|v| v.len()).sum::<usize>() as f64 / pruned.len().max(1) as f64;
    let label = format!(
        "Final prune (RobustPrune) — avg degree {:.1} → {:.1}",
        avg_in, avg_out
    );
    print_snapshot(&label, wall, &snap);
}
