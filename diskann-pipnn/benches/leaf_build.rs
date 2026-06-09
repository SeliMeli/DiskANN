/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Criterion bench for the **leaf_build** (GEMM-based local k-NN) phase.
//!
//! Iterates `leaf_build::build_leaf` across the cached partition output for
//! each load. The partition fixture is built once per `cargo bench` process
//! (via `common::leaves_for`), then reused across iterations.
//!
//! Run with:
//! ```sh
//! cargo bench -p diskann-pipnn --bench leaf_build
//! PIPNN_BENCH_SCALE=10m cargo bench -p diskann-pipnn --bench leaf_build
//! ```

#[path = "common.rs"]
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use diskann_pipnn::leaf_build;
use half::f16;
use rayon::prelude::*;

fn bench_leaf_build(c: &mut Criterion) {
    for load in common::Load::selected() {
        let Some(dataset) = common::dataset_for(load) else {
            continue;
        };
        let Some(leaves) = common::leaves_for(load) else {
            continue;
        };
        let cfg = load.pipnn_config();
        let ndims = load.ndims();
        let npoints = load.npoints();
        let data: &[f16] = dataset.data();
        let pool = common::pool();

        // Pre-convert f16→f32 once for the f32 variant.
        let f32_data: Vec<f32> = pool.install(|| {
            let mut buf = vec![0.0f32; npoints * ndims];
            buf.par_chunks_mut(ndims)
                .enumerate()
                .for_each(|(i, dst)| {
                    let src = &data[i * ndims..(i + 1) * ndims];
                    for (d, s) in dst.iter_mut().zip(src.iter()) {
                        *d = s.to_f32();
                    }
                });
            buf
        });

        let mut group = c.benchmark_group(format!("leaf_build/{}", load.name()));
        if load.is_10m() {
            group
                .sample_size(10)
                .measurement_time(Duration::from_secs(90));
        } else {
            group
                .sample_size(20)
                .measurement_time(Duration::from_secs(30));
        }

        // Allow: inside `pool.install(|| ...)` rayon work runs on the bench pool.
        #[allow(clippy::disallowed_methods)]
        group.bench_function("f16", |b| {
            b.iter(|| {
                let edges = AtomicUsize::new(0);
                pool.install(|| {
                    leaves.par_iter().for_each(|leaf| {
                        let indices_usize: Vec<usize> =
                            leaf.indices.iter().map(|&i| i as usize).collect();
                        let e = leaf_build::build_leaf(data, ndims, &indices_usize, cfg.k, cfg.metric);
                        edges.fetch_add(e.len(), Ordering::Relaxed);
                    });
                });
                criterion::black_box(edges.load(Ordering::Relaxed))
            });
        });

        #[allow(clippy::disallowed_methods)]
        group.bench_function("f32_preconv", |b| {
            b.iter(|| {
                let edges = AtomicUsize::new(0);
                pool.install(|| {
                    leaves.par_iter().for_each(|leaf| {
                        let indices_usize: Vec<usize> =
                            leaf.indices.iter().map(|&i| i as usize).collect();
                        let e = leaf_build::build_leaf(&f32_data, ndims, &indices_usize, cfg.k, cfg.metric);
                        edges.fetch_add(e.len(), Ordering::Relaxed);
                    });
                });
                criterion::black_box(edges.load(Ordering::Relaxed))
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_leaf_build);
criterion_main!(benches);
