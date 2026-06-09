/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Criterion bench for the **partition** (RBC) phase.
//!
//! Runs `partition::partition` on each selected load with the canonical
//! per-load `PartitionConfig` from `common.rs`. 16 threads, seed=42.
//!
//! Run with:
//! ```sh
//! cargo bench -p diskann-pipnn --bench partition
//! PIPNN_BENCH_SCALE=10m cargo bench -p diskann-pipnn --bench partition
//! cargo bench -p diskann-pipnn --bench partition -- bigann_1m   # criterion name filter
//! ```

#[path = "common.rs"]
mod common;

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use diskann_pipnn::partition;
use half::f16;
use rayon::prelude::*;

fn bench_partition(c: &mut Criterion) {
    for load in common::Load::selected() {
        let Some(dataset) = common::dataset_for(load) else {
            continue;
        };
        let cfg = load.partition_config();
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

        let mut group = c.benchmark_group(format!("partition/{}", load.name()));
        if load.is_10m() {
            group
                .sample_size(10)
                .measurement_time(Duration::from_secs(90));
        } else {
            group
                .sample_size(20)
                .measurement_time(Duration::from_secs(30));
        }

        group.bench_function("f16", |b| {
            b.iter(|| {
                pool.install(|| partition::partition(data, ndims, npoints, &cfg, 42))
            });
        });

        group.bench_function("f32_preconv", |b| {
            b.iter(|| {
                pool.install(|| partition::partition(&f32_data, ndims, npoints, &cfg, 42))
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_partition);
criterion_main!(benches);
