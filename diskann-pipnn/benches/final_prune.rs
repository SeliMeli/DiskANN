/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Criterion bench for the **final_prune** (RobustPrune) phase.
//!
//! Consumes a disk-cached pre-prune candidate graph (produced once by
//! partition + leaf_build + HashPrune::extract_graph_for_prune) and times the
//! RobustPrune pass itself. Iteration on `final_prune` does not rebuild the
//! graph — `common::candidates_for` hits the on-disk cache.
//!
//! Run with:
//! ```sh
//! cargo bench -p diskann-pipnn --bench final_prune
//! PIPNN_BENCH_SCALE=10m cargo bench -p diskann-pipnn --bench final_prune
//! ```

#[path = "common.rs"]
mod common;

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use diskann_pipnn::builder::final_prune_from_candidates;

fn bench_final_prune(c: &mut Criterion) {
    for load in common::Load::selected() {
        let Some(dataset) = common::dataset_for(load) else {
            continue;
        };
        let Some(candidates) = common::candidates_for(load) else {
            continue;
        };
        let cfg = load.pipnn_config();
        let ndims = load.ndims();
        let data = dataset.data();
        let pool = common::pool();

        let mut group = c.benchmark_group(format!("final_prune/{}", load.name()));
        if load.is_10m() {
            group
                .sample_size(10)
                .measurement_time(Duration::from_secs(90));
        } else {
            group
                .sample_size(20)
                .measurement_time(Duration::from_secs(30));
        }

        group.bench_function("default", |b| {
            b.iter(|| {
                pool.install(|| {
                    final_prune_from_candidates(
                        data,
                        ndims,
                        candidates.as_slice(),
                        cfg.max_degree,
                        cfg.metric,
                        cfg.alpha,
                        true,
                    )
                })
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_final_prune);
criterion_main!(benches);
