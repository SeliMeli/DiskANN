/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::Serialize;
use std::fmt;

use diskann::{
    graph::config,
    utils::{VectorRepr, ONE},
};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_disk::{
    build::builder::build::DiskIndexBuilder,
    disk_index_build_parameter::{
        DiskIndexBuildParameters, MemoryBudget, NumPQChunks, DISK_SECTOR_LEN,
    },
    storage::DiskIndexWriter,
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{model::IndexConfiguration, utils::load_metadata_from_file};
use diskann_vector::distance::Metric;
use opentelemetry::global;
use opentelemetry::trace::Tracer;
use opentelemetry_sdk::trace::SdkTracerProvider;
use scopeguard::defer;

use crate::{
    backend::disk_index::{graph_data_type::GraphData, json_spancollector::JsonSpanCollector},
    inputs::disk::DiskIndexBuild,
};

#[derive(Serialize, Debug)]
pub(super) struct DiskBuildStats {
    build_time: MicroSeconds,
    span_metrics: serde_json::Value,
}

impl DiskBuildStats {
    pub(super) fn new(build_time: MicroSeconds, span_metrics: serde_json::Value) -> Self {
        Self {
            build_time,
            span_metrics,
        }
    }
}

impl fmt::Display for DiskBuildStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let build_time_seconds = self.build_time.as_seconds();
        writeln!(f, "Build time: {:.3}s", build_time_seconds)
    }
}

fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(
                process: *mut std::ffi::c_void,
                pmc: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        unsafe {
            let mut pmc = std::mem::MaybeUninit::<ProcessMemoryCounters>::zeroed().assume_init();
            pmc.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                return pmc.peak_working_set_size as f64 / (1024.0 * 1024.0);
            }
        }
        0.0
    }
    #[cfg(target_os = "linux")]
    {
        // VmHWM = peak resident set size (RSS high-water mark). This is what the
        // OOM killer reacts to and what /usr/bin/time reports as "Maximum resident
        // set size". Earlier versions read VmPeak (peak virtual address space),
        // which over-counts pages that were reserved but never physically committed
        // (e.g., faer's lazy thread-local scratch via `Vec::with_capacity().set_len()`).
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmHWM:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb.parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
        0.0
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    { 0.0 }
}

pub(super) fn build_disk_index<T, StorageProviderType>(
    storage_provider: &StorageProviderType,
    params: &DiskIndexBuild,
) -> anyhow::Result<DiskBuildStats>
where
    T: VectorRepr,
    StorageProviderType: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProviderType as StorageReadProvider>::Reader: std::marker::Send,
{
    let previous_tracer_provider = global::tracer_provider();
    let span_collector = {
        let collector = JsonSpanCollector::new();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(collector.clone())
            .build();
        global::set_tracer_provider(provider.clone());
        Some((collector, provider))
    };
    defer! {
        global::set_tracer_provider(previous_tracer_provider);
    }

    let metric: Metric = params.distance.into();
    // Experimental: for HashPrune-in-Vamana, the build graph must hold the HashPrune reservoir
    // size (l_max) per node while the final RobustPrune cuts to pruned_degree (= max_degree). So
    // widen the graph storage to l_max when VAMANA_HASHPRUNE_LMAX exceeds the default slack.
    let hp_lmax = std::env::var("VAMANA_HASHPRUNE_LMAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let max_degree_setting = if hp_lmax > params.max_degree {
        // The graph cap must be l_max PLUS slack headroom: HashPrune saturates each node to
        // ~l_max, so without slack the node sits exactly at the cap and every reverse edge
        // triggers a re-prune (a re-prune storm — ~40 occludes/insert at l_max=128). Apply the
        // same 1.3× slack DiskANN uses by default so reverse edges accumulate before re-pruning.
        config::MaxDegree::new((hp_lmax * 13 / 10).max(hp_lmax + 1))
    } else {
        config::MaxDegree::default_slack()
    };
    let config = config::Builder::new_with(
        params.max_degree,
        max_degree_setting,
        params.l_build,
        metric.into(),
        |b| {
            b.saturate_after_prune(true);
        },
    )
    .build()?;

    let data_path = params.data.to_string_lossy().to_string();

    let metadata = load_metadata_from_file(storage_provider, &data_path)?;

    let build_parameters = DiskIndexBuildParameters::new_with_algorithm(
        MemoryBudget::try_from_gb(params.build_ram_limit_gb)?,
        params.quantization_type,
        NumPQChunks::new_with(params.num_pq_chunks.get(), metadata.ndims())?,
        params.build_algorithm.clone(),
    );

    let index_configuration = IndexConfiguration::new(
        metric,
        metadata.ndims(),
        metadata.npoints(),
        ONE,
        params.num_threads,
        config,
    )
    .with_pseudo_rng();

    let disk_index_writer = DiskIndexWriter::new(
        data_path,
        params.save_path.clone(),
        Option::None,
        DISK_SECTOR_LEN,
    )?;

    let mut disk_index = DiskIndexBuilder::<GraphData<T>, StorageProviderType>::new(
        storage_provider,
        build_parameters,
        index_configuration,
        disk_index_writer,
    )?;

    let span = {
        let tracer = opentelemetry::global::tracer("benchmark");
        tracer.start("disk-index-build")
    };

    let start = std::time::Instant::now();
    disk_index.build()?;
    let total_time: MicroSeconds = start.elapsed().into();

    println!("Peak RSS: {:.1} MB", peak_rss_mb());

    drop(span);
    let span_metrics = if let Some((collector, provider)) = span_collector {
        provider.shutdown()?;
        collector.to_hierarchical_json()
    } else {
        serde_json::json!({ "span_data": [] })
    };

    Ok(DiskBuildStats::new(total_time, span_metrics))
}
