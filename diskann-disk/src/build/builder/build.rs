/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Async disk index builder implementation.
use std::{
    marker::PhantomData,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use diskann::{
    utils::{async_tools, vecid_from_usize, TryIntoVectorId, VectorRepr, ONE},
    ANNError, ANNErrorKind, ANNResult,
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
#[cfg(feature = "pipnn")]
use diskann_providers::utils::ParallelIteratorInPool;
use diskann_providers::{
    model::{
        graph::{
            provider::async_::inmem::DefaultProviderParameters,
            traits::{AdHoc, GraphDataType},
        },
        IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE, NUM_KMEANS_REPS_PQ, NUM_PQ_CENTROIDS,
    },
    storage::{AsyncIndexMetadata, DiskGraphOnly, PQStorage},
    utils::{
        create_thread_pool, find_medoid_with_sampling, RayonThreadPool, VectorDataIterator,
        MAX_MEDOID_SAMPLE_SIZE,
    },
};
use diskann_utils::io::{read_bin, write_bin};
use diskann_utils::views::MatrixView;
use tokio::task::JoinSet;
use tracing::{debug, info};

use crate::{
    build::{
        builder::{
            core::{
                determine_build_strategy, DiskIndexBuilderCore, IndexBuildStrategy,
                MergedVamanaIndexWorkflow,
            },
            inmem_builder::{load_inmem_index_builder, new_inmem_index_builder, InmemIndexBuilder},
            quantizer::BuildQuantizer,
            tokio::create_runtime,
        },
        chunking::{
            checkpoint::{
                CheckpointContext, CheckpointManager, CheckpointManagerExt,
                NaiveCheckpointRecordManager, OwnedCheckpointContext, Progress, WorkStage,
            },
            continuation::{process_while_resource_is_available_async, ChunkingConfig},
        },
        configuration::build_algorithm::BuildAlgorithm,
    },
    storage::{
        quant::{GeneratorContext, PQGeneration, PQGenerationContext, QuantDataGenerator},
        DiskIndexWriter,
    },
    utils::instrumentation::{
        BuildMergedVamanaIndexCheckpoint, DiskIndexBuildCheckpoint, PerfLogger,
    },
    DiskIndexBuildParameters, QuantizationType,
};
#[cfg(feature = "pipnn")]
use crate::{disk_index_build_parameter::BYTES_IN_GB, utils::partition_with_ram_budget};

/// Disk index builder that composes with DiskIndexBuilderCore.
pub struct DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    pub core: DiskIndexBuilderCore<'a, Data, StorageProvider>,
    /// Async-specific field: actual quantizers for async processing
    pub build_quantizer: BuildQuantizer,
}

impl<'a, Data, StorageProvider> Deref for DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    type Target = DiskIndexBuilderCore<'a, Data, StorageProvider>;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl<'a, Data, StorageProvider> DerefMut for DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.core
    }
}

impl<'a, Data, StorageProvider> DiskIndexBuilder<'a, Data, StorageProvider>
where
    Data: GraphDataType<VectorIdType = u32>,
    Data::VectorDataType: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProvider as StorageReadProvider>::Reader: std::marker::Send,
{
    pub fn new(
        storage_provider: &'a StorageProvider,
        disk_build_param: DiskIndexBuildParameters,
        index_configuration: IndexConfiguration,
        index_writer: DiskIndexWriter,
    ) -> ANNResult<Self> {
        Self::new_with_chunking_config(
            storage_provider,
            disk_build_param,
            index_configuration,
            index_writer,
            ChunkingConfig::default(),
            Box::<NaiveCheckpointRecordManager>::default(),
        )
    }

    /// Create a new async disk index builder with custom chunking configuration.
    pub fn new_with_chunking_config(
        storage_provider: &'a StorageProvider,
        disk_build_param: DiskIndexBuildParameters,
        index_configuration: IndexConfiguration,
        index_writer: DiskIndexWriter,
        chunking_config: ChunkingConfig,
        mut checkpoint_record_manager: Box<dyn CheckpointManager>,
    ) -> ANNResult<Self> {
        checkpoint_record_manager.execute_stage(
            WorkStage::Start,
            WorkStage::TrainBuildQuantizer,
            || Ok(()),
            || Ok(()),
        )?;

        let pq_storage = PQStorage::new(
            &(index_writer.get_index_path_prefix() + "_pq_pivots.bin"),
            &(index_writer.get_index_path_prefix() + "_pq_compressed.bin"),
            Some(&index_writer.get_dataset_file()),
        );

        let build_quantizer = Self::train_or_load_build_quantizer(
            disk_build_param.build_quantization(),
            &index_writer.get_index_path_prefix(),
            &index_configuration,
            &pq_storage,
            storage_provider,
            checkpoint_record_manager.as_mut(),
        )?;

        let core = DiskIndexBuilderCore {
            disk_build_param,
            index_configuration,
            index_writer,
            storage_provider,
            pq_storage,
            chunking_config,
            checkpoint_record_manager,
            _phantom: std::marker::PhantomData,
        };

        Ok(Self {
            core,
            build_quantizer,
        })
    }

    /// Train or load a quantizer for async builds, using checkpoint management.
    fn train_or_load_build_quantizer(
        build_quantization_type: &QuantizationType,
        index_path_prefix: &str,
        index_configuration: &IndexConfiguration,
        pq_storage: &PQStorage,
        storage_provider: &StorageProvider,
        checkpoint_record_manager: &mut dyn CheckpointManager,
    ) -> ANNResult<BuildQuantizer> {
        info!(
            "Training quantizer for {} quantized builds.",
            build_quantization_type.to_string()
        );

        checkpoint_record_manager.execute_stage(
            WorkStage::TrainBuildQuantizer,
            WorkStage::QuantizeFPV,
            || {
                BuildQuantizer::train::<Data, _>(
                    build_quantization_type,
                    index_path_prefix,
                    index_configuration,
                    pq_storage,
                    storage_provider,
                )
            },
            || {
                info!(
                "Skipping quantizer training, instead loading from already trained quantizer saved in the file system.",
                );
                BuildQuantizer::load(
                    build_quantization_type,
                    index_path_prefix,
                    storage_provider,
                )
            },
        )
    }

    pub fn build(&mut self) -> ANNResult<()> {
        // PiPNN is fully synchronous (rayon-only, no async). Run it outside tokio
        // to avoid the async future capturing all build state.
        #[cfg(feature = "pipnn")]
        if let &BuildAlgorithm::PiPNN { .. } = self.disk_build_param.build_algorithm() {
            return self.build_sync_pipnn();
        }

        let runtime = create_runtime(self.index_configuration.num_threads)?;
        runtime.block_on(async {
            match self.build_internal().await {
                Err(err) if err.kind() == ANNErrorKind::BuildInterrupted => {
                    info!(
                        "Index build was interrupted by continuation_checker, progress saved for resumption"
                    );
                    Ok(()) // Return success for controlled interruptions
                }
                result => result, // Pass through any other result (Ok or Err)
            }
        })
    }

    async fn build_internal(&mut self) -> ANNResult<()> {
        let mut logger = PerfLogger::new_disk_index_build_logger();

        let pool = create_thread_pool(self.index_configuration.num_threads)?;

        info!(
            "Starting index build: R={} L={} Indexing RAM budget={} T={}",
            self.index_configuration.config.pruned_degree(),
            self.index_configuration.config.l_build(),
            self.disk_build_param.build_memory_limit().in_bytes(),
            self.index_configuration.num_threads
        );

        let t_pq = std::time::Instant::now();
        self.generate_compressed_data(&pool).await?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);
        let pq_secs = t_pq.elapsed().as_secs_f64();

        let t_index = std::time::Instant::now();
        self.build_inmem_index(&pool).await?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);
        let index_secs = t_index.elapsed().as_secs_f64();

        // Return freed memory (f32 data, graph, PiPNN internals) to the OS
        // before disk layout starts. Without this, ~1.7 GB of freed-but-retained
        // memory inflates peak RSS during the disk layout phase.
        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" {
                fn malloc_trim(pad: usize) -> i32;
            }
            malloc_trim(0);
        }

        // Use physical file to pass the memory index to the disk writer
        let t_layout = std::time::Instant::now();
        self.create_disk_layout()?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::DiskLayout);
        let layout_secs = t_layout.elapsed().as_secs_f64();

        println!("Disk Index Build Phases");
        println!("  PQ compression: {:.3}s", pq_secs);
        println!("  Graph build:    {:.3}s", index_secs);
        println!("  Disk layout:    {:.3}s", layout_secs);

        Ok(())
    }

    async fn generate_compressed_data(&mut self, pool: &RayonThreadPool) -> ANNResult<()> {
        let num_points = self.index_configuration.max_points;
        let num_chunks = self.disk_build_param.search_pq_chunks();

        let storage_provider = self.core.storage_provider;

        info!(
            "Compressing data into {} bytes per vector for disk search",
            num_chunks.get()
        );

        let mut checkpoint_context = OwnedCheckpointContext::new(
            self.checkpoint_record_manager.clone_box(),
            WorkStage::QuantizeFPV,
            WorkStage::InMemIndexBuild,
        );

        let offset = match checkpoint_context.get_resumption_point()? {
            Some(offset) => offset,
            None => {
                info!("Skip the DataCompression");
                return Ok(());
            }
        };

        let quantizer_context = PQGenerationContext {
            pq_storage: self.pq_storage.clone(),
            num_chunks: num_chunks.get(),
            max_kmeans_reps: NUM_KMEANS_REPS_PQ,
            num_centers: NUM_PQ_CENTROIDS,
            seed: self.index_configuration.random_seed,
            p_val: MAX_PQ_TRAINING_SET_SIZE / (num_points as f64),
            storage_provider,
            pool,
            dim: self.index_configuration.dim,
            metric: self.index_configuration.dist_metric,
        };

        let generator_context =
            GeneratorContext::new(offset, self.pq_storage.get_compressed_data_path().into());

        let generator = QuantDataGenerator::<
            Data::VectorDataType,
            PQGeneration<Data::VectorDataType, StorageProvider, &RayonThreadPool>,
        >::new(
            self.index_writer.get_dataset_file(),
            generator_context,
            &quantizer_context,
        )?;
        let progress = generator.generate_data(storage_provider, &pool, &self.chunking_config)?;

        checkpoint_context.update(progress.clone())?;
        if let Progress::Processed(progress_point) = progress {
            let message = format!(
                "[Stage:{:?}] Build interrupt at progress {}",
                checkpoint_context.current_stage(),
                progress_point
            );
            return Err(ANNError::log_build_interrupted(message));
        }

        Ok(())
    }

    async fn build_inmem_index(&mut self, pool: &RayonThreadPool) -> ANNResult<()> {
        // PiPNN is handled by build_sync_pipnn() — should not reach here.
        #[cfg(feature = "pipnn")]
        if let &BuildAlgorithm::PiPNN { .. } = self.disk_build_param.build_algorithm() {
            return Err(ANNError::log_index_error(
                "PiPNN should use build_sync_pipnn(), not the async path",
            ));
        }

        #[cfg(not(feature = "pipnn"))]
        if !matches!(
            self.disk_build_param.build_algorithm(),
            &BuildAlgorithm::Vamana
        ) {
            return Err(ANNError::log_index_error(
                "PiPNN build algorithm requires the 'pipnn' feature to be enabled",
            ));
        }

        match determine_build_strategy::<Data>(
            &self.index_configuration,
            self.disk_build_param.build_memory_limit().in_bytes() as f64,
            self.disk_build_param.build_quantization(),
        ) {
            IndexBuildStrategy::Merged => self.build_merged_vamana_index(pool).await,
            IndexBuildStrategy::OneShot => {
                self.build_one_shot_vamana_index_with_checkpoint_record()
                    .await
            }
        }
    }

    /// Fully synchronous PiPNN build: PQ compression + PiPNN graph + disk layout.
    /// Runs without tokio runtime, avoiding the ~1.6 GB async future overhead.
    ///
    /// When the estimated one-shot RAM exceeds the build memory budget, this
    /// automatically routes to the merged shard path.
    #[cfg(feature = "pipnn")]
    fn build_sync_pipnn(&mut self) -> ANNResult<()> {
        // Check if we need the merged shard path.
        // Only applies to non-SQ1 (full precision) builds — SQ1 already compresses data.
        if !matches!(&self.build_quantizer, BuildQuantizer::Scalar1Bit(_)) {
            let config = self
                .disk_build_param
                .build_algorithm()
                .to_pipnn_config(
                    self.index_configuration.config.pruned_degree().get(),
                    self.index_configuration.dist_metric,
                    self.index_configuration.config.alpha(),
                    self.index_configuration.num_threads,
                )
                .ok_or_else(|| {
                    ANNError::log_index_error(
                        "build_pipnn_index called but build algorithm is not PiPNN",
                    )
                })?;

            let npoints = self.index_configuration.max_points;
            let ndims = self.index_configuration.dim;
            let type_size = std::mem::size_of::<Data::VectorDataType>();
            let estimated =
                estimate_pipnn_shard_ram(npoints as u64, ndims as u64, type_size, &config);
            let budget = self.disk_build_param.build_memory_limit().in_bytes() as f64;

            if estimated > budget {
                info!(
                    "PiPNN merged build: estimated {:.1} GB > budget {:.1} GB, using shard path",
                    estimated / BYTES_IN_GB,
                    budget / BYTES_IN_GB
                );
                return self.build_sync_pipnn_merged();
            }
        }

        let mut logger = PerfLogger::new_disk_index_build_logger();
        let pool = create_thread_pool(self.index_configuration.num_threads)?;

        info!(
            "Starting PiPNN build (sync): R={} L={} T={}",
            self.index_configuration.config.pruned_degree(),
            self.index_configuration.config.l_build(),
            self.index_configuration.num_threads
        );

        // PQ compression (sync — generate_compressed_data has no .await calls).
        let t_pq = std::time::Instant::now();
        {
            let runtime = create_runtime(self.index_configuration.num_threads)?;
            runtime.block_on(self.generate_compressed_data(&pool))?;
        }
        // Runtime dropped — reclaim RSS from PQ phase before PiPNN starts.
        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" {
                fn malloc_trim(pad: usize) -> i32;
            }
            malloc_trim(0);
        }
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);
        let pq_secs = t_pq.elapsed().as_secs_f64();

        // PiPNN graph build (pure rayon, no tokio).
        let t_index = std::time::Instant::now();
        self.build_pipnn_index_sync()?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);
        let index_secs = t_index.elapsed().as_secs_f64();

        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" {
                fn malloc_trim(pad: usize) -> i32;
            }
            malloc_trim(0);
        }

        let t_layout = std::time::Instant::now();
        self.create_disk_layout()?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::DiskLayout);
        let layout_secs = t_layout.elapsed().as_secs_f64();

        println!("Disk Index Build Phases");
        println!("  PQ compression: {:.3}s", pq_secs);
        println!("  Graph build:    {:.3}s", index_secs);
        println!("  Disk layout:    {:.3}s", layout_secs);

        Ok(())
    }

    /// PiPNN graph construction — sync version of build_pipnn_index.
    #[cfg(feature = "pipnn")]
    fn build_pipnn_index_sync(&mut self) -> ANNResult<()> {
        use diskann_pipnn::builder;

        let config = self
            .disk_build_param
            .build_algorithm()
            .to_pipnn_config(
                self.index_configuration.config.pruned_degree().get(),
                self.index_configuration.dist_metric,
                self.index_configuration.config.alpha(),
                self.index_configuration.num_threads,
            )
            .ok_or_else(|| {
                ANNError::log_index_error(
                    "build_pipnn_index called but build algorithm is not PiPNN",
                )
            })?;

        config
            .validate()
            .map_err(|e| ANNError::log_index_error(format!("PiPNN config error: {}", e)))?;

        info!("Building PiPNN index: max_degree={}", config.max_degree);

        let data_path = self.index_writer.get_dataset_file();

        // Build the PiPNN graph, using pre-trained SQ if available.
        let graph = match &self.build_quantizer {
            BuildQuantizer::Scalar1Bit(with_bits) => {
                // Chunked parallel quantize: load 100K vectors at a time, quantize in
                // parallel with rayon, accumulate centroid, drop chunk. Peak memory is
                // chunk (75 MB) + quantized (48 MB) instead of full f16 (790 MB).
                let sq = with_bits.quantizer();
                let scale = sq.scale();
                let inverse_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };

                let t_q = std::time::Instant::now();
                let pool = create_thread_pool(self.index_configuration.num_threads)?;
                let (qdata, medoid) = chunked_quantize_and_medoid::<Data::VectorDataType, _>(
                    &data_path,
                    self.storage_provider,
                    sq.shift(),
                    inverse_scale,
                    &pool,
                )?;
                let npoints = qdata.npoints();
                let ndims = qdata.ndims();
                info!(
                    "Chunked quantize + medoid: {:.3}s ({} points × {}d)",
                    t_q.elapsed().as_secs_f64(),
                    npoints,
                    ndims
                );

                builder::build_from_quantized(qdata, npoints, ndims, medoid, &config)
                    .map_err(|e| ANNError::log_index_error(format!("PiPNN build failed: {}", e)))?
            }
            _ => {
                // Full precision or PQ build quantization — load data in native type
                // and use build_typed to avoid upfront f32 conversion (saves ~793 MB
                // peak RSS for f16 data).
                let (npoints, ndims, data) =
                    load_data_typed::<Data::VectorDataType, _>(&data_path, self.storage_provider)?;
                builder::build_typed(&data, npoints, ndims, &config)
                    .map_err(|e| ANNError::log_index_error(format!("PiPNN build failed: {}", e)))?
            }
        };

        let save_path = self.index_writer.get_mem_index_file();
        graph
            .save_graph(std::path::Path::new(&save_path))
            .map_err(|e| ANNError::log_index_error(format!("PiPNN graph save failed: {}", e)))?;

        info!(
            "PiPNN build complete: avg_degree={:.1}, max_degree={}, isolated={}, total={:.3}s",
            graph.avg_degree(),
            graph.max_degree(),
            graph.num_isolated(),
            graph.build_stats.total_secs
        );
        // Print timing breakdown to stdout (tracing goes to OpenTelemetry spans,
        // not stdout, so use print! for user-visible output like Vamana does).
        print!("{}", graph.build_stats);

        // Mark checkpoint stages as complete so the checkpoint system is consistent.
        self.checkpoint_record_manager.execute_stage(
            WorkStage::InMemIndexBuild,
            WorkStage::WriteDiskLayout,
            || Ok(()),
            || Ok(()),
        )?;

        Ok(())
    }

    /// PiPNN merged shard build for datasets that exceed the RAM budget.
    ///
    /// Same pattern as `build_merged_vamana_index`:
    /// 1. `partition_with_ram_budget` -- K overlapping shards via streaming k-means
    /// 2. For each shard: gather data by ID map, `build_typed`, save graph, drop
    /// 3. `merge_shards` -- combine per-shard graphs into a single unified graph
    /// 4. Disk layout
    #[cfg(feature = "pipnn")]
    fn build_sync_pipnn_merged(&mut self) -> ANNResult<()> {
        use diskann_pipnn::builder;

        let mut logger = PerfLogger::new_disk_index_build_logger();
        let pool = create_thread_pool(self.index_configuration.num_threads)?;

        // PQ compression first (same as one-shot).
        let t_pq = std::time::Instant::now();
        {
            let runtime = create_runtime(self.index_configuration.num_threads)?;
            runtime.block_on(self.generate_compressed_data(&pool))?;
        }
        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" {
                fn malloc_trim(pad: usize) -> i32;
            }
            malloc_trim(0);
        }
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);
        let pq_secs = t_pq.elapsed().as_secs_f64();

        // Build PiPNN config from build parameters.
        let config = self
            .disk_build_param
            .build_algorithm()
            .to_pipnn_config(
                self.index_configuration.config.pruned_degree().get(),
                self.index_configuration.dist_metric,
                self.index_configuration.config.alpha(),
                self.index_configuration.num_threads,
            )
            .ok_or_else(|| ANNError::log_index_error("not PiPNN"))?;

        let data_path = self.index_writer.get_dataset_file();
        let merged_index_prefix = self.index_writer.get_merged_index_prefix();
        let max_degree = self.index_configuration.config.pruned_degree_u32().get();
        let ndims = self.index_configuration.dim;
        let type_size = std::mem::size_of::<Data::VectorDataType>();

        // Phase 1: Partition into overlapping shards.
        let t_part = std::time::Instant::now();
        let k_base = 2; // each point appears in 2 shards
        let sampling_rate = 0.05; // 5% subsample for k-means

        let ram_budget = self.disk_build_param.build_memory_limit().in_bytes() as f64;
        let num_threads = if config.num_threads > 0 {
            config.num_threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        };
        let ram_estimator = |npoints: u64, dim: u64| -> f64 {
            let data = npoints as f64 * dim as f64 * type_size as f64;
            let reservoirs = npoints as f64 * config.l_max as f64 * 8.0;
            let sketches = npoints as f64 * config.num_hash_planes as f64 * 4.0;
            // Partition GEMM: each thread allocates stripe(4K) × dim × 4 (f32 point data)
            // + stripe × num_leaders × 4 (dot products). This is the real peak.
            let stripe = 4096.0f64;
            let num_leaders = (npoints as f64 * config.p_samp).ceil().min(1000.0);
            let partition_bufs =
                num_threads as f64 * stripe * (dim as f64 * 4.0 + num_leaders * 4.0);
            let overhead = 100.0 * 1024.0 * 1024.0;
            data + reservoirs + sketches + partition_bufs + overhead
        };

        let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(
            self.index_configuration.random_seed,
        );
        let num_parts = partition_with_ram_budget::<Data::VectorDataType, _, _, _>(
            &data_path,
            ndims,
            sampling_rate,
            ram_budget,
            k_base,
            &merged_index_prefix,
            self.storage_provider,
            &mut rng,
            &pool,
            ram_estimator,
        )?;
        let part_secs = t_part.elapsed().as_secs_f64();
        info!(
            "PiPNN merged: partition into {} shards ({:.3}s)",
            num_parts, part_secs
        );

        // Phase 2: Build PiPNN graph per shard.
        let t_build = std::time::Instant::now();
        for shard in 0..num_parts {
            let t_shard = std::time::Instant::now();

            // Load shard data using ID map.
            let id_map_file = DiskIndexWriter::get_merged_index_subshard_id_map_file(
                &merged_index_prefix,
                shard,
            );
            let id_map = self.read_idmap(id_map_file)?;
            let shard_npoints = id_map.len();

            // Gather shard vectors from the dataset file.
            let shard_data = gather_shard_data::<Data::VectorDataType, _>(
                &data_path,
                self.storage_provider,
                &id_map,
                ndims,
            )?;

            let shard_est = ram_estimator(shard_npoints as u64, ndims as u64);
            println!(
                "  Shard {}/{}: {} pts, data={:.1} MB, est_ram={:.1} MB",
                shard,
                num_parts,
                shard_npoints,
                (shard_npoints * ndims * type_size) as f64 / (1024.0 * 1024.0),
                shard_est / (1024.0 * 1024.0),
            );

            // Build one-shot PiPNN on this shard.
            let graph = builder::build_typed(&shard_data, shard_npoints, ndims, &config)
                .map_err(|e| {
                    ANNError::log_index_error(format!("PiPNN shard {} build failed: {}", shard, e))
                })?;

            // Save shard graph in DiskANN format.
            let shard_graph_file = DiskIndexWriter::get_merged_index_subshard_mem_index_file(
                &merged_index_prefix,
                shard,
            );
            graph
                .save_graph(std::path::Path::new(&shard_graph_file))
                .map_err(|e| {
                    ANNError::log_index_error(format!("PiPNN shard {} save failed: {}", shard, e))
                })?;

            info!(
                "Shard {}/{}: built in {:.3}s (avg_degree={:.1})",
                shard,
                num_parts,
                t_shard.elapsed().as_secs_f64(),
                graph.avg_degree()
            );

            // Shard data + graph dropped here -- memory freed.
            #[cfg(target_os = "linux")]
            unsafe {
                extern "C" {
                    fn malloc_trim(pad: usize) -> i32;
                }
                malloc_trim(0);
            }
        }
        let build_secs = t_build.elapsed().as_secs_f64();
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);

        // Phase 3: Merge shard graphs.
        let t_merge = std::time::Instant::now();
        let output_vamana = self.index_writer.get_mem_index_file();
        self.merge_shards(
            &merged_index_prefix,
            num_parts,
            max_degree,
            output_vamana,
            &mut rng,
        )?;
        // Cleanup: delete only the files PiPNN created (ID maps + shard graphs).
        // Unlike Vamana, PiPNN doesn't create per-shard data files.
        for p in 0..num_parts {
            let shard_ids_file =
                DiskIndexWriter::get_merged_index_subshard_id_map_file(&merged_index_prefix, p);
            let shard_index_file =
                DiskIndexWriter::get_merged_index_subshard_mem_index_file(&merged_index_prefix, p);
            let _ = self.storage_provider.delete(&shard_ids_file);
            let _ = self.storage_provider.delete(&shard_index_file);
        }
        let merge_secs = t_merge.elapsed().as_secs_f64();

        // Phase 4: Disk layout.
        let t_layout = std::time::Instant::now();
        self.create_disk_layout()?;
        logger.log_checkpoint(DiskIndexBuildCheckpoint::DiskLayout);
        let layout_secs = t_layout.elapsed().as_secs_f64();

        println!("PiPNN Merged Build Phases");
        println!("  PQ compression: {:.3}s", pq_secs);
        println!(
            "  Partition:      {:.3}s ({} shards)",
            part_secs, num_parts
        );
        println!("  Graph build:    {:.3}s", build_secs);
        println!("  Merge:          {:.3}s", merge_secs);
        println!("  Disk layout:    {:.3}s", layout_secs);

        Ok(())
    }

    async fn build_merged_vamana_index(&mut self, pool: &RayonThreadPool) -> ANNResult<()> {
        let mut logger = PerfLogger::new_disk_index_build_logger();
        let mut workflow = MergedVamanaIndexWorkflow::new(self, pool);

        // Partition data stage
        let num_parts = workflow.partition_data(self)?;
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::PartitionData);

        // build in-memory index for each partition
        for p in 0..num_parts {
            let checkpoint_context = workflow.get_shard_context(self, p, num_parts);

            // build in-memory disk for current shard partition:{shard_base_file} and save to disk
            self.build_shard_index(&workflow.merged_index_prefix, p, checkpoint_context)
                .await?;
        }
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::BuildIndicesOnShards);

        workflow.merge_and_cleanup(self, num_parts)?;
        logger.log_checkpoint(BuildMergedVamanaIndexCheckpoint::MergeIndices);

        Ok(())
    }

    async fn build_shard_index(
        &self,
        merged_index_prefix: &str,
        shard_id: usize,
        checkpoint_context: CheckpointContext<'_>,
    ) -> ANNResult<()> {
        let stage = checkpoint_context.current_stage();
        let shard_base_file =
            DiskIndexWriter::get_merged_index_subshard_data_file(merged_index_prefix, shard_id);

        // Determine what action to take based on the checkpoint state
        let offset = match checkpoint_context.get_resumption_point()? {
            Some(offset) => offset,
            None => {
                info!(
                    "[Stage:{:?}] Skip build_shard_index for shard {} - no valid checkpoint exists",
                    stage, shard_id
                );
                return Ok(());
            }
        };

        // 1. If checkpoint is at 0, create the shard data from IDs
        if offset == 0 {
            let shard_ids_file = DiskIndexWriter::get_merged_index_subshard_id_map_file(
                merged_index_prefix,
                shard_id,
            );

            // based on id_maps, partition original data into {num_parts} shards and save them to disk temporarily
            self.retrieve_shard_data_from_ids::<Data::VectorDataType>(
                &self.index_writer.get_dataset_file(),
                &shard_ids_file,
                &shard_base_file,
            )?;
            info!("[Stage:{:?}] Generate data for shard {}", stage, shard_id);
        } else {
            info!(
                "[Stage:{:?}] Resume shard {} build with existing data",
                stage, shard_id
            );
        }

        // 2. build in-memory disk for current shard partition:{shard_base_file} and save to disk
        let index_config = self.create_shard_index_config(&shard_base_file)?;
        let shard_prefix =
            DiskIndexWriter::get_merged_index_subshard_prefix(merged_index_prefix, shard_id);
        let shard_index_file = DiskIndexWriter::get_merged_index_subshard_mem_index_file(
            merged_index_prefix,
            shard_id,
        );

        self.build_inmem_index_with_checkpoint(
            index_config,
            checkpoint_context.to_owned(),
            &shard_base_file,
            &shard_prefix,
            &shard_index_file,
        )
        .await
    }

    async fn build_one_shot_vamana_index_with_checkpoint_record(&mut self) -> ANNResult<()> {
        let checkpoint_context = OwnedCheckpointContext::new(
            self.checkpoint_record_manager.clone_box(),
            WorkStage::InMemIndexBuild,
            WorkStage::WriteDiskLayout,
        );

        self.build_inmem_index_with_checkpoint(
            self.index_configuration.clone(),
            checkpoint_context,
            &self.index_writer.get_dataset_file(),
            &self.index_writer.get_index_path_prefix(),
            &self.index_writer.get_mem_index_file(),
        )
        .await
    }

    async fn build_inmem_index_with_checkpoint(
        &self,
        config: IndexConfiguration,
        mut checkpoint_context: OwnedCheckpointContext,
        data_path: &str,
        index_path_prefix: &str,
        save_path: &str,
    ) -> ANNResult<()> {
        let stage = checkpoint_context.current_stage();
        // Check if we have a valid checkpoint for in-memory index building
        let offset = match checkpoint_context.get_resumption_point()? {
            Some(offset) => offset,
            None => {
                info!(
                    "[Stage:{:?}] Skip in-memory index build - no valid checkpoint exists",
                    stage
                );
                return Ok(());
            }
        };

        // Mark the checkpoint record as invalid. In case of a crash, the index build will start from scratch.
        checkpoint_context.mark_as_invalid()?;

        let progress = build_inmem_index::<Data::VectorDataType, _>(
            config,
            &self.build_quantizer,
            data_path,
            index_path_prefix,
            save_path,
            offset,
            &self.chunking_config,
            self.core.storage_provider,
        )
        .await?;

        checkpoint_context.update(progress.clone())?;

        match progress {
            Progress::Processed(processed) => {
                let message = format!(
                    "[Stage:{:?}] Build interrupt at progress {}",
                    stage, processed
                );
                Err(ANNError::log_build_interrupted(message))
            }
            Progress::Completed => Ok(()),
        }
    }
}

/// Estimate peak RAM (in bytes) for a PiPNN one-shot build on `npoints` vectors
/// of `ndims` dimensions with element size `type_size` bytes.
///
/// Used by `partition_with_ram_budget` (as a closure) to find the right shard count,
/// and by the routing logic to decide one-shot vs merged.
///
/// Components:
/// - data:       npoints * ndims * type_size
/// - reservoirs: npoints * l_max * 8  (each reservoir entry is a (u32,f32) pair)
/// - sketches:   npoints * num_hash_planes * 4  (one f32 per plane per point)
/// - overhead:   150 MB (partition buffers, leaf GEMM buffers, allocator fragmentation)
#[cfg(feature = "pipnn")]
fn estimate_pipnn_shard_ram(
    npoints: u64,
    ndims: u64,
    type_size: usize,
    config: &diskann_pipnn::PiPNNConfig,
) -> f64 {
    let data = npoints as f64 * ndims as f64 * type_size as f64;
    let reservoirs = npoints as f64 * config.l_max as f64 * 8.0;
    let sketches = npoints as f64 * config.num_hash_planes as f64 * 4.0;
    let overhead = 150.0 * 1024.0 * 1024.0;
    data + reservoirs + sketches + overhead
}

/// Read a subset of vectors from a DiskANN `.bin` file by their global IDs.
///
/// Returns a contiguous `Vec<T>` in shard-local order (i.e. `result[i*ndims..(i+1)*ndims]`
/// is the vector for `id_map[i]`). Global IDs are sorted before reading so I/O is
/// mostly sequential, minimising seeks.
#[cfg(feature = "pipnn")]
fn gather_shard_data<T, SP>(
    data_path: &str,
    storage_provider: &SP,
    id_map: &[u32],
    ndims: usize,
) -> ANNResult<Vec<T>>
where
    T: VectorRepr,
    SP: StorageReadProvider,
{
    use std::io::{Read, Seek, SeekFrom};

    let type_size = std::mem::size_of::<T>();
    let row_bytes = ndims * type_size;
    let shard_npoints = id_map.len();

    let mut reader = storage_provider.open_reader(data_path)?;

    // Skip the 8-byte header (u32 npoints + u32 ndims).
    let mut header = [0u8; 8];
    Read::read_exact(&mut reader, &mut header)?;

    let mut data = vec![T::zeroed(); shard_npoints * ndims];
    let mut row_buf = vec![0u8; row_bytes];

    // Sort IDs so we read mostly sequentially (minimise seeking).
    let mut sorted_indices: Vec<(usize, u32)> = id_map
        .iter()
        .enumerate()
        .map(|(shard_idx, &global_id)| (shard_idx, global_id))
        .collect();
    sorted_indices.sort_unstable_by_key(|&(_, gid)| gid);

    let mut current_pos = 8u64; // right after header
    for &(shard_idx, global_id) in &sorted_indices {
        let target_pos = 8 + (global_id as u64) * (row_bytes as u64);
        if target_pos != current_pos {
            Seek::seek(&mut reader, SeekFrom::Start(target_pos))?;
        }
        Read::read_exact(&mut reader, &mut row_buf)?;
        let src: &[T] = bytemuck::cast_slice(&row_buf);
        data[shard_idx * ndims..(shard_idx + 1) * ndims].copy_from_slice(src);
        current_pos = target_pos + row_bytes as u64;
    }

    Ok(data)
}

/// Load data in its native type T without converting to f32.
#[cfg(feature = "pipnn")]
fn load_data_typed<T, SP>(
    data_path: &str,
    storage_provider: &SP,
) -> ANNResult<(usize, usize, Vec<T>)>
where
    T: VectorRepr,
    SP: StorageReadProvider,
{
    let matrix = read_bin::<T>(&mut storage_provider.open_reader(data_path)?)?;
    let npoints = matrix.nrows();
    let ndims = matrix.ncols();
    let data: Vec<T> = matrix.into_inner().into_vec();

    Ok((npoints, ndims, data))
}

/// Chunked parallel quantize: read data in 100K-vector chunks, quantize each
/// chunk in parallel with rayon, accumulate centroid, then find medoid.
/// Peak memory: chunk (~75 MB for 100K×384×f16) + quantized (~48 MB).
#[cfg(feature = "pipnn")]
fn chunked_quantize_and_medoid<T, SP>(
    data_path: &str,
    storage_provider: &SP,
    shift: &[f32],
    inverse_scale: f32,
    pool: &RayonThreadPool,
) -> ANNResult<(diskann_pipnn::quantize::QuantizedData, usize)>
where
    T: VectorRepr,
    SP: StorageReadProvider,
{
    use diskann_utils::io::Metadata;
    use std::io::{Read, Seek};

    use rayon::prelude::*;

    let mut reader = storage_provider.open_reader(data_path)?;
    let metadata = Metadata::read(&mut reader)
        .map_err(|e| ANNError::log_index_error(format!("Failed to read header: {}", e)))?;
    let npoints = metadata.npoints();
    let ndims = metadata.ndims();

    // Pre-allocate quantized output (u64-aligned).
    let bytes_per_vec = ndims.div_ceil(64) * 8;
    let total_bytes = npoints * bytes_per_vec;
    let u64s_total = total_bytes / 8;
    let mut bits_u64 = vec![0u64; u64s_total];
    let mut bits = unsafe {
        let ptr = bits_u64.as_mut_ptr() as *mut u8;
        let len = bits_u64.len().checked_mul(8).expect("overflow");
        let cap = bits_u64.capacity().checked_mul(8).expect("overflow");
        std::mem::forget(bits_u64);
        Vec::from_raw_parts(ptr, len, cap)
    };

    // Read + quantize + centroid in 100K-vector chunks.
    // Each chunk: read from disk → parallel T→f32 convert + quantize via rayon.
    let chunk_size: usize = 100_000;
    let mut centroid = vec![0.0f32; ndims];
    let mut chunk_t = vec![<T as bytemuck::Zeroable>::zeroed(); chunk_size * ndims];

    let mut offset = 0;
    while offset < npoints {
        let n = (npoints - offset).min(chunk_size);
        let chunk_slice = &mut chunk_t[..n * ndims];
        reader
            .read_exact(bytemuck::must_cast_slice_mut::<T, u8>(chunk_slice))
            .map_err(|e| ANNError::log_index_error(format!("Read failed: {}", e)))?;

        // Parallel quantize this chunk into the output bits buffer.
        let bits_chunk = &mut bits[offset * bytes_per_vec..(offset + n) * bytes_per_vec];
        bits_chunk
            .par_chunks_mut(bytes_per_vec)
            .enumerate()
            .for_each_in_pool(pool, |(i, out)| {
                let src = &chunk_slice[i * ndims..(i + 1) * ndims];
                thread_local! {
                    static BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
                }
                BUF.with(|cell| {
                    let mut buf = cell.borrow_mut();
                    if buf.len() < ndims {
                        buf.resize(ndims, 0.0);
                    }
                    let f = &mut buf[..ndims];
                    T::as_f32_into(src, f).expect("f32 conversion");
                    for d in 0..ndims {
                        let code =
                            ((f[d] - shift[d]) * inverse_scale).clamp(0.0, 1.0).round() as u8;
                        if code > 0 {
                            out[d / 8] |= 1 << (d % 8);
                        }
                    }
                });
            });

        // Accumulate centroid (sequential, cheap).
        let mut f32_buf = vec![0.0f32; ndims];
        for i in 0..n {
            T::as_f32_into(&chunk_slice[i * ndims..(i + 1) * ndims], &mut f32_buf)
                .expect("f32 conversion");
            for d in 0..ndims {
                centroid[d] += f32_buf[d];
            }
        }

        offset += n;
    }
    drop(chunk_t); // Free chunk buffer before medoid pass.

    let qdata =
        diskann_pipnn::quantize::QuantizedData::from_raw(bits, bytes_per_vec, ndims, npoints);

    // Medoid: find point closest to centroid. Re-read from disk (OS page cache hit).
    let inv_n = 1.0 / npoints as f32;
    for c in centroid.iter_mut() {
        *c *= inv_n;
    }

    reader
        .seek(std::io::SeekFrom::Start(8))
        .map_err(|e| ANNError::log_index_error(format!("Seek failed: {}", e)))?;
    let mut chunk_t2 = vec![<T as bytemuck::Zeroable>::zeroed(); chunk_size * ndims];
    let mut f32_buf = vec![0.0f32; ndims];
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    offset = 0;
    while offset < npoints {
        let n = (npoints - offset).min(chunk_size);
        let chunk_slice = &mut chunk_t2[..n * ndims];
        reader
            .read_exact(bytemuck::must_cast_slice_mut::<T, u8>(chunk_slice))
            .map_err(|e| ANNError::log_index_error(format!("Read failed: {}", e)))?;

        for i in 0..n {
            T::as_f32_into(&chunk_slice[i * ndims..(i + 1) * ndims], &mut f32_buf)
                .expect("f32 conversion");
            let mut dist = 0.0f32;
            for d in 0..ndims {
                let diff = f32_buf[d] - centroid[d];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_idx = offset + i;
            }
        }
        offset += n;
    }

    Ok((qdata, best_idx))
}

#[allow(clippy::too_many_arguments)]
async fn build_inmem_index<T, StorageProvider>(
    config: IndexConfiguration,
    quantizer: &BuildQuantizer,
    data_path: &str,
    index_path_prefix: &str,
    save_path: &str,
    offset: usize,
    chunking_config: &ChunkingConfig,
    storage_provider: &StorageProvider,
) -> ANNResult<Progress>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProvider as StorageReadProvider>::Reader: std::marker::Send,
{
    if offset >= config.max_points {
        return Err(ANNError::log_index_error(format!(
            "Offset {} exceeds max points {}",
            offset, config.max_points
        )));
    }

    // use either user-specified number of threads or default to available parallelism
    let num_tasks = NonZeroUsize::new(config.num_threads)
        .or_else(|| std::thread::available_parallelism().ok())
        .ok_or_else(|| ANNError::log_index_error("Failed to determine number of threads"))?;

    // Associated data will only be used in the write_disk_layout function which only requires the none-partitioned associated data stream.
    let dataset_iter = Arc::new(Mutex::new({
        let iter =
            VectorDataIterator::<_, AdHoc<T>>::new(data_path, Option::None, storage_provider)?;
        iter.enumerate().skip(offset)
    }));

    let index_store: IndexStore<'_, _, T> = if offset == 0 {
        IndexStore::new_index(
            config,
            quantizer,
            data_path,
            index_path_prefix,
            storage_provider,
        )
        .await?
    } else {
        IndexStore::from_checkpoint(
            config,
            quantizer,
            index_path_prefix,
            offset,
            storage_provider,
        )
        .await?
    };

    let index = &index_store.index;
    let progress =
        run_build_with_chunking(index, dataset_iter, num_tasks, offset, chunking_config).await?;

    #[cfg(debug_assertions)]
    log_build_stats::<_>(index).await?;

    match progress {
        Progress::Processed(processed) => {
            index_store.save_checkpoint(processed).await?;
        }

        Progress::Completed => {
            // If the progress is completed, we can run the final prune and save the index to disk.
            run_final_prune(index, num_tasks).await?;

            index_store.save_final_index(save_path).await?;
        }
    }

    Ok(progress)
}

#[cfg(debug_assertions)]
/// Log statistics about the build process
async fn log_build_stats<T: VectorRepr>(index: &Arc<dyn InmemIndexBuilder<T>>) -> ANNResult<()> {
    debug!(
        "Number of points reachable in the graph: {}",
        index.count_reachable_nodes().await?
    );

    let (full_vector, quant_vector) = index.counts_for_get_vector();
    let capacity = index.capacity();
    debug!(
        "Number of get vector calls per insert: {}",
        full_vector as f32 / capacity as f32
    );
    debug!(
        "Number of get quantized vector calls per insert: {}",
        quant_vector as f32 / capacity as f32
    );

    Ok(())
}

fn set_start_point_to_medoid<T, StorageReader>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    path: &str,
    random_seed: Option<u64>,
    reader: &StorageReader,
) -> ANNResult<usize>
where
    T: VectorRepr,
    StorageReader: StorageReadProvider,
{
    let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(random_seed);
    let (medoid, medoid_id) =
        find_medoid_with_sampling::<T, _>(path, reader, MAX_MEDOID_SAMPLE_SIZE, &mut rng)?;

    index.set_start_point(medoid.as_slice())?;

    debug!("Set start point to medoid ID: {}", medoid_id);

    Ok(medoid_id)
}

async fn run_build_with_chunking<T, I>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    iterator: Arc<Mutex<I>>,
    num_tasks: NonZeroUsize,
    offset: usize,
    chunking_config: &ChunkingConfig,
) -> ANNResult<Progress>
where
    T: VectorRepr,
    I: Iterator<Item = (usize, (Box<[T]>, ()))> + Send + 'static,
{
    let total_points = index.capacity();
    let chunk_size = chunking_config.inmemory_build_chunk_vector_count;

    let chunks = (offset..total_points)
        .step_by(chunk_size) // Create an infinite iterator that steps by `chunk_size`.
        .take_while(move |start| start < &total_points) // Take elements while the start is less than the range.
        .map(move |start| (start, usize::min(start + chunk_size, total_points)));

    let progress = process_while_resource_is_available_async(
        |chunk| process_chunk(index, iterator.clone(), num_tasks, chunk.0, chunk.1),
        chunks,
        chunking_config.continuation_checker.clone_box(),
    )
    .await?
    .map(|num_chunks| num_chunks * chunk_size);

    match progress {
        Progress::Processed(num_points) => {
            info!(
                "Linked #{} points. Start #{}, end #{} ",
                num_points,
                offset,
                num_points + offset
            );
        }
        Progress::Completed => {
            info!("Linked all points. Num points: #{}", total_points);
        }
    }

    let progress = progress.map(|num_points| num_points + offset);

    Ok(progress)
}

async fn process_chunk<T, Iter>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    iterator: Arc<Mutex<Iter>>,
    num_tasks: NonZeroUsize,
    start: usize,
    end: usize,
) -> ANNResult<()>
where
    T: VectorRepr,
    Iter: Iterator<Item = (usize, (Box<[T]>, ()))> + Send + 'static,
{
    debug!("Processing chunk from #{} to #{}", start, end);

    let partitions = async_tools::PartitionIter::new(end - start, num_tasks);

    let mut tasks = JoinSet::new();

    for partition in partitions {
        let index_clone = index.clone();
        let iterator_clone = iterator.clone();
        tasks.spawn(async move {
            for _ in partition {
                let vector_data = {
                    let mut guard = iterator_clone.lock().map_err(|_| {
                        ANNError::log_index_error("Poisoned mutex during construction")
                    })?;
                    guard.next()
                };

                match vector_data {
                    Some((i, (vector, _))) => {
                        let id = vecid_from_usize(i)?;
                        index_clone.insert_vector(id, vector.as_ref()).await?;
                    }
                    None => break,
                }
            }
            ANNResult::Ok(())
        });
    }

    // Wait for all tasks to complete.
    while let Some(res) = tasks.join_next().await {
        res.map_err(|_| ANNError::log_index_error("A spawned insert task failed"))??;
    }

    debug!("Completed chunk #{} to #{}", start, end);
    Ok(())
}

async fn run_final_prune<T: VectorRepr>(
    index: &Arc<dyn InmemIndexBuilder<T>>,
    num_tasks: NonZeroUsize,
) -> ANNResult<()> {
    let partitions = async_tools::PartitionIter::new(index.total_points(), num_tasks);

    let mut tasks = JoinSet::new();

    for partition in partitions {
        let index_clone = index.clone();
        tasks.spawn(async move {
            let start_index = partition.start.try_into_vector_id()?;
            let end_index = partition.end.try_into_vector_id()?;

            let range = start_index..end_index;
            index_clone.final_prune(range).await
        });
    }

    // Wait for all final prune tasks to complete
    while let Some(res) = tasks.join_next().await {
        res.map_err(|_| ANNError::log_index_error("A spawned final prune task failed"))??;
    }

    Ok(())
}

/// Manages the persistence operations for in-memory index building
struct IndexStore<'a, S, T>
where
    S: StorageReadProvider + StorageWriteProvider,
{
    pub index: Arc<dyn InmemIndexBuilder<T>>,
    start_point: StartPoint,

    metadata: AsyncIndexMetadata,
    storage_provider: &'a S,
    _phantom: PhantomData<T>,
}

impl<'a, S, T> IndexStore<'a, S, T>
where
    S: StorageReadProvider + StorageWriteProvider + 'static,
    <S as StorageReadProvider>::Reader: std::marker::Send,
    T: VectorRepr,
{
    /// Create a new persistence manager with a fresh index
    pub async fn new_index(
        config: IndexConfiguration,
        build_quantizer: &BuildQuantizer,
        data_path: &'a str,
        index_path_prefix: &str,
        storage_provider: &'a S,
    ) -> ANNResult<IndexStore<'a, S, T>> {
        // Create new index
        let index_config = config.config.clone();

        let provider_parameters = DefaultProviderParameters {
            max_points: config.max_points,
            frozen_points: ONE,
            metric: config.dist_metric,
            dim: config.dim,
            // This is the true maximum degree.
            max_degree: index_config.max_degree_u32().get(),
            prefetch_lookahead: config.prefetch_lookahead.map(|x| x.get()),
            prefetch_cache_line_level: config.prefetch_cache_line_level,
        };

        let index =
            new_inmem_index_builder::<T>(index_config, provider_parameters, build_quantizer)?;

        let medoid_id = set_start_point_to_medoid::<T, _>(
            &index,
            data_path,
            config.random_seed,
            storage_provider,
        )?;
        let start_point = StartPoint::new(vecid_from_usize(medoid_id)?);

        Ok(Self {
            index,
            start_point,
            metadata: AsyncIndexMetadata::new(index_path_prefix),
            storage_provider,
            _phantom: PhantomData,
        })
    }

    /// Load an existing index from a checkpoint
    pub async fn from_checkpoint(
        config: IndexConfiguration,
        build_quantizer: &BuildQuantizer,
        index_path_prefix: &str,
        _offset: usize,
        storage_provider: &'a S,
    ) -> ANNResult<Self> {
        let metadata = AsyncIndexMetadata::new(index_path_prefix);

        // Load existing index from resumable context
        let index = load_inmem_index_builder::<T, _>(
            storage_provider,
            build_quantizer,
            config,
            index_path_prefix,
        )
        .await?;

        // Load existing start point
        let start_point =
            StartPoint::load(&metadata.additional_points_id_path(), storage_provider)?;

        Ok(Self {
            index,
            start_point,
            metadata,
            storage_provider,
            _phantom: PhantomData,
        })
    }

    /// Save intermediate state during build interruption
    pub async fn save_checkpoint(&self, _processed: usize) -> ANNResult<()> {
        self.index
            .save_index(self.storage_provider, &self.metadata)
            .await?;

        self.start_point.save(
            &self.metadata.additional_points_id_path(),
            self.storage_provider,
        )?;

        Ok(())
    }

    /// Save the finalized index to disk
    pub async fn save_final_index(&self, save_path: &str) -> ANNResult<()> {
        self.clean_temp_files()?;

        // Use physical file to contact with index writer
        self.index
            .save_graph(
                self.storage_provider,
                &(self.start_point.id(), DiskGraphOnly::new(save_path)),
            )
            .await?;

        Ok(())
    }

    /// Removes temporary files created during index building
    fn clean_temp_files(&self) -> ANNResult<()> {
        let files = [
            self.metadata.prefix().to_string(),
            self.metadata.data_path(),
            self.metadata.additional_points_id_path(),
        ];

        for file in files.iter() {
            if self.storage_provider.exists(file) {
                debug!("Deleting temporary file: {}", file);
                self.storage_provider.delete(file)?;
            }
        }
        Ok(())
    }
}

/// Manages persistence of start point IDs for resumable builds
struct StartPoint(u32);

impl StartPoint {
    fn new(id: u32) -> Self {
        Self(id)
    }

    fn load<StorageReader>(path: &str, reader: &StorageReader) -> ANNResult<Self>
    where
        StorageReader: StorageReadProvider,
    {
        if !reader.exists(path) {
            return Err(ANNError::log_file_not_found_error(format!(
                "Start point ID file {} does not exist",
                path
            )));
        }
        let data = read_bin::<u32>(&mut reader.open_reader(path)?)?;

        let start_point_id = data.try_get(0, 0).ok_or_else(|| {
            ANNError::log_invalid_file_format(format!("Start point ID file {} is empty", path))
        })?;

        debug!("Loaded start point ID {} from {}", *start_point_id, path);
        Ok(Self(*start_point_id))
    }

    fn save<StorageWriter>(&self, path: &str, storage_provider: &StorageWriter) -> ANNResult<()>
    where
        StorageWriter: StorageWriteProvider,
    {
        write_bin(
            MatrixView::row_vector(std::slice::from_ref(&self.0)),
            &mut storage_provider.create_for_write(path)?,
        )?;
        debug!("Saved start point ID {} to {}", self.0, path);
        Ok(())
    }

    fn id(&self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod start_point_tests {
    use std::io::Write;

    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_utils::io::Metadata;

    use super::*;

    #[test]
    fn test_start_point_creation() {
        let id = 42u32;
        let start_point = StartPoint::new(id);
        assert_eq!(start_point.id(), id);
    }

    #[test]
    fn test_start_point_save_and_load() {
        let file_path = "/start_point_test.bin";
        let storage_provider = VirtualStorageProvider::new_memory();

        // Create and save a start point
        let id = 42u32;
        let start_point = StartPoint::new(id);
        start_point.save(file_path, &storage_provider).unwrap();

        // Load the start point and verify it matches the original
        let loaded_start_point = StartPoint::load(file_path, &storage_provider).unwrap();
        assert_eq!(loaded_start_point.id(), id);
    }

    #[test]
    fn test_start_point_load_nonexistent_file() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let result = StartPoint::load("/nonexistent_file.bin", &storage_provider);
        assert_eq!(
            result.err().unwrap().kind(),
            ANNErrorKind::FileNotFoundError
        );
    }

    #[test]
    fn test_start_point_load_empty_file() {
        let file_path = "/empty_file.bin";
        let storage_provider = VirtualStorageProvider::new_memory();

        // Create an empty file
        {
            let mut file = storage_provider.create_for_write(file_path).unwrap();
            file.write_all(&[]).unwrap();
        }

        let result = StartPoint::load(file_path, &storage_provider);
        assert_eq!(result.err().unwrap().kind(), ANNErrorKind::IOError);
    }

    #[test]
    fn test_start_point_load_invalid_data() {
        let file_path = "/invalid_data.bin";
        let storage_provider = VirtualStorageProvider::new_memory();

        // Create a file with invalid data
        {
            let mut file = storage_provider.create_for_write(file_path).unwrap();
            let npts = 0;
            let dim = 1;
            Metadata::new(npts, dim).unwrap().write(&mut file).unwrap();
        }

        let result = StartPoint::load(file_path, &storage_provider);
        assert_eq!(
            result.err().unwrap().kind(),
            ANNErrorKind::InvalidFileFormatError
        );
    }
}

#[cfg(test)]
#[cfg(feature = "pipnn")]
mod pipnn_merged_tests {
    use super::*;
    use diskann_vector::distance::Metric;

    /// Helper to create a PiPNNConfig for tests.
    fn test_config(l_max: usize, num_hash_planes: usize) -> diskann_pipnn::PiPNNConfig {
        diskann_pipnn::PiPNNConfig {
            num_hash_planes,
            c_max: 1024,
            c_min: 256,
            p_samp: 0.005,
            fanout: vec![10, 3],
            k: 2,
            max_degree: 64,
            replicas: 1,
            l_max,
            metric: Metric::CosineNormalized,
            final_prune: false,
            alpha: 1.2,
            num_threads: 1,
        }
    }

    #[test]
    fn test_pipnn_shard_ram_estimate_enron() {
        // Enron 1M, fp16 (2 bytes), 384d, l_max=64, 14 planes
        // With K=4 shards, k_base=2: shard_npts ~544K
        let config = test_config(64, 14);
        let est = estimate_pipnn_shard_ram(544_000, 384, 2, &config);
        // Should be < 1 GB but > 500 MB
        assert!(
            est < 1.0 * 1024.0 * 1024.0 * 1024.0,
            "estimated {:.1} MB should be < 1 GB",
            est / (1024.0 * 1024.0)
        );
        assert!(
            est > 500.0 * 1024.0 * 1024.0,
            "estimated {:.1} MB should be > 500 MB",
            est / (1024.0 * 1024.0)
        );
    }

    #[test]
    fn test_pipnn_shard_ram_estimate_components() {
        // Verify each component contributes correctly.
        let config = test_config(64, 14);
        let npoints = 100_000u64;
        let ndims = 384u64;
        let type_size = 2usize; // fp16

        let est = estimate_pipnn_shard_ram(npoints, ndims, type_size, &config);

        let data = npoints as f64 * ndims as f64 * type_size as f64;
        let reservoirs = npoints as f64 * 64.0 * 8.0;
        let sketches = npoints as f64 * 14.0 * 4.0;
        let overhead = 150.0 * 1024.0 * 1024.0;
        let expected = data + reservoirs + sketches + overhead;

        assert!(
            (est - expected).abs() < 1.0,
            "estimate {est} should match manual calc {expected}"
        );
    }

    #[test]
    fn test_pipnn_shard_ram_scales_with_points() {
        let config = test_config(64, 14);
        let est_small = estimate_pipnn_shard_ram(100_000, 384, 2, &config);
        let est_large = estimate_pipnn_shard_ram(1_000_000, 384, 2, &config);
        assert!(
            est_large > est_small,
            "larger dataset should need more RAM"
        );
        // The overhead is constant, so 10x points should give roughly (but not exactly) 10x
        let ratio = (est_large - 150.0 * 1024.0 * 1024.0) / (est_small - 150.0 * 1024.0 * 1024.0);
        assert!(
            (ratio - 10.0).abs() < 0.01,
            "variable part should scale linearly, ratio={ratio}"
        );
    }

    /// Write a DiskANN .bin file with the given f32 data (npoints x ndims).
    fn write_test_bin(
        storage: &impl StorageWriteProvider,
        path: &str,
        data: &[f32],
        npoints: usize,
        ndims: usize,
    ) {
        use diskann_utils::io::Metadata;
        use std::io::Write;

        let mut w = storage.create_for_write(path).unwrap();
        let meta = Metadata::new(npoints, ndims).unwrap();
        meta.write(&mut w).unwrap();
        w.write_all(bytemuck::cast_slice::<f32, u8>(data)).unwrap();
    }

    #[test]
    fn test_gather_shard_data_basic() {
        let storage = diskann_providers::storage::VirtualStorageProvider::new_memory();
        let ndims = 3;
        let npoints = 5;
        // 5 points, 3 dims: point i has values [i*10+1, i*10+2, i*10+3]
        let data: Vec<f32> = (0..npoints)
            .flat_map(|i| {
                let base = (i * 10) as f32;
                vec![base + 1.0, base + 2.0, base + 3.0]
            })
            .collect();
        write_test_bin(&storage, "/test.bin", &data, npoints, ndims);

        // Gather points 3, 0, 4 (out of order).
        let id_map = vec![3u32, 0, 4];
        let result = gather_shard_data::<f32, _>("/test.bin", &storage, &id_map, ndims).unwrap();

        // result[0] should be point 3: [31, 32, 33]
        assert_eq!(&result[0..3], &[31.0, 32.0, 33.0]);
        // result[1] should be point 0: [1, 2, 3]
        assert_eq!(&result[3..6], &[1.0, 2.0, 3.0]);
        // result[2] should be point 4: [41, 42, 43]
        assert_eq!(&result[6..9], &[41.0, 42.0, 43.0]);
    }

    #[test]
    fn test_gather_shard_data_sequential() {
        let storage = diskann_providers::storage::VirtualStorageProvider::new_memory();
        let ndims = 2;
        let npoints = 4;
        let data: Vec<f32> = (0..npoints)
            .flat_map(|i| vec![i as f32, (i * 100) as f32])
            .collect();
        write_test_bin(&storage, "/seq.bin", &data, npoints, ndims);

        // Gather all points in order.
        let id_map = vec![0u32, 1, 2, 3];
        let result = gather_shard_data::<f32, _>("/seq.bin", &storage, &id_map, ndims).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_gather_shard_data_single_point() {
        let storage = diskann_providers::storage::VirtualStorageProvider::new_memory();
        let ndims = 4;
        let npoints = 10;
        let data: Vec<f32> = (0..npoints * ndims).map(|i| i as f32).collect();
        write_test_bin(&storage, "/single.bin", &data, npoints, ndims);

        let id_map = vec![7u32];
        let result = gather_shard_data::<f32, _>("/single.bin", &storage, &id_map, ndims).unwrap();
        // Point 7 starts at index 7*4=28
        assert_eq!(&result[..], &data[28..32]);
    }

    #[test]
    fn test_gather_shard_data_preserves_shard_order() {
        // Ensure the output is in shard-local order (id_map order), not sorted order.
        let storage = diskann_providers::storage::VirtualStorageProvider::new_memory();
        let ndims = 2;
        let npoints = 3;
        let data: Vec<f32> = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        write_test_bin(&storage, "/order.bin", &data, npoints, ndims);

        // Request in reverse order: [2, 1, 0]
        let id_map = vec![2u32, 1, 0];
        let result = gather_shard_data::<f32, _>("/order.bin", &storage, &id_map, ndims).unwrap();

        // Shard-local index 0 = global 2 = [30, 31]
        assert_eq!(&result[0..2], &[30.0, 31.0]);
        // Shard-local index 1 = global 1 = [20, 21]
        assert_eq!(&result[2..4], &[20.0, 21.0]);
        // Shard-local index 2 = global 0 = [10, 11]
        assert_eq!(&result[4..6], &[10.0, 11.0]);
    }

    #[test]
    fn test_routing_oneshot_when_budget_sufficient() {
        // Enron 1M, fp16, 384d — estimated ~1.5 GB.
        // With 32 GB budget, one-shot should be chosen.
        let config = test_config(64, 14);
        let estimated = estimate_pipnn_shard_ram(1_090_000, 384, 2, &config);
        let budget = 32.0 * 1024.0 * 1024.0 * 1024.0; // 32 GB
        assert!(
            estimated <= budget,
            "estimated {:.1} GB should fit in 32 GB budget",
            estimated / (1024.0 * 1024.0 * 1024.0)
        );
    }

    #[test]
    fn test_routing_merged_when_budget_exceeded() {
        // Enron 1M, fp16, 384d — estimated ~1.5 GB.
        // With 1 GB budget, merged should be chosen.
        let config = test_config(64, 14);
        let estimated = estimate_pipnn_shard_ram(1_090_000, 384, 2, &config);
        let budget = 1.0 * 1024.0 * 1024.0 * 1024.0; // 1 GB
        assert!(
            estimated > budget,
            "estimated {:.1} GB should exceed 1 GB budget",
            estimated / (1024.0 * 1024.0 * 1024.0)
        );
    }

    #[test]
    fn test_routing_merged_when_budget_tight() {
        // 500K points, fp32 (4 bytes), 128d, l_max=128, 12 planes.
        // data:       500K * 128 * 4 = 256 MB
        // reservoirs: 500K * 128 * 8 = 512 MB
        // sketches:   500K * 12 * 4  =  24 MB
        // overhead:                     150 MB
        // total:                       ~942 MB
        let config = test_config(128, 12);
        let estimated = estimate_pipnn_shard_ram(500_000, 128, 4, &config);
        // 942 MB should exceed a 900 MB budget
        let budget = 900.0 * 1024.0 * 1024.0;
        assert!(
            estimated > budget,
            "estimated {:.1} MB should exceed 900 MB budget",
            estimated / (1024.0 * 1024.0)
        );
        // But should fit in a 1 GB budget
        let budget_1g = 1024.0 * 1024.0 * 1024.0;
        assert!(
            estimated < budget_1g,
            "estimated {:.1} MB should fit in 1 GB budget",
            estimated / (1024.0 * 1024.0)
        );
    }
}
