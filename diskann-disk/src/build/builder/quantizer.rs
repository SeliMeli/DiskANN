/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Disk index quantizer implementation.
use diskann::{ANNError, ANNResult};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    index::diskann_async::train_pq,
    model::{
        graph::{
            provider::async_::{common::NoStore, inmem::{WithBits, WithTurboQuant}},
            traits::GraphDataType,
        },
        FixedChunkPQTable, IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE,
    },
    storage::{PQStorage, SQStorage},
    utils::{BridgeErr, PQPathNames},
};
use diskann_quantization::scalar::train::ScalarQuantizationParameters;
use diskann_quantization::turboquant::TurboQuantQuantizer;
use diskann_utils::views::MatrixView;
use rand::{SeedableRng, rngs::StdRng};
use tracing::info;

use crate::QuantizationType;

/// Quantizer types used specifically for async disk index building.
#[derive(Clone)]
pub enum BuildQuantizer {
    NoQuant(NoStore),
    Scalar1Bit(WithBits<1>),
    PQ(FixedChunkPQTable),
    TurboQuant(WithTurboQuant),
}

impl BuildQuantizer {
    /// Train a new quantizer from scratch.
    pub fn train<Data, StorageProvider>(
        build_quantization_type: &QuantizationType,
        index_path_prefix: &str,
        index_configuration: &IndexConfiguration,
        pq_storage: &PQStorage,
        storage_provider: &StorageProvider,
    ) -> ANNResult<Self>
    where
        Data: GraphDataType<VectorIdType = u32>,
        StorageProvider: StorageReadProvider + StorageWriteProvider,
    {
        let num_points = index_configuration.max_points;
        let p_val = MAX_PQ_TRAINING_SET_SIZE / (num_points as f64);
        match *build_quantization_type {
            QuantizationType::FP => Ok(Self::NoQuant(NoStore)),
            QuantizationType::PQ { num_chunks } => {
                let table = {
                    //generate pq pivots.
                    let seed = index_configuration.random_seed;
                    let mut rnd =
                        diskann_providers::utils::create_rnd_provider_from_optional_seed(seed)
                            .create_rnd();
                    let (train_data, train_size, train_dim) = pq_storage
                        .get_random_train_data_slice::<Data::VectorDataType, _>(
                            p_val,
                            storage_provider,
                            &mut rnd,
                        )?;
                    train_pq(
                        MatrixView::try_from(&train_data, train_size, train_dim).bridge_err()?,
                        num_chunks,
                        &mut rnd,
                        index_configuration.num_threads,
                    )?
                };
                // Save at checkpoint. Note the the compressed data path and pivots path here
                // are different than the ones used in quant vector generation.
                let pq_paths = PQPathNames::new(index_path_prefix);
                let pq_build_storage =
                    PQStorage::new(&pq_paths.pivots, &pq_paths.compressed_data, None);
                pq_build_storage.write_pivot_data(
                    table.get_pq_table(),
                    table.get_centroids(),
                    table.get_chunk_offsets(),
                    table.get_num_centers(),
                    table.get_dim(),
                    storage_provider,
                )?;
                Ok(Self::PQ(table))
            }
            QuantizationType::SQ {
                nbits,
                standard_deviation,
            } => {
                if nbits != 1 {
                    return Err(ANNError::log_index_config_error(
                        "build_quantization_type".to_string(),
                        "SQ quantization is only supported for 1 bit".to_string(),
                    ));
                }
                let rng = diskann_providers::utils::create_rnd_provider_from_optional_seed(
                    index_configuration.random_seed,
                );
                let (train_data_vector, train_size, train_dim) = pq_storage
                    .get_random_train_data_slice::<Data::VectorDataType, _>(
                        p_val,
                        storage_provider,
                        &mut rng.create_rnd(),
                    )?;

                let quantizer_params = if let Some(std_dev) = standard_deviation {
                    ScalarQuantizationParameters::new(std_dev)
                } else {
                    ScalarQuantizationParameters::default()
                };

                let quantizer = quantizer_params.train(
                    MatrixView::try_from(&train_data_vector, train_size, train_dim).bridge_err()?,
                );

                info!("Now quantizer is trained and saving to file");
                let sq_storage = SQStorage::new(index_path_prefix);
                sq_storage.save_quantizer(&quantizer, storage_provider)?;

                Ok(Self::Scalar1Bit(WithBits::<1>::new(quantizer)))
            }
            QuantizationType::TQ {
                nbits,
                seed,
                use_hadamard,
            } => {
                let dim = index_configuration.dim;
                let mut rng = StdRng::seed_from_u64(seed);
                let quantizer = if use_hadamard {
                    TurboQuantQuantizer::new_hadamard(dim, nbits, &mut rng)
                } else {
                    TurboQuantQuantizer::new(dim, nbits, &mut rng)
                };
                info!(
                    "TurboQuant quantizer created: {}bit, dim={}, hadamard={}, seed={}",
                    nbits, dim, use_hadamard, seed
                );
                Ok(Self::TurboQuant(WithTurboQuant::new(quantizer)))
            }
        }
    }

    /// Load a previously trained quantizer from storage.
    pub fn load<StorageProvider>(
        build_quantization_type: &QuantizationType,
        index_path_prefix: &str,
        storage_provider: &StorageProvider,
    ) -> ANNResult<Self>
    where
        StorageProvider: StorageReadProvider,
    {
        match build_quantization_type {
            QuantizationType::FP => Ok(Self::NoQuant(NoStore)),
            QuantizationType::PQ { num_chunks } => {
                let pq_pivots_paths = PQPathNames::new(index_path_prefix);
                let pq_build_storage = PQStorage::new(
                    &pq_pivots_paths.pivots,
                    &pq_pivots_paths.compressed_data,
                    None,
                );
                let table = pq_build_storage.load_pq_pivots_bin::<StorageProvider>(
                    &pq_pivots_paths.pivots,
                    *num_chunks,
                    storage_provider,
                )?;
                Ok(Self::PQ(table))
            }
            QuantizationType::SQ { .. } => {
                let sq_storage = SQStorage::new(index_path_prefix);
                let sq_quantizer = sq_storage.load_quantizer(storage_provider)?;
                Ok(Self::Scalar1Bit(WithBits::<1>::new(sq_quantizer)))
            }
            QuantizationType::TQ {
                nbits,
                seed,
                use_hadamard,
            } => {
                // TQ is data-oblivious: recreate from seed (no training data needed).
                let dim = 0; // Will be set when index metadata is loaded
                // For load, we need the dim from config. But dim isn't available here.
                // The disk index build path recreates the quantizer during build, not load.
                // For now, error on load since TQ disk index load is not yet supported.
                let _ = (nbits, seed, use_hadamard, dim);
                Err(ANNError::log_index_config_error(
                    "build_quantization_type".to_string(),
                    "TurboQuant load is not yet supported for disk index".to_string(),
                ))
            }
        }
    }
}
