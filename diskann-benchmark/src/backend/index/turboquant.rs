/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::registry::Benchmarks;

crate::utils::stub_impl!("turboquant-quantization", inputs::async_::IndexTQOperation);

pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    #[cfg(feature = "turboquant-quantization")]
    {
        use half::f16;
        use crate::backend::index::benchmarks::register;

        register!(
            benchmarks,
            "async-tq-4-bit-f16",
            imp::TurboQuantized<'static, f16>
        );
        register!(
            benchmarks,
            "async-tq-4-bit-f32",
            imp::TurboQuantized<'static, f32>
        );
    }

    #[cfg(not(feature = "turboquant-quantization"))]
    imp::register("async-tq", benchmarks);
}

#[cfg(feature = "turboquant-quantization")]
mod imp {
    use std::{io::Write, sync::Arc};

    use anyhow::Context;
    use diskann_benchmark_runner::{
        describeln,
        dispatcher::{self, DispatchRule, FailureScore, MatchScore},
        utils::{datatype, MicroSeconds},
        Any, Checkpoint, Output,
    };
    use diskann_providers::{
        index::diskann_async::{self},
        model::{
            configuration::IndexConfiguration,
            graph::provider::async_::{common, inmem},
        },
    };
    use diskann_utils::views::{Matrix, MatrixView};
    use half::f16;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::{
        backend::index::{
            benchmarks::{run_build, run_search_outer, BuildAndSearch, FullPrecision},
            build::{self, load_index, only_single_insert, save_index, BuildStats},
            result::QuantBuildResult,
        },
        inputs::async_::{IndexTQOperation, IndexSource},
        utils::{self, datafiles},
    };

    pub(super) struct TurboQuantized<'a, T> {
        input: &'a IndexTQOperation,
        _type: std::marker::PhantomData<T>,
    }

    impl<'a, T> TurboQuantized<'a, T> {
        fn new(input: &'a IndexTQOperation) -> Self {
            Self {
                input,
                _type: std::marker::PhantomData,
            }
        }
    }

    impl<T> dispatcher::Map for TurboQuantized<'static, T>
    where
        T: 'static,
    {
        type Type<'a> = TurboQuantized<'a, T>;
    }

    impl<'a, T> DispatchRule<&'a IndexTQOperation> for TurboQuantized<'a, T>
    where
        datatype::Type<T>: DispatchRule<datatype::DataType>,
    {
        type Error = std::convert::Infallible;

        fn try_match(from: &&'a IndexTQOperation) -> Result<MatchScore, FailureScore> {
            let mut failure_score: Option<u32> = None;
            match from.index_operation.source {
                IndexSource::Load(_) => {}
                IndexSource::Build(ref build) => {
                    if build.multi_insert.is_some() {
                        failure_score = Some(1);
                    }
                }
            }

            if let Err(FailureScore(_)) =
                FullPrecision::<'a, T>::try_match(&&from.index_operation)
            {
                *failure_score.get_or_insert(0) += 1;
            }

            match failure_score {
                None => Ok(MatchScore(0)),
                Some(score) => Err(FailureScore(score)),
            }
        }

        fn convert(from: &'a IndexTQOperation) -> Result<Self, Self::Error> {
            Ok(Self::new(from))
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a IndexTQOperation>,
        ) -> std::fmt::Result {
            match from {
                None => {
                    describeln!(f, "- TurboQuant Index Build and Search")?;
                    describeln!(
                        f,
                        "- Requires `{}` data",
                        dispatcher::Description::<datatype::DataType, datatype::Type<T>>::new(),
                    )?;
                }
                Some(input) => {
                    let mut check_match = |data_type: &datatype::DataType| {
                        if datatype::Type::<T>::try_match(data_type).is_err() {
                            describeln!(
                                f,
                                "- Only `{}` data type supported, got {}",
                                dispatcher::Description::<datatype::DataType, datatype::Type<T>>::new(),
                                data_type
                            )
                            .unwrap();
                        }
                    };
                    match &input.index_operation.source {
                        IndexSource::Load(load) => check_match(&load.data_type),
                        IndexSource::Build(build) => {
                            check_match(&build.data_type);
                            if build.multi_insert.is_some() {
                                describeln!(f, "- TurboQuant does not support multi-insert")?;
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    }

    impl<'a, T> DispatchRule<&'a Any> for TurboQuantized<'a, T>
    where
        datatype::Type<T>: DispatchRule<datatype::DataType>,
    {
        type Error = anyhow::Error;

        fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
            from.try_match::<IndexTQOperation, Self>()
        }

        fn convert(from: &'a Any) -> Result<Self, Self::Error> {
            from.convert::<IndexTQOperation, Self>()
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a Any>,
        ) -> std::fmt::Result {
            Any::description::<IndexTQOperation, Self>(f, from, IndexTQOperation::tag())
        }
    }

    macro_rules! impl_tq_build {
        ($T:ty) => {
            impl<'a> BuildAndSearch<'a> for TurboQuantized<'a, $T> {
                type Data = QuantBuildResult;
                fn run(
                    self,
                    checkpoint: Checkpoint<'_>,
                    mut output: &mut dyn Output,
                ) -> Result<Self::Data, anyhow::Error> {
                    writeln!(output, "{}", self.input)?;

                    let IndexSource::Build(build) = &self.input.index_operation.source else {
                        anyhow::bail!("TurboQuant load not yet supported");
                    };

                    let data: Arc<Matrix<$T>> = Arc::new(
                        datafiles::load_dataset(datafiles::BinFile(&build.data))?,
                    );

                    let start = std::time::Instant::now();
                    let dim = data.ncols();
                    let nbits = self.input.nbits;
                    let seed = self.input.seed;
                    let mut rng = StdRng::seed_from_u64(seed);
                    let quantizer = if self.input.use_hadamard {
                        diskann_quantization::turboquant::TurboQuantQuantizer::new_hadamard(
                            dim, nbits, &mut rng,
                        )
                    } else {
                        diskann_quantization::turboquant::TurboQuantQuantizer::new(
                            dim, nbits, &mut rng,
                        )
                    };
                    let quant_training_time: MicroSeconds = start.elapsed().into();

                    // Build index with FP + TQ (same setup as SQ benchmark)
                    let create_index = |data_view: MatrixView<$T>| {
                        let index = diskann_async::new_quant_index::<$T, _, _>(
                            self.input.try_as_config()?.build()?,
                            self.input.inmem_parameters(
                                data_view.nrows(), data_view.ncols(),
                            )?,
                            inmem::WithTurboQuant::new(quantizer),
                            common::NoDeletes,
                        )?;
                        build::set_start_points(
                            index.provider(), data_view, build.start_point_strategy,
                        )?;
                        Ok(index)
                    };
                    let (index, build_stats) = if self.input.use_fp_for_build {
                        run_build(
                            &build, common::FullPrecision, None, output,
                            create_index, only_single_insert,
                        )?
                    } else {
                        run_build(
                            &build, common::Quantized, None, output,
                            create_index, only_single_insert,
                        )?
                    };

                    let build = if self.input.use_fp_for_search {
                        run_search_outer(
                            &self.input.index_operation.search_phase,
                            common::FullPrecision,
                            index,
                            Some(build_stats),
                            checkpoint,
                        )?
                    } else {
                        run_search_outer(
                            &self.input.index_operation.search_phase,
                            common::Quantized,
                            index,
                            Some(build_stats),
                            checkpoint,
                        )?
                    };

                    let result = QuantBuildResult {
                        quant_training_time,
                        build,
                    };

                    writeln!(output, "\n\n{}", result)?;
                    Ok(result)
                }
            }
        };
    }

    impl_tq_build!(f32);
    impl_tq_build!(f16);
}
