/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;

use diskann::utils::VectorRepr;
use diskann_providers::storage::StorageReadProvider;
use diskann_utils::io::Metadata;

use crate::{PiPNNError, PiPNNResult};

pub(crate) trait PointSource: Sync {
    fn copy_points_into(&self, indices: &[usize], out: &mut [f32]) -> PiPNNResult<()>;
}

pub(crate) struct FilePointSource<'a, T, SP>
where
    T: VectorRepr,
    SP: StorageReadProvider,
{
    data_path: &'a str,
    storage_provider: &'a SP,
    npoints: usize,
    ndims: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T, SP> FilePointSource<'a, T, SP>
where
    T: VectorRepr,
    SP: StorageReadProvider,
{
    pub(crate) fn new(
        data_path: &'a str,
        storage_provider: &'a SP,
        npoints: usize,
        ndims: usize,
    ) -> Self {
        Self {
            data_path,
            storage_provider,
            npoints,
            ndims,
            _phantom: PhantomData,
        }
    }
}

impl<T, SP> PointSource for FilePointSource<'_, T, SP>
where
    T: VectorRepr,
    SP: StorageReadProvider,
{
    fn copy_points_into(&self, indices: &[usize], out: &mut [f32]) -> PiPNNResult<()> {
        let expected_len = indices
            .len()
            .checked_mul(self.ndims)
            .ok_or_else(|| PiPNNError::Config("point read size overflow".into()))?;
        if out.len() != expected_len {
            return Err(PiPNNError::DataLengthMismatch {
                expected: expected_len,
                actual: out.len(),
                npoints: indices.len(),
                ndims: self.ndims,
            });
        }

        let mut reader = self.storage_provider.open_reader(self.data_path)?;
        let metadata = Metadata::read(&mut reader)?;
        if metadata.npoints() != self.npoints || metadata.ndims() != self.ndims {
            return Err(PiPNNError::Config(format!(
                "dataset metadata changed while reading {}: expected {}x{}, found {}x{}",
                self.data_path,
                self.npoints,
                self.ndims,
                metadata.npoints(),
                metadata.ndims()
            )));
        }

        let row_bytes = self
            .ndims
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| PiPNNError::Config("row byte size overflow".into()))?;
        let mut typed_row = vec![<T as bytemuck::Zeroable>::zeroed(); self.ndims];

        for (row_idx, &point_idx) in indices.iter().enumerate() {
            if point_idx >= self.npoints {
                return Err(PiPNNError::Config(format!(
                    "point index {} out of bounds for {} points",
                    point_idx, self.npoints
                )));
            }

            let offset = 8u64
                .checked_add(
                    (point_idx as u64)
                        .checked_mul(row_bytes as u64)
                        .ok_or_else(|| PiPNNError::Config("point offset overflow".into()))?,
                )
                .ok_or_else(|| PiPNNError::Config("point offset overflow".into()))?;
            reader.seek(SeekFrom::Start(offset))?;
            reader.read_exact(bytemuck::must_cast_slice_mut::<T, u8>(&mut typed_row))?;

            let dst = &mut out[row_idx * self.ndims..(row_idx + 1) * self.ndims];
            T::as_f32_into(&typed_row, dst).expect("f32 conversion");
        }

        Ok(())
    }
}
