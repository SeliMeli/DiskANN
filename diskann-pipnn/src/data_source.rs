/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Abstraction over vector data storage — in-memory slice or memory-mapped file.
//!
//! `VectorDataSource` provides indexed access to vectors by point ID,
//! enabling PiPNN to build graphs from data that may not fit in memory.

use diskann::utils::VectorRepr;

/// Indexed access to vector data by point ID.
///
/// Implementations must be `Send + Sync` for use in rayon parallel iterators.
/// The `get()` method returns a slice of `ndims` elements for the given point.
pub trait VectorDataSource: Send + Sync {
    type Elem: VectorRepr + Send + Sync;

    /// Return the vector for point `idx`: ndims elements.
    /// Panics if `idx >= npoints()`.
    fn get(&self, idx: usize) -> &[Self::Elem];

    fn npoints(&self) -> usize;
    fn ndims(&self) -> usize;
}

/// Zero-cost wrapper around a contiguous `&[T]` slice (row-major, npoints × ndims).
pub struct SliceDataSource<'a, T> {
    data: &'a [T],
    npoints: usize,
    ndims: usize,
}

impl<'a, T: VectorRepr + Send + Sync> SliceDataSource<'a, T> {
    pub fn new(data: &'a [T], npoints: usize, ndims: usize) -> Self {
        debug_assert_eq!(data.len(), npoints * ndims);
        Self { data, npoints, ndims }
    }
}

impl<T: VectorRepr + Send + Sync> VectorDataSource for SliceDataSource<'_, T> {
    type Elem = T;

    #[inline(always)]
    fn get(&self, idx: usize) -> &[T] {
        debug_assert!(idx < self.npoints, "index {} out of range (npoints={})", idx, self.npoints);
        &self.data[idx * self.ndims..(idx + 1) * self.ndims]
    }

    #[inline(always)]
    fn npoints(&self) -> usize { self.npoints }

    #[inline(always)]
    fn ndims(&self) -> usize { self.ndims }
}

/// Memory-mapped vector data from a DiskANN .bin file.
///
/// Header: 4 bytes npoints (u32) + 4 bytes ndims (u32), then npoints × ndims × sizeof(T) data.
/// OS pages in/out on demand — only active pages consume RSS.
pub struct MmapDataSource<T: VectorRepr + Send + Sync + bytemuck::Pod> {
    mmap: std::sync::Arc<memmap2::Mmap>,
    /// Byte offset into the mmap where this source's data starts.
    data_offset: usize,
    npoints: usize,
    ndims: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: VectorRepr + Send + Sync + bytemuck::Pod> MmapDataSource<T> {
    /// Open a DiskANN .bin file. Optionally restrict to a window of
    /// `count` points starting at `point_offset` for sharded access.
    pub fn open(
        path: &std::path::Path,
        point_offset: usize,
        count: Option<usize>,
    ) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file)? });

        let header = &mmap[..8];
        let total_npoints = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let ndims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

        let npoints = count
            .unwrap_or(total_npoints.saturating_sub(point_offset))
            .min(total_npoints.saturating_sub(point_offset));

        let elem_size = std::mem::size_of::<T>();
        let data_offset = 8 + point_offset * ndims * elem_size;

        Ok(Self { mmap, data_offset, npoints, ndims, _phantom: std::marker::PhantomData })
    }

    /// Open the full file (no windowing).
    pub fn open_full(path: &std::path::Path) -> std::io::Result<Self> {
        Self::open(path, 0, None)
    }
}

impl<T: VectorRepr + Send + Sync + bytemuck::Pod> VectorDataSource for MmapDataSource<T> {
    type Elem = T;

    #[inline]
    fn get(&self, idx: usize) -> &[T] {
        debug_assert!(idx < self.npoints, "index {} out of range (npoints={})", idx, self.npoints);
        let elem_size = std::mem::size_of::<T>();
        let byte_start = self.data_offset + idx * self.ndims * elem_size;
        let byte_end = byte_start + self.ndims * elem_size;
        bytemuck::cast_slice(&self.mmap[byte_start..byte_end])
    }

    #[inline]
    fn npoints(&self) -> usize { self.npoints }

    #[inline]
    fn ndims(&self) -> usize { self.ndims }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_data_source() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src = SliceDataSource::new(&data, 2, 3);
        assert_eq!(src.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(src.get(1), &[4.0, 5.0, 6.0]);
        assert_eq!(src.npoints(), 2);
        assert_eq!(src.ndims(), 3);
    }

    #[test]
    fn test_slice_data_source_f16() {
        use half::f16;
        let data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter().map(f16::from_f32).collect();
        let src = SliceDataSource::new(&data, 2, 2);
        assert_eq!(src.get(0).len(), 2);
        assert_eq!(src.get(1).len(), 2);
    }

    #[test]
    fn test_mmap_data_source() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("pipnn_mmap_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.fbin");

        // Write a DiskANN .bin file: 3 points × 2 dims, f32
        let npoints: u32 = 3;
        let ndims: u32 = 2;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&npoints.to_le_bytes()).unwrap();
            f.write_all(&ndims.to_le_bytes()).unwrap();
            f.write_all(bytemuck::cast_slice(&data)).unwrap();
        }

        let src = MmapDataSource::<f32>::open_full(&path).unwrap();
        assert_eq!(src.npoints(), 3);
        assert_eq!(src.ndims(), 2);
        assert_eq!(src.get(0), &[1.0, 2.0]);
        assert_eq!(src.get(1), &[3.0, 4.0]);
        assert_eq!(src.get(2), &[5.0, 6.0]);

        // Windowed: skip first point, take 2
        let src2 = MmapDataSource::<f32>::open(&path, 1, Some(2)).unwrap();
        assert_eq!(src2.npoints(), 2);
        assert_eq!(src2.get(0), &[3.0, 4.0]);
        assert_eq!(src2.get(1), &[5.0, 6.0]);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
