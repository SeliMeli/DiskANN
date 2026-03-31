/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! TurboQuant vector store for DiskANN.
//!
//! This module provides a complete vector store implementation using TurboQuant quantization
//! (arXiv:2504.19874). Unlike scalar quantization, TurboQuant applies a random orthogonal
//! rotation before quantizing each coordinate independently using Lloyd-Max optimal centroids
//! for N(0,1). This is data-oblivious: no training pass over the dataset is needed.
//!
//! # Architecture
//!
//! - **[`TQStore`]**: Stores encoded vectors (codes + norm + quant_norm_sq per vector).
//! - **[`WithTurboQuant`]**: Factory implementing [`CreateVectorStore`].
//! - **[`TQDistanceComputer`]**: Quantized-to-quantized distances for graph pruning.
//! - **[`TQQueryComputer`]**: Query-to-quantized distances for search. Rotates the query once
//!   at construction, then computes O(d) distances per candidate via centroid lookups.
//! - **[`TQAccessor`]**: Accessor for graph traversal with prefetching support.
//!
//! # Distance computation (L2 squared)
//!
//! ```text
//! L2²(q, x̃) = ||q'||² - 2 * (norm / sqrt(d)) * Σⱼ q'[j] * centroid[code[j]] + quant_norm_sq
//! ```
//!
//! where `q' = Π·q` is the rotated query (precomputed once).

use std::{future::Future, sync::Mutex};

use diskann::{
    ANNError, ANNResult,
    graph::glue::{
        self, ExpandBeam, FillSet, FilterStartPoints, InsertStrategy, PruneStrategy, SearchExt,
        SearchStrategy,
    },
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DelegateNeighbor, ExecutionContext,
        HasId,
    },
    utils::{IntoUsize, VectorRepr},
};
use diskann_quantization::turboquant::{
    LloydMaxCentroids, TurboQuantQuantizer,
    TurboQuantMeta,
};
use diskann_utils::{Reborrow, future::AsyncFriendly};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use super::{DefaultProvider, GetFullPrecision, Rerank};
use crate::{
    common::IgnoreLockPoison,
    model::graph::{
        provider::async_::{
            FastMemoryVectorProviderAsync, SimpleNeighborProviderAsync,
            common::{
                AlignedMemoryVectorStore, CreateVectorStore, NoStore, Quantized, SetElementHelper,
                TestCallCount, VectorStore,
            },
            inmem::{FullPrecisionProvider, FullPrecisionStore},
            postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
        },
        traits::AdHoc,
    },
    storage::{self, AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith,
              StorageReadProvider, StorageWriteProvider},
};

/// A lightweight reference into the TQStore's flat buffers.
///
/// This is the TurboQuant equivalent of `CVRef` in the SQ implementation.
/// It borrows the codes slice and the per-vector metadata without copying.
#[derive(Debug, Clone, Copy)]
pub struct TQRef<'a> {
    pub codes: &'a [u8],
    pub meta: TurboQuantMeta,
}

/// `Reborrow` allows shortening the lifetime of a `TQRef`.
///
/// Since `TQRef` is `Copy` and covariant in `'a`, reborrowing simply
/// re-borrows the codes slice with a shorter lifetime.
impl<'short, 'a> Reborrow<'short> for TQRef<'a> {
    type Target = TQRef<'short>;

    fn reborrow(&'short self) -> Self::Target {
        TQRef {
            codes: self.codes,
            meta: self.meta,
        }
    }
}

// ---------------------------------------------------------------------------
// Storage layout
// ---------------------------------------------------------------------------
//
// Each vector occupies `dim + 8` bytes in the AlignedMemoryVectorStore<u8>:
//   [code_0, code_1, ..., code_{d-1}, norm_bytes[0..4], quant_norm_sq_bytes[0..4]]
//
// The 8 trailing bytes are two little-endian f32 values (norm, quant_norm_sq).

/// Whether this bit-width supports fast packed storage (divides 8 evenly).
#[inline]
fn is_packable(nbits: usize) -> bool {
    nbits == 1 || nbits == 2 || nbits == 4 || nbits == 8
}

/// Bytes needed for packed codes at given bit-width.
#[inline]
fn packed_code_bytes(dim: usize, nbits: usize) -> usize {
    if is_packable(nbits) {
        (dim * nbits + 7) / 8
    } else {
        // Non-power-of-2 bit widths: 1 byte per code
        dim
    }
}

/// Bytes needed per encoded vector in the flat store.
#[inline]
fn bytes_per_vector(dim: usize, nbits: usize) -> usize {
    packed_code_bytes(dim, nbits) + 8 // packed codes + 4 bytes (norm) + 4 bytes (quant_norm_sq)
}

/// Extract a [`TQRef`] from a raw slice obtained from the aligned store.
#[inline]
fn tqref_from_slice(slice: &[u8], dim: usize, nbits: usize) -> TQRef<'_> {
    let code_bytes = packed_code_bytes(dim, nbits);
    debug_assert!(
        slice.len() >= code_bytes + 8,
        "slice length {} < required {}",
        slice.len(),
        code_bytes + 8
    );
    let codes = &slice[..code_bytes];
    let norm = f32::from_le_bytes([
        slice[code_bytes],
        slice[code_bytes + 1],
        slice[code_bytes + 2],
        slice[code_bytes + 3],
    ]);
    let quant_norm_sq = f32::from_le_bytes([
        slice[code_bytes + 4],
        slice[code_bytes + 5],
        slice[code_bytes + 6],
        slice[code_bytes + 7],
    ]);
    TQRef {
        codes,
        meta: TurboQuantMeta { norm, quant_norm_sq },
    }
}

/// Pack codes into a byte slice.
/// For nbits ∈ {1,2,4,8}: pack multiple codes per byte.
/// For other nbits (e.g., 3,5,6,7): store 1 code per byte (no packing).
#[inline]
fn pack_codes(codes: &[u8], nbits: usize) -> Vec<u8> {
    if !is_packable(nbits) {
        return codes.to_vec();
    }
    let codes_per_byte = 8 / nbits;
    let mask = (1u8 << nbits) - 1;
    let packed_len = (codes.len() + codes_per_byte - 1) / codes_per_byte;
    let mut packed = vec![0u8; packed_len];
    for (i, &code) in codes.iter().enumerate() {
        let byte_idx = i / codes_per_byte;
        let bit_offset = (i % codes_per_byte) * nbits;
        packed[byte_idx] |= (code & mask) << bit_offset;
    }
    packed
}

/// Extract code at position `idx` from packed bytes.
#[inline(always)]
fn extract_code(packed: &[u8], idx: usize, nbits: usize) -> u8 {
    if !is_packable(nbits) {
        return packed[idx];
    }
    let codes_per_byte = 8 / nbits;
    let mask = (1u8 << nbits) - 1;
    let byte_idx = idx / codes_per_byte;
    let bit_offset = (idx % codes_per_byte) * nbits;
    (packed[byte_idx] >> bit_offset) & mask
}

/// Write encoded data into a mutable raw slice with packed codes.
#[inline]
fn write_encoded(slice: &mut [u8], dim: usize, nbits: usize, codes: &[u8], norm: f32, quant_norm_sq: f32) {
    let packed = pack_codes(codes, nbits);
    let code_bytes = packed.len();
    slice[..code_bytes].copy_from_slice(&packed);
    slice[code_bytes..code_bytes + 4].copy_from_slice(&norm.to_le_bytes());
    slice[code_bytes + 4..code_bytes + 8].copy_from_slice(&quant_norm_sq.to_le_bytes());
}

// ---------------------------------------------------------------------------
// WithTurboQuant - factory
// ---------------------------------------------------------------------------

/// Factory for creating a [`TQStore`].
///
/// Analogous to [`super::WithBits`] for scalar quantization.
#[derive(Clone)]
pub struct WithTurboQuant {
    quantizer: TurboQuantQuantizer,
}

impl WithTurboQuant {
    pub fn new(quantizer: TurboQuantQuantizer) -> Self {
        Self { quantizer }
    }

    pub fn quantizer(&self) -> &TurboQuantQuantizer {
        &self.quantizer
    }
}

// ---------------------------------------------------------------------------
// TQStore
// ---------------------------------------------------------------------------

/// This controls how many vectors share a write lock.
const WRITE_LOCK_GRANULARITY: usize = 16;

/// The default prefetch lookahead to use if not configured externally.
const PREFETCH_DEFAULT: usize = 8;

/// In-memory vector store for TurboQuant-encoded vectors.
///
/// Stores flat `[codes | norm | quant_norm_sq]` records in an aligned buffer
/// and holds the shared [`TurboQuantQuantizer`] used for encoding and distance
/// computation.
pub struct TQStore {
    data: AlignedMemoryVectorStore<u8>,
    quantizer: TurboQuantQuantizer,
    metric: Metric,

    // Write-only locks (reads are unsynchronized for speed).
    // sync::Mutex is fine because the lock is never held across an await.
    write_locks: Vec<Mutex<()>>,

    /// Prefetching lookahead for bulk operations.
    prefetch_lookahead: usize,

    num_get_calls: TestCallCount,
}

impl TQStore {
    pub(super) fn new(
        quantizer: TurboQuantQuantizer,
        num_vectors: usize,
        metric: Metric,
        prefetch_lookahead: Option<usize>,
    ) -> Self {
        let write_locks = (0..num_vectors.div_ceil(WRITE_LOCK_GRANULARITY))
            .map(|_| Mutex::new(()))
            .collect::<Vec<_>>();
        let bpv = bytes_per_vector(quantizer.dim(), quantizer.nbits());
        Self {
            data: AlignedMemoryVectorStore::with_capacity(num_vectors, bpv),
            quantizer,
            metric,
            write_locks,
            num_get_calls: TestCallCount::default(),
            prefetch_lookahead: prefetch_lookahead.unwrap_or(PREFETCH_DEFAULT),
        }
    }

    /// Prefetch the first few cache lines of vector `i`.
    pub(crate) fn prefetch_hint(&self, i: usize) {
        // SAFETY: Racing is acceptable; this is an architectural prefetch hint.
        let data = unsafe { self.data.get_slice(i) };
        diskann_vector::prefetch_hint_max::<4, _>(data);
    }

    pub(super) fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    pub(super) fn get_vector(&self, i: usize) -> Result<TQRef<'_>, TQError> {
        self.num_get_calls.increment();
        let slice = unsafe { self.data.get_slice(i) };
        Ok(tqref_from_slice(slice, self.quantizer.dim(), self.quantizer.nbits()))
    }

    pub(super) fn set_vector<T>(&self, i: usize, v: &[T]) -> Result<(), TQError>
    where
        T: VectorRepr,
    {
        let vf32: &[f32] =
            &T::as_f32(v).map_err(|e| TQError::FullPrecisionConversionErr(format!("{:?}", e)))?;

        debug_assert!(
            vf32.len() == self.dim(),
            "vector f32 dimension {} does not match dimension {}",
            vf32.len(),
            self.dim()
        );

        let encoded = self.quantizer.encode(vf32);

        let lock_id = i / WRITE_LOCK_GRANULARITY;
        let _guard = self.write_locks[lock_id].lock_or_panic();

        let slice = unsafe { self.data.get_mut_slice(i) };
        write_encoded(
            slice,
            self.dim(),
            self.quantizer.nbits(),
            &encoded.codes,
            encoded.norm,
            encoded.quant_norm_sq,
        );
        Ok(())
    }

    /// Store raw encoded bytes at position `i` (for deserialization).
    ///
    /// # Safety
    ///
    /// Writers are mutex-protected but readers may race. Caller must handle that.
    pub(crate) unsafe fn set_raw_vector(&self, i: usize, v: &[u8]) -> ANNResult<()> {
        let expected_len = bytes_per_vector(self.dim(), self.quantizer.nbits());
        debug_assert!(
            v.len() == expected_len,
            "raw vector length {} does not match expected {}",
            v.len(),
            expected_len
        );

        let lock_id = i / WRITE_LOCK_GRANULARITY;
        let _guard = self.write_locks[lock_id].lock_or_panic();
        unsafe { self.data.get_mut_slice(i) }.copy_from_slice(v);
        Ok(())
    }

    // Return a distance computer for quantized-to-quantized distance.
    pub(super) fn distance_computer(&self) -> Result<TQDistanceComputer, TQError> {
        match self.metric {
            Metric::L2 => Ok(TQDistanceComputer {
                centroids: self.quantizer.centroids().clone(),
                inv_sqrt_dim: 1.0 / (self.dim() as f32).sqrt(),
                dim: self.dim(),
                nbits: self.quantizer.nbits(),
            }),
            unsupported => Err(TQError::UnsupportedDistanceMetric(unsupported)),
        }
    }

    pub(super) fn query_computer<T>(
        &self,
        query: &[T],
        _allow_rescale: bool,
    ) -> Result<TQQueryComputer, TQError>
    where
        T: VectorRepr,
    {
        let q = T::as_f32(query)
            .map_err(|e| TQError::FullPrecisionConversionErr(format!("{:?}", e)))?;

        // Rotate query ONCE.
        let rotated_query = self.quantizer.rotate_query(q.as_ref());
        let query_norm_sq: f32 = rotated_query.iter().map(|v| v * v).sum();

        Ok(TQQueryComputer {
            rotated_query,
            query_norm_sq,
            centroids: self.quantizer.centroids().clone(),
            inv_sqrt_dim: 1.0 / (self.dim() as f32).sqrt(),
            nbits: self.quantizer.nbits(),
            dim: self.dim(),
        })
    }

    pub fn prefetch_lookahead(&self) -> usize {
        self.prefetch_lookahead
    }
}

// ---------------------------------------------------------------------------
// TQDistanceComputer - quantized-to-quantized distance
// ---------------------------------------------------------------------------

/// Computes L2 squared distance between two TurboQuant-encoded vectors.
///
/// Used during graph pruning. O(d) per pair -- no matrix operations.
///
/// ```text
/// ||x̃₁ - x̃₂||² = Σⱼ (c[code₁[j]] * s₁ - c[code₂[j]] * s₂)²
/// ```
/// where `sᵢ = normᵢ / sqrt(d)`.
#[derive(Debug)]
pub struct TQDistanceComputer {
    centroids: LloydMaxCentroids,
    inv_sqrt_dim: f32,
    dim: usize,
    nbits: usize,
}

impl<'a, 'b> DistanceFunction<TQRef<'a>, TQRef<'b>, f32> for TQDistanceComputer {
    #[inline(always)]
    fn evaluate_similarity(&self, left: TQRef<'a>, right: TQRef<'b>) -> f32 {
        let scale_ab = self.inv_sqrt_dim * self.inv_sqrt_dim * left.meta.norm * right.meta.norm;
        let centroids = self.centroids.centroids();

        let ip = if self.nbits == 4 {
            ip_qq_4bit_packed(left.codes, right.codes, centroids, self.dim)
        } else if self.nbits == 2 {
            ip_qq_2bit_packed(left.codes, right.codes, centroids, self.dim)
        } else {
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                let ca = extract_code(left.codes, j, self.nbits) as usize;
                let cb = extract_code(right.codes, j, self.nbits) as usize;
                unsafe { sum += *centroids.get_unchecked(ca) * *centroids.get_unchecked(cb); }
            }
            sum
        };

        (left.meta.quant_norm_sq + right.meta.quant_norm_sq - 2.0 * ip * scale_ab).max(0.0)
    }
}

/// Fast 4-bit quantized-quantized inner product.
#[inline(always)]
fn ip_qq_4bit_packed(a: &[u8], b: &[u8], centroids: &[f32], dim: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let nbytes = dim / 2;
    for i in 0..nbytes {
        unsafe {
            let ba = *a.get_unchecked(i);
            let bb = *b.get_unchecked(i);
            let ca_lo = *centroids.get_unchecked((ba & 0x0F) as usize);
            let cb_lo = *centroids.get_unchecked((bb & 0x0F) as usize);
            let ca_hi = *centroids.get_unchecked((ba >> 4) as usize);
            let cb_hi = *centroids.get_unchecked((bb >> 4) as usize);
            sum0 += ca_lo * cb_lo;
            sum1 += ca_hi * cb_hi;
        }
    }
    sum0 + sum1
}

/// Fast 2-bit quantized-quantized inner product.
#[inline(always)]
fn ip_qq_2bit_packed(a: &[u8], b: &[u8], centroids: &[f32], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    let nbytes = dim / 4;
    for i in 0..nbytes {
        unsafe {
            let ba = *a.get_unchecked(i);
            let bb = *b.get_unchecked(i);
            for shift in [0u8, 2, 4, 6] {
                let ca = *centroids.get_unchecked(((ba >> shift) & 0x03) as usize);
                let cb = *centroids.get_unchecked(((bb >> shift) & 0x03) as usize);
                sum += ca * cb;
            }
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// TQQueryComputer - preprocessed query-to-quantized distance
// ---------------------------------------------------------------------------

/// Computes L2 squared distance between a (rotated) query and encoded candidates.
///
/// The query rotation `q' = Π·q` is done once at construction. Each
/// `evaluate_similarity` call is O(d) via centroid lookups:
///
/// ```text
/// L2²(q, x̃) = ||q'||² - 2 * (norm/√d) * Σⱼ q'[j] * centroid[code[j]] + quant_norm_sq
/// ```
pub struct TQQueryComputer {
    /// Rotated query: q' = Π·q, computed once.
    rotated_query: Vec<f32>,
    /// ||q'||² = ||q||² (rotation preserves norms).
    query_norm_sq: f32,
    /// Shared centroids.
    centroids: LloydMaxCentroids,
    /// 1 / sqrt(dim).
    inv_sqrt_dim: f32,
    /// Number of bits per code.
    nbits: usize,
    /// Vector dimension.
    dim: usize,
}

impl<'a> PreprocessedDistanceFunction<TQRef<'a>, f32> for TQQueryComputer {
    #[inline(always)]
    fn evaluate_similarity(&self, changing: TQRef<'a>) -> f32 {
        let scale = self.inv_sqrt_dim * changing.meta.norm;
        let codes = changing.codes;
        let query = &self.rotated_query;
        let centroids = self.centroids.centroids();

        let ip_sum = if self.nbits == 4 {
            // Fast path for 4-bit: 2 codes per byte, direct nibble extraction
            ip_4bit_packed(query, codes, centroids, self.dim)
        } else if self.nbits == 2 {
            // Fast path for 2-bit: 4 codes per byte
            ip_2bit_packed(query, codes, centroids, self.dim)
        } else {
            // General path
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                let c = extract_code(codes, j, self.nbits) as usize;
                unsafe { sum += *query.get_unchecked(j) * *centroids.get_unchecked(c); }
            }
            sum
        };

        (self.query_norm_sq - 2.0 * ip_sum * scale + changing.meta.quant_norm_sq).max(0.0)
    }
}

/// Fast 4-bit inner product: processes 2 codes per byte.
#[inline(always)]
fn ip_4bit_packed(query: &[f32], packed: &[u8], centroids: &[f32], dim: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let nbytes = dim / 2;
    for i in 0..nbytes {
        let byte = unsafe { *packed.get_unchecked(i) };
        let c_lo = (byte & 0x0F) as usize;
        let c_hi = (byte >> 4) as usize;
        let qi = i * 2;
        unsafe {
            sum0 += *query.get_unchecked(qi) * *centroids.get_unchecked(c_lo);
            sum1 += *query.get_unchecked(qi + 1) * *centroids.get_unchecked(c_hi);
        }
    }
    // Handle odd dimension
    if dim % 2 == 1 {
        let byte = unsafe { *packed.get_unchecked(nbytes) };
        let c = (byte & 0x0F) as usize;
        unsafe { sum0 += *query.get_unchecked(dim - 1) * *centroids.get_unchecked(c); }
    }
    sum0 + sum1
}

/// Fast 2-bit inner product: processes 4 codes per byte.
#[inline(always)]
fn ip_2bit_packed(query: &[f32], packed: &[u8], centroids: &[f32], dim: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let nbytes = dim / 4;
    for i in 0..nbytes {
        let byte = unsafe { *packed.get_unchecked(i) };
        let c0 = (byte & 0x03) as usize;
        let c1 = ((byte >> 2) & 0x03) as usize;
        let c2 = ((byte >> 4) & 0x03) as usize;
        let c3 = (byte >> 6) as usize;
        let qi = i * 4;
        unsafe {
            sum0 += *query.get_unchecked(qi) * *centroids.get_unchecked(c0)
                + *query.get_unchecked(qi + 2) * *centroids.get_unchecked(c2);
            sum1 += *query.get_unchecked(qi + 1) * *centroids.get_unchecked(c1)
                + *query.get_unchecked(qi + 3) * *centroids.get_unchecked(c3);
        }
    }
    // Handle remaining dimensions
    let remaining_start = nbytes * 4;
    if remaining_start < dim {
        let byte = unsafe { *packed.get_unchecked(nbytes) };
        for (off, shift) in [(0, 0), (1, 2), (2, 4), (3, 6)] {
            let j = remaining_start + off;
            if j >= dim { break; }
            let c = ((byte >> shift) & 0x03) as usize;
            unsafe { sum0 += *query.get_unchecked(j) * *centroids.get_unchecked(c); }
        }
    }
    sum0 + sum1
}

// ---------------------------------------------------------------------------
// CreateVectorStore
// ---------------------------------------------------------------------------

impl CreateVectorStore for WithTurboQuant {
    type Target = TQStore;

    fn create(
        self,
        max_points: usize,
        metric: Metric,
        prefetch_lookahead: Option<usize>,
    ) -> Self::Target {
        TQStore::new(self.quantizer, max_points, metric, prefetch_lookahead)
    }
}

impl VectorStore for TQStore {
    fn total(&self) -> usize {
        self.data.max_vectors()
    }

    fn count_for_get_vector(&self) -> usize {
        self.num_get_calls.get()
    }
}

// ---------------------------------------------------------------------------
// SetElementHelper
// ---------------------------------------------------------------------------

impl<T> SetElementHelper<T> for TQStore
where
    T: VectorRepr,
{
    fn set_element(&self, id: &u32, element: &[T]) -> ANNResult<()> {
        self.set_vector(id.into_usize(), element)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TQAccessor
// ---------------------------------------------------------------------------

/// The accessor for TurboQuant, analogous to `QuantAccessor` for SQ.
pub struct TQAccessor<'a, V, D, Ctx> {
    provider: &'a DefaultProvider<V, TQStore, D, Ctx>,
    id_buffer: Vec<u32>,
    is_search: bool,
}

impl<V, D, Ctx> GetFullPrecision for TQAccessor<'_, FullPrecisionStore<V>, D, Ctx>
where
    V: VectorRepr,
{
    type Repr = V;
    fn as_full_precision(&self) -> &FastMemoryVectorProviderAsync<AdHoc<V>> {
        &self.provider.base_vectors
    }
}

impl<V, D, Ctx> HasId for TQAccessor<'_, V, D, Ctx> {
    type Id = u32;
}

impl<V, D, Ctx> SearchExt for TQAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a, V, D, Ctx> TQAccessor<'a, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    pub(crate) fn new(
        provider: &'a DefaultProvider<V, TQStore, D, Ctx>,
        is_search: bool,
    ) -> Self {
        Self {
            provider,
            id_buffer: Vec::with_capacity(32),
            is_search,
        }
    }
}

impl<'a, V, D, Ctx> Accessor for TQAccessor<'a, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    /// The extended element inherits the lifetime of the Accessor.
    type Extended = TQRef<'a>;

    type Element<'b>
        = TQRef<'a>
    where
        Self: 'b;

    /// `ElementRef` has an arbitrarily short lifetime.
    type ElementRef<'b> = TQRef<'b>;

    type GetError = ANNError;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        std::future::ready(
            match self.provider.aux_vectors.get_vector(id.into_usize()) {
                Ok(v) => Ok(v),
                Err(err) => Err(err.into()),
            },
        )
    }

    fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> impl Future<Output = Result<(), Self::GetError>> + Send
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + for<'b> FnMut(Self::ElementRef<'b>, Self::Id),
    {
        let id_buffer = &mut self.id_buffer;
        id_buffer.clear();
        id_buffer.extend(itr);

        let len = id_buffer.len();
        let lookahead = self.provider.aux_vectors.prefetch_lookahead();

        // Prefetch the first few vectors.
        for id in id_buffer.iter().take(lookahead) {
            self.provider.aux_vectors.prefetch_hint(id.into_usize());
        }

        for (i, id) in id_buffer.iter().enumerate() {
            if lookahead > 0 && i + lookahead < len {
                self.provider
                    .aux_vectors
                    .prefetch_hint(id_buffer[i + lookahead].into_usize());
            }

            let vector = match self.provider.aux_vectors.get_vector(id.into_usize()) {
                Ok(v) => v,
                Err(e) => return std::future::ready(Err(e.into())),
            };

            f(vector, *id);
        }

        std::future::ready(Ok(()))
    }
}

impl<'a, V, D, Ctx> DelegateNeighbor<'a> for TQAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<V, D, Ctx, T> BuildQueryComputer<[T]> for TQAccessor<'_, V, D, Ctx>
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type QueryComputerError = ANNError;
    type QueryComputer = TQQueryComputer;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(self
            .provider
            .aux_vectors
            .query_computer(from, self.is_search)?)
    }
}

impl<V, D, Ctx, T> ExpandBeam<[T]> for TQAccessor<'_, V, D, Ctx>
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
}

impl<V, D, Ctx> BuildDistanceComputer for TQAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type DistanceComputerError = ANNError;
    type DistanceComputer = TQDistanceComputer;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(self.provider.aux_vectors.distance_computer()?)
    }
}

impl<V, D, Ctx> AsDeletionCheck for TQAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// SearchStrategy for quantized search when a full-precision store exists alongside
/// the TQ store. Post-processing includes a [`Rerank`] step using original vectors.
impl<D, Ctx, T>
    SearchStrategy<FullPrecisionProvider<T, TQStore, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer = TQQueryComputer;
    type SearchAccessor<'a> = TQAccessor<'a, FullPrecisionStore<T>, D, Ctx>;
    type SearchAccessorError = ANNError;
    type PostProcessor = glue::Pipeline<FilterStartPoints, Rerank>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, TQStore, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(TQAccessor::new(provider, true))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// SearchStrategy for quantized search when only the TQ store is present.
/// No full-precision reranking is possible.
impl<D, Ctx, T>
    SearchStrategy<DefaultProvider<NoStore, TQStore, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer = TQQueryComputer;
    type SearchAccessor<'a> = TQAccessor<'a, NoStore, D, Ctx>;
    type SearchAccessorError = ANNError;
    type PostProcessor = glue::Pipeline<FilterStartPoints, RemoveDeletedIdsAndCopy>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<NoStore, TQStore, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(TQAccessor::new(provider, true))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<V, D, Ctx> PruneStrategy<DefaultProvider<V, TQStore, D, Ctx>> for Quantized
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DistanceComputer = TQDistanceComputer;
    type PruneAccessor<'a> = TQAccessor<'a, V, D, Ctx>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<V, TQStore, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(TQAccessor::new(provider, false))
    }
}

impl<V, D, Ctx> FillSet for TQAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
}

impl<V, D, Ctx, T>
    InsertStrategy<DefaultProvider<V, TQStore, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    Quantized: SearchStrategy<DefaultProvider<V, TQStore, D, Ctx>, [T]>,
{
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

// ---------------------------------------------------------------------------
// SaveWith and LoadWith
// ---------------------------------------------------------------------------

impl SaveWith<AsyncIndexMetadata> for TQStore {
    type Ok = usize;
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        write_provider: &P,
        metadata: &AsyncIndexMetadata,
    ) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        // Save the compressed data using the bin format.
        let prefix = metadata.prefix();
        let data_path = format!("{}_tq_compressed.bin", prefix);
        let bytes_written = storage::bin::save_to_bin(self, write_provider, &data_path)?;
        // TODO: Save the quantizer (rotation matrix + nbits + dim) separately.
        // For now, return just the data bytes. The quantizer must be reconstructed
        // with the same seed on load.
        Ok(bytes_written)
    }
}

impl LoadWith<AsyncQuantLoadContext> for TQStore {
    type Error = ANNError;

    async fn load_with<P>(read_provider: &P, ctx: &AsyncQuantLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        let prefix = ctx.metadata.prefix();
        let data_path = format!("{}_tq_compressed.bin", prefix);

        // TODO: Load the quantizer from a stored protobuf/binary. For now, create a
        // placeholder that will be replaced by the caller with the correct quantizer.
        // This is a temporary stub -- real loading requires serialization support in
        // TurboQuantQuantizer.
        storage::bin::load_from_bin(
            read_provider,
            &data_path,
            |num_points, dim| {
                // Reconstruct dimension from the stored per-vector size.
                // Each vector is `dim` codes + 8 bytes of metadata, so the "dim" seen
                // by the bin loader is bytes_per_vector. The true vector dimension is
                // dim - 8.
                let true_dim = dim.saturating_sub(8);
                let mut rng = rand::rng();
                let quantizer = TurboQuantQuantizer::new(true_dim, 4, &mut rng);
                Ok(TQStore::new(
                    quantizer,
                    num_points,
                    ctx.metric,
                    ctx.prefetch_lookahead,
                ))
            },
        )
    }
}

/// Hook into [`storage::bin::load_from_bin`] by implementing [`storage::bin::SetData`].
impl storage::bin::SetData for TQStore {
    type Item = u8;

    fn set_data(&mut self, i: usize, element: &[Self::Item]) -> ANNResult<()> {
        // SAFETY: No race -- we have a mutable reference to `self`.
        unsafe { self.set_raw_vector(i, element) }
    }
}

/// Hook into [`storage::bin::save_to_bin`] by implementing [`storage::bin::GetData`].
impl storage::bin::GetData for TQStore {
    type Element = u8;
    type Item<'a> = &'a [u8];

    fn get_data(&self, i: usize) -> ANNResult<Self::Item<'_>> {
        Ok(unsafe { self.data.get_slice(i) })
    }

    fn total(&self) -> usize {
        self.data.max_vectors()
    }

    fn dim(&self) -> usize {
        self.data.dim()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum TQError {
    #[error("Input full-precision conversion error: {0}")]
    FullPrecisionConversionErr(String),

    #[error("Unsupported distance metric for TurboQuant: {0:?}")]
    UnsupportedDistanceMetric(Metric),
}

impl From<TQError> for ANNError {
    #[cold]
    fn from(err: TQError) -> Self {
        ANNError::log_sq_error(err)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use diskann_vector::distance::Metric;
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    const DIM: usize = 32;
    const NPTS: usize = 10;
    const NBITS: usize = 4;

    fn make_quantizer() -> TurboQuantQuantizer {
        let mut rng = StdRng::seed_from_u64(42);
        TurboQuantQuantizer::new(DIM, NBITS, &mut rng)
    }

    fn make_store(metric: Metric) -> TQStore {
        let quantizer = make_quantizer();
        TQStore::new(quantizer, NPTS, metric, None)
    }

    fn random_vector(seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let v: Vec<f32> = (0..DIM).map(|_| rand::Rng::random::<f32>(&mut rng) - 0.5).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_dim() {
        let store = make_store(Metric::L2);
        assert_eq!(store.dim(), DIM);
    }

    #[test]
    fn test_set_and_get_vector() {
        let store = make_store(Metric::L2);
        let v = random_vector(1);
        store.set_vector(0, &v).unwrap();
        let got = store.get_vector(0).unwrap();
        assert_eq!(got.codes.len(), packed_code_bytes(DIM, NBITS));
        assert!(got.meta.norm > 0.0);
    }

    #[test]
    fn test_distance_computer_symmetry() {
        let store = make_store(Metric::L2);
        let v1 = random_vector(1);
        let v2 = random_vector(2);
        store.set_vector(0, &v1).unwrap();
        store.set_vector(1, &v2).unwrap();

        let dc = store.distance_computer().unwrap();
        let r0 = store.get_vector(0).unwrap();
        let r1 = store.get_vector(1).unwrap();

        let d01 = dc.evaluate_similarity(r0, r1);
        let d10 = dc.evaluate_similarity(r1, r0);
        assert!(
            (d01 - d10).abs() < 1e-6,
            "distance is not symmetric: {d01} vs {d10}"
        );
    }

    #[test]
    fn test_distance_computer_self_is_zero() {
        let store = make_store(Metric::L2);
        let v = random_vector(3);
        store.set_vector(0, &v).unwrap();

        let dc = store.distance_computer().unwrap();
        let r = store.get_vector(0).unwrap();
        let d = dc.evaluate_similarity(r, r);
        assert!(d < 1e-6, "self-distance should be ~0, got {d}");
    }

    #[test]
    fn test_query_computer() {
        let store = make_store(Metric::L2);
        let v = random_vector(4);
        store.set_vector(0, &v).unwrap();

        let query = random_vector(5);
        let qc = store.query_computer(&query, false).unwrap();
        let r = store.get_vector(0).unwrap();
        let d = qc.evaluate_similarity(r);
        assert!(d >= 0.0, "distance should be non-negative, got {d}");
    }

    #[test]
    fn test_query_distance_is_reasonable() {
        let store = make_store(Metric::L2);
        let v = random_vector(6);
        store.set_vector(0, &v).unwrap();

        // Query with the same vector should yield near-zero distance.
        let qc = store.query_computer(&v, false).unwrap();
        let r = store.get_vector(0).unwrap();
        let d = qc.evaluate_similarity(r);
        // With 4-bit quantization, there is reconstruction error, but it should be small.
        assert!(
            d < 0.5,
            "distance from a vector to its own encoding should be small, got {d}"
        );
    }

    #[test]
    fn test_distance_preserves_ordering() {
        let store = make_store(Metric::L2);
        let query = random_vector(100);

        // Insert several vectors with known true distances.
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for i in 0..8 {
            let v = random_vector(200 + i);
            store.set_vector(i as usize, &v).unwrap();
            vectors.push(v);
        }

        // Compute true L2 and approximate L2 distances.
        let qc = store.query_computer(&query, false).unwrap();
        let mut true_dists: Vec<(usize, f32)> = Vec::new();
        let mut approx_dists: Vec<(usize, f32)> = Vec::new();
        for i in 0..8 {
            let true_l2: f32 = query
                .iter()
                .zip(vectors[i].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            true_dists.push((i, true_l2));

            let r = store.get_vector(i).unwrap();
            let approx_l2 = qc.evaluate_similarity(r);
            approx_dists.push((i, approx_l2));
        }

        true_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        approx_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check that the nearest neighbor under true distance is in the top-3
        // under approximate distance (a basic sanity check).
        let nearest_true = true_dists[0].0;
        let top3_approx: Vec<usize> = approx_dists.iter().take(3).map(|x| x.0).collect();
        assert!(
            top3_approx.contains(&nearest_true),
            "true nearest neighbor {nearest_true} not in approx top-3: {:?}",
            top3_approx,
        );
    }

    #[test]
    fn test_prefetch_hint_ok() {
        let store = make_store(Metric::L2);
        store.prefetch_hint(NPTS - 1);
    }

    #[test]
    #[should_panic]
    fn test_prefetch_hint_oob() {
        let store = make_store(Metric::L2);
        store.prefetch_hint(NPTS);
    }

    #[test]
    fn test_unsupported_metric() {
        let store = make_store(Metric::Cosine);
        let err = store.distance_computer().unwrap_err();
        match err {
            TQError::UnsupportedDistanceMetric(Metric::Cosine) => {}
            _ => panic!("expected UnsupportedDistanceMetric error"),
        }
    }

    #[test]
    fn test_set_raw_vector() {
        let store = make_store(Metric::L2);
        let bpv = bytes_per_vector(DIM, NBITS);
        let raw = vec![1u8; bpv];
        unsafe {
            store.set_raw_vector(0, &raw).unwrap();
        }
        let slice = unsafe { store.data.get_slice(0) };
        assert_eq!(&slice[..bpv], raw.as_slice());
    }

    #[test]
    fn test_vector_store_trait() {
        let store = make_store(Metric::L2);
        assert_eq!(store.total(), NPTS);
        assert_eq!(store.count_for_get_vector(), 0);
        let _ = store.get_vector(0).unwrap();
        assert_eq!(store.count_for_get_vector(), 1);
    }

    #[test]
    fn test_bytes_per_vector() {
        // 4-bit: 32/2 + 8 = 24; 128/2 + 8 = 72
        assert_eq!(bytes_per_vector(32, 4), 24);
        assert_eq!(bytes_per_vector(128, 4), 72);
        // 2-bit: 32/4 + 8 = 16; 128/4 + 8 = 40
        assert_eq!(bytes_per_vector(32, 2), 16);
        assert_eq!(bytes_per_vector(128, 2), 40);
        // 8-bit: 32 + 8 = 40
        assert_eq!(bytes_per_vector(32, 8), 40);
    }

    #[test]
    fn test_pack_and_extract_codes() {
        // 4-bit packing
        let codes = [3u8, 1, 4, 2, 15, 0, 7, 9];
        let packed = pack_codes(&codes, 4);
        assert_eq!(packed.len(), 4); // 8 codes / 2 per byte = 4 bytes
        for (i, &expected) in codes.iter().enumerate() {
            assert_eq!(extract_code(&packed, i, 4), expected, "4-bit code {i} mismatch");
        }

        // 2-bit packing
        let codes2 = [0u8, 1, 2, 3, 1, 0, 3, 2];
        let packed2 = pack_codes(&codes2, 2);
        assert_eq!(packed2.len(), 2); // 8 codes / 4 per byte = 2 bytes
        for (i, &expected) in codes2.iter().enumerate() {
            assert_eq!(extract_code(&packed2, i, 2), expected, "2-bit code {i} mismatch");
        }
    }

    #[test]
    fn test_tqref_roundtrip() {
        let dim = 4;
        let nbits = NBITS;
        let mut buf = vec![0u8; bytes_per_vector(dim, nbits)];
        let codes = [3u8, 1, 4, 2];
        let norm: f32 = 1.5;
        let qns: f32 = 2.25;
        write_encoded(&mut buf, dim, nbits, &codes, norm, qns);

        let r = tqref_from_slice(&buf, dim, nbits);
        // Codes are now packed, so verify via extraction
        for (i, &expected) in codes.iter().enumerate() {
            assert_eq!(extract_code(r.codes, i, nbits), expected, "code {i} mismatch");
        }
        assert!((r.meta.norm - norm).abs() < 1e-7);
        assert!((r.meta.quant_norm_sq - qns).abs() < 1e-7);
    }
}
