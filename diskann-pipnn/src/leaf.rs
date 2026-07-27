/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fused nearest-neighbor selection for a leaf's lower dot-product matrix.

use diskann_vector::distance::Metric;
use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

/// One leaf-local neighbor and its metric distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeafNeighbor {
    /// Position in the leaf, not a dataset ID.
    pub position: u32,
    /// Distance from the row point to `position`.
    pub distance: f32,
}

impl LeafNeighbor {
    /// Construct a leaf-local neighbor.
    pub const fn new(position: u32, distance: f32) -> Self {
        Self { position, distance }
    }
}

impl Default for LeafNeighbor {
    fn default() -> Self {
        Self::new(u32::MAX, f32::MAX)
    }
}

/// Lower-triangular dot products consumed by [`nearest_leaf_neighbors`].
#[derive(Clone, Copy, Debug)]
pub struct LeafTopK<'a> {
    /// Row-major `points * points` matrix. Only entries with `column <= row` are read.
    pub dots: &'a [f32],
    /// Number of points represented by the matrix.
    pub points: usize,
    /// Metric used to rank pairs.
    pub metric: Metric,
}

/// Reusable temporary storage for leaf top-k selection.
#[derive(Debug, Default)]
pub struct LeafTopKWorkspace {
    norms: Vec<f32>,
    worst: Vec<f32>,
}

impl LeafTopKWorkspace {
    /// Construct an empty workspace.
    pub const fn new() -> Self {
        Self {
            norms: Vec::new(),
            worst: Vec::new(),
        }
    }
}

/// Validation or allocation error returned by [`nearest_leaf_neighbors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LeafKernelError {
    /// The point count cannot be represented in leaf-local `u32` positions.
    #[error("point count {0} exceeds the u32 position limit")]
    TooManyPoints(usize),
    /// A declared shape overflowed `usize`.
    #[error("{buffer} shape {rows} x {cols} overflows usize")]
    ShapeOverflow {
        /// Name of the buffer whose shape overflowed.
        buffer: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },
    /// A supplied slice did not match its declared shape.
    #[error("invalid {buffer} length: expected {expected}, got {actual}")]
    InvalidBufferLength {
        /// Name of the invalid buffer.
        buffer: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// Temporary storage could not be reserved.
    #[error("failed to reserve {additional} values for {buffer}")]
    Allocation {
        /// Name of the temporary buffer.
        buffer: &'static str,
        /// Additional element capacity requested.
        additional: usize,
    },
    /// A row did not contain enough rankable pair distances to fill its output.
    #[error("row {row} has fewer than {neighbors} rankable leaf neighbors")]
    InsufficientRankableNeighbors {
        /// Zero-based row position in the leaf.
        row: usize,
        /// Required number of non-self neighbors.
        neighbors: usize,
    },
}

/// Select the nearest non-self leaf positions for every row.
///
/// The strictly lower triangle is scanned once. Each pair updates both row
/// trackers, so the upper triangle is neither read nor materialized. The
/// returned value is `min(k, points - 1)`, and `output` contains exactly
/// `points * returned_k` entries grouped by row and ordered by ascending
/// distance. Equal distances retain pair scan order.
pub fn nearest_leaf_neighbors(
    input: LeafTopK<'_>,
    k: usize,
    output: &mut [LeafNeighbor],
    workspace: &mut LeafTopKWorkspace,
) -> Result<usize, LeafKernelError> {
    let actual_k = validate(input, k, output)?;
    if actual_k == 0 {
        return Ok(0);
    }

    resize("norms", &mut workspace.norms, input.points, 0.0)?;
    resize(
        "worst distances",
        &mut workspace.worst,
        input.points,
        f32::MAX,
    )?;
    for (row, norm) in workspace.norms.iter_mut().enumerate() {
        let squared_norm = input.dots[row * input.points + row];
        *norm = if input.metric == Metric::Cosine {
            // Match diskann-vector: a finite/subnormal squared norm below this
            // threshold is a zero vector, while NaN continues through the
            // distance calculation as non-rankable.
            if squared_norm < f32::MIN_POSITIVE {
                0.0
            } else {
                squared_norm.sqrt()
            }
        } else {
            squared_norm
        };
    }
    output.fill(LeafNeighbor::default());
    workspace.worst.fill(f32::MAX);

    diskann_wide::arch::dispatch(LeafKernel {
        input,
        k: actual_k,
        output,
        norms: &workspace.norms,
        worst: &mut workspace.worst,
    });
    if let Some(row) = output.chunks_exact(actual_k).position(|neighbors| {
        neighbors
            .iter()
            .any(|neighbor| neighbor.position == u32::MAX)
    }) {
        return Err(LeafKernelError::InsufficientRankableNeighbors {
            row,
            neighbors: actual_k,
        });
    }
    Ok(actual_k)
}

fn validate(
    input: LeafTopK<'_>,
    k: usize,
    output: &[LeafNeighbor],
) -> Result<usize, LeafKernelError> {
    if input.points > u32::MAX as usize {
        return Err(LeafKernelError::TooManyPoints(input.points));
    }
    let matrix_len = checked_area("lower dot-product matrix", input.points, input.points)?;
    check_length("lower dot-product matrix", input.dots.len(), matrix_len)?;
    let actual_k = k.min(input.points.saturating_sub(1));
    let output_len = checked_area("output", input.points, actual_k)?;
    check_length("output", output.len(), output_len)?;
    Ok(actual_k)
}

fn resize<T: Clone>(
    buffer: &'static str,
    values: &mut Vec<T>,
    len: usize,
    value: T,
) -> Result<(), LeafKernelError> {
    if len > values.len() {
        let additional = len - values.len();
        values
            .try_reserve(additional)
            .map_err(|_| LeafKernelError::Allocation { buffer, additional })?;
        values.resize(len, value);
    } else {
        values.truncate(len);
    }
    Ok(())
}

fn checked_area(buffer: &'static str, rows: usize, cols: usize) -> Result<usize, LeafKernelError> {
    rows.checked_mul(cols)
        .ok_or(LeafKernelError::ShapeOverflow { buffer, rows, cols })
}

fn check_length(
    buffer: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), LeafKernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(LeafKernelError::InvalidBufferLength {
            buffer,
            expected,
            actual,
        })
    }
}

struct LeafKernel<'a, 'o, 'w> {
    input: LeafTopK<'a>,
    k: usize,
    output: &'o mut [LeafNeighbor],
    norms: &'w [f32],
    worst: &'w mut [f32],
}

impl LeafKernel<'_, '_, '_> {
    fn run_scalar(self) {
        process_pairs_scalar(self.input, self.k, self.output, self.norms, self.worst);
    }

    fn run_simd<F>(self, arch: F::Arch)
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        process_pairs_simd::<F>(
            arch,
            self.input,
            self.k,
            self.output,
            self.norms,
            self.worst,
        );
    }
}

impl diskann_wide::arch::Target<diskann_wide::arch::Scalar, ()> for LeafKernel<'_, '_, '_> {
    #[inline(always)]
    fn run(self, _: diskann_wide::arch::Scalar) {
        self.run_scalar();
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target<diskann_wide::arch::x86_64::V3, ()> for LeafKernel<'_, '_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V3) {
        diskann_wide::alias!(F32x8 = <diskann_wide::arch::x86_64::V3>::f32x8);
        self.run_simd::<F32x8>(arch);
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target<diskann_wide::arch::x86_64::V4, ()> for LeafKernel<'_, '_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V4) {
        diskann_wide::alias!(F32x16 = <diskann_wide::arch::x86_64::V4>::f32x16);
        self.run_simd::<F32x16>(arch);
    }
}

#[cfg(target_arch = "aarch64")]
impl diskann_wide::arch::Target<diskann_wide::arch::aarch64::Neon, ()> for LeafKernel<'_, '_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::aarch64::Neon) {
        let _scalar = arch.retarget();
        self.run_scalar();
    }
}

fn process_pairs_scalar(
    input: LeafTopK<'_>,
    k: usize,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) {
    for row in 1..input.points {
        for column in 0..row {
            let dot = input.dots[row * input.points + column];
            let distance = pair_distance(input.metric, dot, norms[row], norms[column]);
            insert_row(output, worst, k, row, column as u32, distance);
            insert_row(output, worst, k, column, row as u32, distance);
        }
    }
}

fn process_pairs_simd<F>(
    arch: F::Arch,
    input: LeafTopK<'_>,
    k: usize,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
{
    for row in 1..input.points {
        let row_start = row * input.points;
        let row_norm = F::splat(arch, norms[row]);
        let mut column = 0;
        while column + F::LANES <= row {
            // SAFETY: the full chunk is contained in the strict lower row prefix.
            let dots = unsafe { F::load_simd(arch, input.dots.as_ptr().add(row_start + column)) };
            // SAFETY: `column + F::LANES <= row < input.points == norms.len()`.
            let column_norms = unsafe { F::load_simd(arch, norms.as_ptr().add(column)) };
            let distances = pair_distances::<F>(arch, input.metric, dots, row_norm, column_norms);
            update_lanes(distances, row, column, k, output, worst);
            column += F::LANES;
        }
        while column < row {
            let dot = input.dots[row_start + column];
            let distance = pair_distance(input.metric, dot, norms[row], norms[column]);
            insert_row(output, worst, k, row, column as u32, distance);
            insert_row(output, worst, k, column, row as u32, distance);
            column += 1;
        }
    }
}

#[inline(always)]
fn pair_distances<F>(arch: F::Arch, metric: Metric, dot: F, row_norm: F, column_norm: F) -> F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
{
    let zero = F::default(arch);
    match metric {
        Metric::L2 => {
            let distance = row_norm + column_norm - F::splat(arch, 2.0) * dot;
            distance.lt_simd(zero).select(zero, distance)
        }
        Metric::CosineNormalized => {
            let distance = F::splat(arch, 1.0) - dot;
            distance.lt_simd(zero).select(zero, distance)
        }
        Metric::InnerProduct => zero - dot,
        Metric::Cosine => {
            let one = F::splat(arch, 1.0);
            let denominator = row_norm * column_norm;
            let zero_row = row_norm.eq_simd(zero);
            let zero_column = column_norm.eq_simd(zero);
            let safe_denominator = zero_row.select(one, zero_column.select(one, denominator));
            let cosine = zero_row.select(zero, zero_column.select(zero, dot / safe_denominator));
            let distance = one - cosine;
            // Comparisons with NaN are false, so this explicit lower clamp
            // preserves non-rankable NaNs while matching the existing PiPNN
            // distance formulas for finite values.
            distance.lt_simd(zero).select(zero, distance)
        }
    }
}

fn update_lanes<F>(
    distances: F,
    row: usize,
    column: usize,
    k: usize,
    output: &mut [LeafNeighbor],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32>,
{
    let mut values = [0.0f32; 16];
    // SAFETY: `values` has capacity for every f32 SIMD width DiskANN exposes.
    unsafe { distances.store_simd(values.as_mut_ptr()) };
    for (lane, &distance) in values[..F::LANES].iter().enumerate() {
        let column = column + lane;
        insert_row(output, worst, k, row, column as u32, distance);
        insert_row(output, worst, k, column, row as u32, distance);
    }
}

#[inline(always)]
fn pair_distance(metric: Metric, dot: f32, row_norm: f32, column_norm: f32) -> f32 {
    match metric {
        Metric::L2 => {
            let distance = row_norm + column_norm - 2.0 * dot;
            if distance < 0.0 {
                0.0
            } else {
                distance
            }
        }
        Metric::CosineNormalized => {
            let distance = 1.0 - dot;
            if distance < 0.0 {
                0.0
            } else {
                distance
            }
        }
        Metric::InnerProduct => -dot,
        Metric::Cosine => {
            let denominator = row_norm * column_norm;
            let cosine = if row_norm != 0.0 && column_norm != 0.0 {
                dot / denominator
            } else {
                0.0
            };
            let distance = 1.0 - cosine;
            if distance < 0.0 {
                0.0
            } else {
                distance
            }
        }
    }
}

#[inline(always)]
fn insert_row(
    output: &mut [LeafNeighbor],
    worst: &mut [f32],
    k: usize,
    row: usize,
    position: u32,
    distance: f32,
) {
    if distance.partial_cmp(&worst[row]) != Some(std::cmp::Ordering::Less) {
        return;
    }

    let start = row * k;
    let row_output = &mut output[start..start + k];
    row_output[k - 1] = LeafNeighbor::new(position, distance);
    let mut index = k - 1;
    while index > 0 && row_output[index].distance < row_output[index - 1].distance {
        row_output.swap(index, index - 1);
        index -= 1;
    }
    worst[row] = row_output[k - 1].distance;
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_wide::Architecture;
    use std::cmp::Ordering;

    fn make_input(metric: Metric, points: usize) -> (Vec<f32>, Vec<f32>) {
        let mut dots = vec![f32::NAN; points * points];
        for row in 0..points {
            dots[row * points + row] = if metric == Metric::Cosine && row == 0 {
                0.0
            } else if row == 2 {
                2.0
            } else {
                1.0 + (row % 5) as f32
            };
            for column in 0..row {
                let pair = ((row * 17 + column * 11) % 23) as f32 - 11.0;
                dots[row * points + column] = if row == points - 1 && column == 0 {
                    f32::NAN
                } else if column == 1 || column == 2 {
                    0.5
                } else {
                    pair * 0.03125
                };
            }
        }

        let norms = (0..points)
            .map(|row| {
                let squared = dots[row * points + row];
                if metric == Metric::Cosine {
                    if squared < f32::MIN_POSITIVE {
                        0.0
                    } else {
                        squared.sqrt()
                    }
                } else {
                    squared
                }
            })
            .collect();
        (dots, norms)
    }

    fn reference(input: LeafTopK<'_>, k: usize) -> Vec<LeafNeighbor> {
        let k = k.min(input.points.saturating_sub(1));
        let mut output = vec![LeafNeighbor::default(); input.points * k];
        if k == 0 {
            return output;
        }

        let norms: Vec<_> = (0..input.points)
            .map(|row| {
                let diagonal = input.dots[row * input.points + row];
                if input.metric == Metric::Cosine {
                    if diagonal < f32::MIN_POSITIVE {
                        0.0
                    } else {
                        diagonal.sqrt()
                    }
                } else {
                    diagonal
                }
            })
            .collect();

        for row in 0..input.points {
            let mut candidates = Vec::with_capacity(input.points - 1);
            for position in 0..input.points {
                if position == row {
                    continue;
                }
                let (lower_row, lower_column) = if row > position {
                    (row, position)
                } else {
                    (position, row)
                };
                let dot = input.dots[lower_row * input.points + lower_column];
                let distance = match input.metric {
                    Metric::L2 => {
                        let distance = norms[row] + norms[position] - 2.0 * dot;
                        if distance < 0.0 {
                            0.0
                        } else {
                            distance
                        }
                    }
                    Metric::CosineNormalized => {
                        let distance = 1.0 - dot;
                        if distance < 0.0 {
                            0.0
                        } else {
                            distance
                        }
                    }
                    Metric::InnerProduct => -dot,
                    Metric::Cosine => {
                        let denominator = norms[row] * norms[position];
                        let similarity = if norms[row] != 0.0 && norms[position] != 0.0 {
                            dot / denominator
                        } else {
                            0.0
                        };
                        let distance = 1.0 - similarity;
                        if distance < 0.0 {
                            0.0
                        } else {
                            distance
                        }
                    }
                };
                if distance.partial_cmp(&f32::MAX) == Some(Ordering::Less) {
                    candidates.push(LeafNeighbor::new(position as u32, distance));
                }
            }

            candidates.sort_by(|left, right| {
                left.distance
                    .partial_cmp(&right.distance)
                    .expect("NaN distances were filtered")
            });
            let count = candidates.len().min(k);
            output[row * k..row * k + count].copy_from_slice(&candidates[..count]);
        }
        output
    }

    fn assert_available_architectures(metric: Metric, points: usize, requested_k: usize) {
        let (dots, norms) = make_input(metric, points);
        let input = LeafTopK {
            dots: &dots,
            points,
            metric,
        };
        let k = requested_k.min(points.saturating_sub(1));
        let expected = reference(input, requested_k);

        let run = |arch_name: &str, output: &[LeafNeighbor]| {
            assert_eq!(
                output, expected,
                "{arch_name}, {metric:?}, n={points}, k={k}"
            );
        };

        let mut scalar = vec![LeafNeighbor::default(); points * k];
        let mut scalar_worst = vec![f32::MAX; points];
        diskann_wide::arch::Scalar::new().run(LeafKernel {
            input,
            k,
            output: &mut scalar,
            norms: &norms,
            worst: &mut scalar_worst,
        });
        run("Scalar", &scalar);

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            let mut output = vec![LeafNeighbor::default(); points * k];
            let mut worst = vec![f32::MAX; points];
            arch.run(LeafKernel {
                input,
                k,
                output: &mut output,
                norms: &norms,
                worst: &mut worst,
            });
            run("V3", &output);
        }

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            let mut output = vec![LeafNeighbor::default(); points * k];
            let mut worst = vec![f32::MAX; points];
            arch.run(LeafKernel {
                input,
                k,
                output: &mut output,
                norms: &norms,
                worst: &mut worst,
            });
            run("V4", &output);
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            let mut output = vec![LeafNeighbor::default(); points * k];
            let mut worst = vec![f32::MAX; points];
            arch.run(LeafKernel {
                input,
                k,
                output: &mut output,
                norms: &norms,
                worst: &mut worst,
            });
            run("Neon", &output);
        }
    }

    #[test]
    fn every_available_architecture_matches_independent_reference() {
        for metric in [
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct,
        ] {
            for points in [7, 8, 9, 15, 16, 17, 64] {
                for k in [1, 2, 4, usize::MAX] {
                    assert_available_architectures(metric, points, k);
                }
            }
            for points in [256, 512] {
                for k in [1, 2, 4] {
                    assert_available_architectures(metric, points, k);
                }
            }
        }
    }
}
