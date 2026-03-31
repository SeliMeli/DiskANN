/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! TurboQuant vector representations and distance functors.
//!
//! Provides owned, ref, and mut types for TurboQuant-encoded vectors,
//! and distance functors for L2 and inner product.

use super::centroids::LloydMaxCentroids;

/// Metadata stored per TurboQuant-encoded vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurboQuantMeta {
    /// L2 norm of the original vector.
    pub norm: f32,
    /// ||ỹ||² = Σⱼ (centroid[code[j]] / sqrt(d) * norm)².
    pub quant_norm_sq: f32,
}

/// Owned TurboQuant-encoded vector.
#[derive(Debug, Clone)]
pub struct TurboQuantData {
    /// Quantization codes (one byte per dimension, even for sub-byte bit widths).
    codes: Vec<u8>,
    /// Per-vector metadata.
    meta: TurboQuantMeta,
}

impl TurboQuantData {
    pub fn new(codes: Vec<u8>, norm: f32, quant_norm_sq: f32) -> Self {
        Self {
            codes,
            meta: TurboQuantMeta {
                norm,
                quant_norm_sq,
            },
        }
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn meta(&self) -> &TurboQuantMeta {
        &self.meta
    }

    pub fn as_ref(&self) -> TurboQuantDataRef<'_> {
        TurboQuantDataRef {
            codes: &self.codes,
            meta: &self.meta,
        }
    }

    pub fn as_mut(&mut self) -> TurboQuantDataMut<'_> {
        TurboQuantDataMut {
            codes: &mut self.codes,
            meta: &mut self.meta,
        }
    }
}

/// Borrowed reference to a TurboQuant-encoded vector.
#[derive(Debug, Clone, Copy)]
pub struct TurboQuantDataRef<'a> {
    pub codes: &'a [u8],
    pub meta: &'a TurboQuantMeta,
}

/// Mutable reference to a TurboQuant-encoded vector.
pub struct TurboQuantDataMut<'a> {
    pub codes: &'a mut [u8],
    pub meta: &'a mut TurboQuantMeta,
}

/// L2 squared distance functor for TurboQuant.
///
/// Computes L2²(q, x̃) = ||q'||² - 2·⟨q', ỹ⟩ + ||ỹ||²
///
/// where q' is the rotated query (precomputed), ỹ is the reconstructed
/// vector in rotated space, and ||ỹ||² is precomputed per vector.
#[derive(Debug, Clone)]
pub struct TurboQuantL2 {
    /// Centroids shared across all vectors.
    centroids: LloydMaxCentroids,
    /// 1.0 / sqrt(dim).
    inv_sqrt_dim: f32,
}

impl TurboQuantL2 {
    pub fn new(centroids: LloydMaxCentroids, dim: usize) -> Self {
        Self {
            centroids,
            inv_sqrt_dim: 1.0 / (dim as f32).sqrt(),
        }
    }

    /// Compute L2² between a rotated query and an encoded vector.
    ///
    /// # Arguments
    /// * `rotated_query` - The query vector after rotation: q' = Π·q
    /// * `query_norm_sq` - ||q||² = ||q'||² (rotation preserves norms)
    /// * `codes` - The quantization codes for the database vector
    /// * `meta` - Per-vector metadata (norm, quant_norm_sq)
    #[inline]
    pub fn evaluate(
        &self,
        rotated_query: &[f32],
        query_norm_sq: f32,
        codes: &[u8],
        meta: &TurboQuantMeta,
    ) -> f32 {
        let scale = self.inv_sqrt_dim * meta.norm;
        let mut ip_sum = 0.0f32;

        // Core loop: Σⱼ q'[j] · centroid[code[j]]
        for (j, &code) in codes.iter().enumerate() {
            ip_sum += rotated_query[j] * self.centroids.centroid(code);
        }
        ip_sum *= scale;

        (query_norm_sq - 2.0 * ip_sum + meta.quant_norm_sq).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbo_quant_data_roundtrip() {
        let codes = vec![0, 1, 2, 3, 2, 1, 0, 3];
        let data = TurboQuantData::new(codes.clone(), 1.5, 2.25);
        assert_eq!(data.codes(), &codes);
        assert_eq!(data.meta().norm, 1.5);
        assert_eq!(data.meta().quant_norm_sq, 2.25);

        let r = data.as_ref();
        assert_eq!(r.codes, &codes);
        assert_eq!(r.meta.norm, 1.5);
    }

    #[test]
    fn test_turbo_quant_l2_zero_distance() {
        let centroids = LloydMaxCentroids::new(2);
        let l2 = TurboQuantL2::new(centroids.clone(), 4);

        // A "vector" that when reconstructed equals the query
        // This requires the query to equal the reconstructed vector in rotated space
        let codes = vec![0, 1, 2, 3]; // Some codes
        let inv_sqrt_d = 1.0 / 2.0; // sqrt(4) = 2

        // Reconstruct what the vector would be in rotated space
        let recon: Vec<f32> = codes
            .iter()
            .map(|&c| centroids.centroid(c) * inv_sqrt_d)
            .collect();
        let norm = 1.0; // Assume unit norm for simplicity
        let recon_scaled: Vec<f32> = recon.iter().map(|&v| v * norm).collect();
        let quant_norm_sq: f32 = recon_scaled.iter().map(|v| v * v).sum();

        // Use the reconstruction as the "rotated query"
        let query_norm_sq: f32 = recon_scaled.iter().map(|v| v * v).sum();

        let meta = TurboQuantMeta {
            norm,
            quant_norm_sq,
        };
        let dist = l2.evaluate(&recon_scaled, query_norm_sq, &codes, &meta);
        assert!(
            dist < 1e-4,
            "Distance to self should be ~0, got {dist}"
        );
    }
}
