/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Precomputed Lloyd-Max optimal centroids for the standard normal distribution.
//!
//! After random rotation, each coordinate of a unit-norm vector follows approximately
//! N(0, 1/d). The Lloyd-Max algorithm finds the optimal scalar quantizer minimizing MSE
//! for this distribution. Since N(0, 1/d) = N(0,1) / sqrt(d), we store centroids for
//! N(0,1) and scale by 1/sqrt(d) at quantization time.
//!
//! Reference: TurboQuant paper Section 3.1, Equation 4.

/// Lloyd-Max optimal centroids for N(0,1) at various bit-widths.
///
/// These are computed offline via the Lloyd-Max iterative algorithm.
/// The centroids minimize E[(X - Q(X))²] where X ~ N(0,1) and Q is the quantizer.
#[derive(Debug, Clone)]
pub struct LloydMaxCentroids {
    /// Centroid values for N(0,1). Must be sorted in ascending order.
    centroids: Vec<f32>,
    /// Decision boundaries between centroids. boundaries[i] = midpoint of
    /// centroids[i] and centroids[i+1]. Length = num_centroids - 1.
    boundaries: Vec<f32>,
    /// Number of bits per coordinate.
    nbits: usize,
}

impl LloydMaxCentroids {
    /// Create centroids for the given bit-width using precomputed values.
    ///
    /// Supported bit-widths: 1, 2, 3, 4.
    /// For other bit-widths, use [`Self::compute`].
    pub fn new(nbits: usize) -> Self {
        let (centroids, boundaries) = match nbits {
            1 => precomputed_1bit(),
            2 => precomputed_2bit(),
            3 => precomputed_3bit(),
            4 => precomputed_4bit(),
            _ => {
                // For unsupported bit-widths, compute at runtime
                return Self::compute(nbits);
            }
        };
        Self {
            centroids,
            boundaries,
            nbits,
        }
    }

    /// Compute Lloyd-Max optimal centroids for N(0,1) at the given bit-width.
    ///
    /// Uses the iterative Lloyd-Max algorithm:
    /// 1. Initialize centroids uniformly over [-4, 4]
    /// 2. Update boundaries as midpoints between adjacent centroids
    /// 3. Update centroids as conditional means E[X | boundary_lo < X < boundary_hi]
    /// 4. Repeat until convergence
    pub fn compute(nbits: usize) -> Self {
        let num_levels = 1usize << nbits;
        let max_iters = 1000;
        let tol: f64 = 1e-12;

        // Initialize centroids uniformly
        let mut centroids: Vec<f64> = (0..num_levels)
            .map(|i| -4.0 + 8.0 * (i as f64 + 0.5) / num_levels as f64)
            .collect();
        let mut boundaries: Vec<f64> = vec![0.0; num_levels + 1];
        boundaries[0] = f64::NEG_INFINITY;
        *boundaries.last_mut().unwrap() = f64::INFINITY;

        for _ in 0..max_iters {
            let old_centroids = centroids.clone();

            // Update boundaries: midpoints between adjacent centroids
            for i in 1..num_levels {
                boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0;
            }

            // Update centroids: E[X | lo < X < hi] for X ~ N(0,1)
            // = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
            for i in 0..num_levels {
                let lo = boundaries[i];
                let hi = boundaries[i + 1];
                let prob = normal_cdf(hi) - normal_cdf(lo);
                if prob < 1e-15 {
                    continue;
                }
                let phi_lo = if lo.is_finite() {
                    normal_pdf(lo)
                } else {
                    0.0
                };
                let phi_hi = if hi.is_finite() {
                    normal_pdf(hi)
                } else {
                    0.0
                };
                centroids[i] = (phi_lo - phi_hi) / prob;
            }

            // Check convergence
            let max_change = centroids
                .iter()
                .zip(old_centroids.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if max_change < tol {
                break;
            }
        }

        Self {
            centroids: centroids.iter().map(|&c| c as f32).collect(),
            boundaries: boundaries[1..num_levels]
                .iter()
                .map(|&b| b as f32)
                .collect(),
            nbits,
        }
    }

    /// Number of bits per coordinate.
    pub fn nbits(&self) -> usize {
        self.nbits
    }

    /// Number of centroid levels (2^nbits).
    pub fn num_levels(&self) -> usize {
        self.centroids.len()
    }

    /// The centroid values for N(0,1), sorted ascending.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// The decision boundaries between centroids.
    pub fn boundaries(&self) -> &[f32] {
        &self.boundaries
    }

    /// Quantize a single coordinate value (already in the N(0,1) domain, i.e., after
    /// rotation and scaling by sqrt(d)).
    ///
    /// Returns the index of the nearest centroid.
    #[inline]
    pub fn quantize(&self, value: f32) -> u8 {
        // Binary search through boundaries
        let mut code: u8 = 0;
        for &b in &self.boundaries {
            if value > b {
                code += 1;
            } else {
                break;
            }
        }
        code
    }

    /// Look up the centroid value for a given code.
    #[inline]
    pub fn centroid(&self, code: u8) -> f32 {
        self.centroids[code as usize]
    }

    /// Centroid value scaled for dimension d: centroid / sqrt(d).
    /// This is the actual reconstructed coordinate value for a unit-norm vector.
    #[inline]
    pub fn centroid_scaled(&self, code: u8, inv_sqrt_d: f32) -> f32 {
        self.centroids[code as usize] * inv_sqrt_d
    }
}

// Precomputed Lloyd-Max centroids (verified against paper values)

fn precomputed_1bit() -> (Vec<f32>, Vec<f32>) {
    // Paper: ±sqrt(2/pi) = ±0.7978845608
    (vec![-0.7978845608, 0.7978845608], vec![0.0])
}

fn precomputed_2bit() -> (Vec<f32>, Vec<f32>) {
    // Paper: ±0.4528, ±1.5104
    (
        vec![-1.5104176085, -0.4527800346, 0.4527800346, 1.5104176085],
        vec![-0.9815988216, 0.0, 0.9815988216],
    )
}

fn precomputed_3bit() -> (Vec<f32>, Vec<f32>) {
    (
        vec![
            -2.1519457045,
            -1.3439092785,
            -0.7560052812,
            -0.2450941789,
            0.2450941789,
            0.7560052812,
            1.3439092785,
            2.1519457045,
        ],
        vec![
            -1.7479274915,
            -1.0499572799,
            -0.5005497301,
            0.0,
            0.5005497301,
            1.0499572799,
            1.7479274915,
        ],
    )
}

fn precomputed_4bit() -> (Vec<f32>, Vec<f32>) {
    (
        vec![
            -2.7325895710,
            -2.0690172266,
            -1.6180463860,
            -1.2562311974,
            -0.9423404565,
            -0.6567591185,
            -0.3880482995,
            -0.1283950299,
            0.1283950299,
            0.3880482995,
            0.6567591185,
            0.9423404565,
            1.2562311974,
            1.6180463860,
            2.0690172266,
            2.7325895710,
        ],
        vec![
            -2.4008033988,
            -1.8435318063,
            -1.4371387917,
            -1.0992858269,
            -0.7995497875,
            -0.5224037090,
            -0.2582216647,
            0.0,
            0.2582216647,
            0.5224037090,
            0.7995497875,
            1.0992858269,
            1.4371387917,
            1.8435318063,
            2.4008033988,
        ],
    )
}

/// Standard normal PDF: phi(x) = (1/sqrt(2*pi)) * exp(-x²/2)
fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF approximation using the error function.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun, formula 7.1.26).
/// Maximum error: 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1bit_centroids_match_paper() {
        let c = LloydMaxCentroids::new(1);
        assert_eq!(c.num_levels(), 2);
        // Paper: ±sqrt(2/pi)
        let expected = (2.0_f32 / std::f32::consts::PI).sqrt();
        assert!((c.centroids()[0] - (-expected)).abs() < 1e-6);
        assert!((c.centroids()[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_2bit_centroids_match_paper() {
        let c = LloydMaxCentroids::new(2);
        assert_eq!(c.num_levels(), 4);
        // Paper: ±0.4528, ±1.5104
        assert!((c.centroids()[0] - (-1.5104)).abs() < 0.001);
        assert!((c.centroids()[1] - (-0.4528)).abs() < 0.001);
        assert!((c.centroids()[2] - 0.4528).abs() < 0.001);
        assert!((c.centroids()[3] - 1.5104).abs() < 0.001);
    }

    #[test]
    fn test_computed_matches_precomputed() {
        for nbits in 1..=4 {
            let precomputed = LloydMaxCentroids::new(nbits);
            let computed = LloydMaxCentroids::compute(nbits);
            assert_eq!(precomputed.num_levels(), computed.num_levels());
            for (a, b) in precomputed.centroids().iter().zip(computed.centroids()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "nbits={nbits}: precomputed {a} != computed {b}"
                );
            }
        }
    }

    #[test]
    fn test_centroids_are_sorted() {
        for nbits in 1..=4 {
            let c = LloydMaxCentroids::new(nbits);
            for i in 1..c.num_levels() {
                assert!(
                    c.centroids()[i] > c.centroids()[i - 1],
                    "centroids not sorted at nbits={nbits}"
                );
            }
        }
    }

    #[test]
    fn test_centroids_are_symmetric() {
        for nbits in 1..=4 {
            let c = LloydMaxCentroids::new(nbits);
            let n = c.num_levels();
            for i in 0..n / 2 {
                assert!(
                    (c.centroids()[i] + c.centroids()[n - 1 - i]).abs() < 1e-6,
                    "centroids not symmetric at nbits={nbits}, i={i}"
                );
            }
        }
    }

    #[test]
    fn test_quantize_basic() {
        let c = LloydMaxCentroids::new(2);
        // Very negative → code 0
        assert_eq!(c.quantize(-3.0), 0);
        // Slightly negative → code 1
        assert_eq!(c.quantize(-0.2), 1);
        // Slightly positive → code 2
        assert_eq!(c.quantize(0.2), 2);
        // Very positive → code 3
        assert_eq!(c.quantize(3.0), 3);
    }

    #[test]
    fn test_quantize_at_boundaries() {
        let c = LloydMaxCentroids::new(2);
        // At boundary 0.0: should go to code 1 (not > 0.0)
        assert_eq!(c.quantize(0.0), 1);
        // Just above 0.0: should go to code 2
        assert_eq!(c.quantize(0.001), 2);
    }

    #[test]
    fn test_centroid_lookup() {
        let c = LloydMaxCentroids::new(2);
        for i in 0..4 {
            assert_eq!(c.centroid(i), c.centroids()[i as usize]);
        }
    }

    #[test]
    fn test_mse_distortion_matches_paper() {
        // Verify MSE by numerical integration
        let expected_mse = [0.3634, 0.1175, 0.03454, 0.009497];
        for nbits in 1..=4usize {
            let c = LloydMaxCentroids::new(nbits);
            // Numerical MSE for N(0,1)
            let n = 1_000_000;
            let dx = 12.0 / n as f64; // from -6 to 6
            let mut mse = 0.0f64;
            for i in 0..n {
                let x = -6.0 + (i as f64 + 0.5) * dx;
                let pdf = normal_pdf(x);
                let code = c.quantize(x as f32);
                let recon = c.centroid(code) as f64;
                mse += (x - recon) * (x - recon) * pdf * dx;
            }
            assert!(
                (mse - expected_mse[nbits - 1] as f64).abs() < 0.002,
                "nbits={nbits}: MSE {mse:.6} vs expected {:.6}",
                expected_mse[nbits - 1]
            );
        }
    }

    #[test]
    fn test_normal_cdf_accuracy() {
        // Known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.0) - 0.8413).abs() < 0.001);
        assert!((normal_cdf(-1.0) - 0.1587).abs() < 0.001);
        assert!((normal_cdf(2.0) - 0.9772).abs() < 0.001);
    }
}
