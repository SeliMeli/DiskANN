/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! TurboQuant: Data-oblivious vector quantization based on random rotation
//! and Lloyd-Max optimal scalar quantizers.
//!
//! Based on the paper "TurboQuant" (arXiv:2504.19874).
//!
//! # Algorithm (TurboQuant_MSE)
//!
//! 1. Apply random orthogonal rotation Π to input vector x
//! 2. Each rotated coordinate is quantized independently using precomputed
//!    Lloyd-Max optimal centroids for N(0, 1/d)
//! 3. Store b-bit codes per dimension + vector norm
//!
//! # Distance Computation
//!
//! For L2 distance between query q and quantized vector x̃:
//!   L2(q, x̃) = ||Πq - ỹ||²
//! where ỹ[j] = centroid[code[j]] is the reconstructed vector in rotated space.
//!
//! This avoids the expensive inverse rotation by working in the rotated domain.

pub mod centroids;
pub mod quantizer;
pub mod vectors;

pub use centroids::LloydMaxCentroids;
pub use quantizer::TurboQuantQuantizer;
pub use vectors::{TurboQuantData, TurboQuantDataMut, TurboQuantDataRef, TurboQuantL2, TurboQuantMeta};
