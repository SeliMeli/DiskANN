/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! TurboQuant data compression for disk index search.
//!
//! Replaces PQ compression: each vector is rotated then scalar-quantized per coordinate
//! using Lloyd-Max centroids. The compressed format per vector is:
//!   [packed_codes (ceil(nbits * dim / 8) bytes)] [norm (f32)] [quant_norm_sq (f32)]

use diskann::{utils::VectorRepr, ANNResult};
use diskann_quantization::turboquant::TurboQuantQuantizer;
use diskann_utils::views::{MatrixView, MutMatrixView};

use crate::storage::quant::compressor::{CompressionStage, QuantCompressor};

/// Context for creating a TQ compressor.
pub struct TQGenerationContext {
    pub quantizer: TurboQuantQuantizer,
}

/// TQ compressor that implements the `QuantCompressor` trait for `QuantDataGenerator`.
pub struct TQGeneration {
    quantizer: TurboQuantQuantizer,
    /// Bytes per compressed vector: ceil(nbits * dim / 8) + 4 (norm) + 4 (quant_norm_sq).
    compressed_bytes: usize,
}

impl TQGeneration {
    fn packed_code_bytes(dim: usize, nbits: usize) -> usize {
        (nbits * dim + 7) / 8
    }

    fn total_bytes(dim: usize, nbits: usize) -> usize {
        Self::packed_code_bytes(dim, nbits) + 4 + 4 // codes + norm + quant_norm_sq
    }

    fn pack_codes(codes: &[u8], nbits: usize, packed: &mut [u8]) {
        match nbits {
            1 | 2 | 4 => {
                let codes_per_byte = 8 / nbits;
                packed.iter_mut().for_each(|b| *b = 0);
                for (i, &code) in codes.iter().enumerate() {
                    let byte_idx = i / codes_per_byte;
                    let bit_offset = (i % codes_per_byte) * nbits;
                    packed[byte_idx] |= code << bit_offset;
                }
            }
            _ => {
                // For 3, 5-8 bits, store one code per byte
                packed[..codes.len()].copy_from_slice(codes);
            }
        }
    }
}

impl<T> QuantCompressor<T> for TQGeneration
where
    T: VectorRepr,
{
    type CompressorContext = TQGenerationContext;

    fn new_at_stage(_stage: CompressionStage, context: &Self::CompressorContext) -> ANNResult<Self> {
        let dim = context.quantizer.dim();
        let nbits = context.quantizer.nbits();
        Ok(Self {
            quantizer: context.quantizer.clone(),
            compressed_bytes: Self::total_bytes(dim, nbits),
        })
    }

    fn compress(&self, vectors: MatrixView<f32>, mut output: MutMatrixView<u8>) -> ANNResult<()> {
        let dim = self.quantizer.dim();
        let nbits = self.quantizer.nbits();
        let code_bytes = Self::packed_code_bytes(dim, nbits);

        for row_idx in 0..vectors.nrows() {
            let vec = vectors.row(row_idx);
            let encoded = self.quantizer.encode(vec);

            let out_row = output.row_mut(row_idx);

            // Pack codes
            Self::pack_codes(&encoded.codes, nbits, &mut out_row[..code_bytes]);

            // Write norm and quant_norm_sq as little-endian f32
            out_row[code_bytes..code_bytes + 4].copy_from_slice(&encoded.norm.to_le_bytes());
            out_row[code_bytes + 4..code_bytes + 8]
                .copy_from_slice(&encoded.quant_norm_sq.to_le_bytes());
        }
        Ok(())
    }

    fn compressed_bytes(&self) -> usize {
        self.compressed_bytes
    }
}
