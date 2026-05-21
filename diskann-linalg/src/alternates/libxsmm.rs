/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Native libxsmm bindings + GEMM implementation.
//!
//! libxsmm (Intel) JIT-compiles a kernel specialised for each unique
//! `(m, n, k, lda, ldb, ldc, alpha, beta, flags)` combination, caches them in
//! its internal registry, and reuses across calls. The cached kernels embed
//! cache-friendly access patterns and architecture-appropriate microkernels
//! (AVX-512, AVX2, AMX, …).
//!
//! Bindings are written by hand (no `bindgen`/wrapper crate), targeting the
//! Ubuntu `libxsmm-dev` 1.17 static library.

#![cfg(feature = "libxsmm")]
#![allow(non_camel_case_types)]

use super::super::common::Transpose;
use std::os::raw::{c_float, c_int};
use std::sync::Once;

// ─── FFI signatures (libxsmm 1.17) ────────────────────────────────────────────

/// JIT-compiled SGEMM kernel: `c = (alpha * op(a) * op(b)) + (beta * c)`.
///
/// `a`, `b`, `c` are raw pointers to row-major matrices. The (m, n, k, lda,
/// ldb, ldc, alpha, beta, flags) parameters are baked into the kernel at JIT
/// time; only the data pointers are passed at call time.
pub type LibxsmmSmmFunction =
    Option<unsafe extern "C" fn(a: *const c_float, b: *const c_float, c: *mut c_float)>;

/// libxsmm GEMM flag values from `libxsmm_typedefs.h`.
pub const LIBXSMM_GEMM_FLAG_NONE: c_int = 0;
pub const LIBXSMM_GEMM_FLAG_TRANS_A: c_int = 1;
pub const LIBXSMM_GEMM_FLAG_TRANS_B: c_int = 2;
pub const LIBXSMM_GEMM_FLAG_BETA_0: c_int = 16;

unsafe extern "C" {
    /// Initialise the library. Idempotent.
    fn libxsmm_init();

    /// Free internal state (kernel registry, JIT memory). Currently unused —
    /// process-lifetime libxsmm state is fine since we want kernel reuse.
    #[allow(dead_code)]
    fn libxsmm_finalize();

    /// JIT-dispatch a SGEMM kernel. `NULL` for any of the pointer arguments
    /// requests libxsmm's default for that parameter (e.g. `lda = m`,
    /// `alpha = 1.0`, `beta = 1.0`).
    ///
    /// Returns `NULL` if JIT compilation is not supported on this CPU /
    /// for these dimensions; caller must fall back.
    fn libxsmm_smmdispatch(
        m: c_int,
        n: c_int,
        k: c_int,
        lda: *const c_int,
        ldb: *const c_int,
        ldc: *const c_int,
        alpha: *const c_float,
        beta: *const c_float,
        flags: *const c_int,
        prefetch: *const c_int,
    ) -> LibxsmmSmmFunction;
}

// ─── One-time init ────────────────────────────────────────────────────────────

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        // SAFETY: `libxsmm_init` is documented to be idempotent and safe to
        // call multiple times; `Once` guarantees we call it exactly once.
        unsafe { libxsmm_init() };
    });
}

// ─── Public GEMM entry point ──────────────────────────────────────────────────

/// SGEMM implementation backed by libxsmm. Same contract as `faer::sgemm_impl`.
///
/// # Falls back to faer if libxsmm dispatch returns NULL (rare — happens when
/// the (m, n, k) shape isn't JIT-able on this CPU).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm_impl(
    atranspose: Transpose,
    btranspose: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: Option<f32>,
    c: &mut [f32],
) {
    // Library expects column-major. We use row-major. The classic identity is:
    //
    //   C_rm = A_rm * B_rm   ⇔   C_cm^T = (A_rm * B_rm)^T = B_rm^T * A_rm^T
    //
    // ...so to compute a row-major `A·B` with a column-major GEMM call we swap
    // arguments: `gemm_cm(B, A, C, ...)` with B as the "left" operand, A as
    // the "right" operand, and (m, n, k) → (n, m, k).
    //
    // Transpose flags are also swapped accordingly.

    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    ensure_init();

    let cm_m: c_int = n.try_into().expect("n fits in i32");
    let cm_n: c_int = m.try_into().expect("m fits in i32");
    let cm_k: c_int = k.try_into().expect("k fits in i32");

    // Leading dims: in column-major terms after the swap, lda = n (stride of
    // the new "A" which is the original row-major B), ldb = k (stride of new
    // "B" = original A), ldc = n.
    let lda_swapped: c_int = if btranspose.is_transpose() { cm_k } else { cm_m };
    let ldb_swapped: c_int = if atranspose.is_transpose() { cm_n } else { cm_k };
    let ldc: c_int = cm_m;

    let mut flags: c_int = LIBXSMM_GEMM_FLAG_NONE;
    if atranspose.is_transpose() {
        // Original A transposed → in swapped form, the "B" operand is transposed.
        flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
    }
    if btranspose.is_transpose() {
        // Original B transposed → swapped "A" is transposed.
        flags |= LIBXSMM_GEMM_FLAG_TRANS_A;
    }
    if beta.is_none() {
        flags |= LIBXSMM_GEMM_FLAG_BETA_0;
    }

    let beta_val: c_float = beta.unwrap_or(0.0);
    let alpha_val: c_float = alpha;

    // SAFETY: `libxsmm_smmdispatch` accepts NULL for unspecified parameters,
    // but we pass concrete addresses for everything we care about (lda, ldb,
    // ldc, alpha, beta, flags). `prefetch = NULL` lets libxsmm pick.
    let kernel = unsafe {
        libxsmm_smmdispatch(
            cm_m,
            cm_n,
            cm_k,
            &lda_swapped,
            &ldb_swapped,
            &ldc,
            &alpha_val,
            &beta_val,
            &flags,
            std::ptr::null(),
        )
    };

    if let Some(k_fn) = kernel {
        // SAFETY: kernel was JIT-compiled for these exact (m, n, k, lda, ldb,
        // ldc, alpha, beta, flags). The slices we pass have the lengths
        // matching what was declared at dispatch (verified by debug_assert
        // above). After the row-major→column-major swap, A_cm = b (row-major
        // B), B_cm = a (row-major A), C_cm = c.
        unsafe {
            k_fn(b.as_ptr(), a.as_ptr(), c.as_mut_ptr());
        }
    } else {
        // JIT not available for this shape on this CPU — fall back to faer.
        // This path is exercised by very small dims (k=1, etc.) where JIT
        // overhead would exceed the work.
        super::super::faer::sgemm_impl(atranspose, btranspose, m, n, k, alpha, a, b, beta, c);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Transpose;

    /// Compare libxsmm's SGEMM result against the reference scalar implementation
    /// for a small case. Verifies the row-major↔column-major swap is correct.
    #[test]
    fn sgemm_matches_reference_small() {
        // C (2x3) = A (2x4) * B (4x3)
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let mut c = vec![0.0f32; 6];

        sgemm_impl(
            Transpose::None,
            Transpose::None,
            2,
            3,
            4,
            1.0,
            &a,
            &b,
            None,
            &mut c,
        );

        // Hand-computed: A · B
        // row 0 of A: [1, 2, 3, 4]; B columns: [1,0,0,1], [0,1,0,1], [0,0,1,1]
        //   c[0] = 1*1 + 2*0 + 3*0 + 4*1 = 5
        //   c[1] = 1*0 + 2*1 + 3*0 + 4*1 = 6
        //   c[2] = 1*0 + 2*0 + 3*1 + 4*1 = 7
        // row 1 of A: [5, 6, 7, 8]
        //   c[3] = 5*1 + 6*0 + 7*0 + 8*1 = 13
        //   c[4] = 5*0 + 6*1 + 7*0 + 8*1 = 14
        //   c[5] = 5*0 + 6*0 + 7*1 + 8*1 = 15
        let expected = [5.0, 6.0, 7.0, 13.0, 14.0, 15.0];
        for (got, want) in c.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }

    /// Verify `Transpose::Ordinary` on B argument computes A · B^T.
    /// This is the exact pattern `sgemm_abt` uses (leaf k-NN and partition).
    #[test]
    fn sgemm_abt_matches_reference() {
        // A is 2x3, B is 4x3, compute C = A · B^T → 2x4
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // 4x3
        let mut c = vec![0.0f32; 8];

        sgemm_impl(
            Transpose::None,
            Transpose::Ordinary,
            2,
            4,
            3,
            1.0,
            &a,
            &b,
            None,
            &mut c,
        );

        // A · B^T row 0 = [1,2,3] · rows of B → [1, 2, 3, 1+2+3]
        // A · B^T row 1 = [4,5,6] · rows of B → [4, 5, 6, 4+5+6]
        let expected = [1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0];
        for (got, want) in c.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }
}
