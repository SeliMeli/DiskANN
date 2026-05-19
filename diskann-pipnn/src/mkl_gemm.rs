/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Mixed-precision GEMM via Intel MKL `cblas_gemm_f16f16f32`:
//! f16 inputs, f32 output. Eliminates fp16→f32 input conversion AND produces
//! f32 dots directly — no output conversion needed before distance/top-k.
//! Requires AVX-512-FP16 hardware (Sapphire Rapids+) for best throughput.

use half::f16;

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct MklF16(u16);

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

extern "C" {
    fn cblas_gemm_f16f16f32(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const MklF16,
        lda: i32,
        b: *const MklF16,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );

    // MKL exports both `mkl_set_num_threads` (lowercase, Fortran-style — expects
    // `*const int`) AND `MKL_Set_Num_Threads` (camel case, C ABI — takes int by
    // value). They live at different addresses (0x28ea80 vs 0x28ea90 in
    // libmkl_intel_lp64.so.3). Calling the Fortran symbol with an int-by-value
    // segfaults instantly (`mov (%rdi),%edi` dereferences address 0x1). Use the
    // C symbol via #[link_name].
    #[link_name = "MKL_Set_Num_Threads"]
    fn mkl_set_num_threads(n: i32);
}

/// One-time initialization: force MKL to single-threaded mode so it doesn't
/// oversubscribe with rayon's thread pool.
static MKL_INIT: std::sync::Once = std::sync::Once::new();

fn ensure_mkl_single_threaded() {
    MKL_INIT.call_once(|| {
        // SAFETY: FFI call to C-ABI MKL_Set_Num_Threads, int by value.
        unsafe { mkl_set_num_threads(1) };
    });
}

/// Compute C = A * B^T where A is m×k (f16), B is n×k (f16), C is m×n (f32).
/// All row-major.
pub fn gemm_f16f16f32_abt(
    a: &[f16],
    m: usize,
    k: usize,
    b: &[f16],
    n: usize,
    c: &mut [f32],
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= n * k);
    debug_assert!(c.len() >= m * n);

    ensure_mkl_single_threaded();

    // SAFETY: MklF16 is repr(C) with the same layout as half::f16 (u16).
    // cblas_gemm_f16f16f32 reads f16 inputs and writes f32 output.
    unsafe {
        cblas_gemm_f16f16f32(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0f32,
            a.as_ptr() as *const MklF16,
            k as i32,
            b.as_ptr() as *const MklF16,
            k as i32,
            0.0f32,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// Compute C = A * A^T where A is m×k (f16), C is m×m (f32).
///
/// MKL's `cblas_gemm_f16f16f32` is a general GEMM with no symmetric (SYRK)
/// variant for f16 inputs. Passing the same pointer for A and B segfaults inside
/// `libmkl_intel_lp64.so.3` (observed: NULL-deref at offset 0x28ea84). We copy
/// A into a thread-local scratch buffer and call abt with two distinct pointers.
pub fn gemm_f16f16f32_aat(a: &[f16], m: usize, k: usize, c: &mut [f32]) {
    use std::cell::RefCell;
    thread_local! {
        static B_SCRATCH: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
    }
    B_SCRATCH.with(|cell| {
        let mut b = cell.borrow_mut();
        let needed = m * k;
        if b.len() < needed {
            b.resize(needed, f16::ZERO);
        }
        b[..needed].copy_from_slice(&a[..needed]);
        gemm_f16f16f32_abt(a, m, k, &b[..needed], m, c);
    });
}
