/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Raw CBLAS bindings for Intel oneMKL. Used by the `mkl` feature gate to
//! provide dtype-dispatched Gram-matrix kernels for PiPNN:
//!   - f32      → `cblas_ssyrk` (SYRK lower) / `cblas_sgemm` (general A·Bᵀ)
//!   - f64      → `cblas_dsyrk` / `cblas_dgemm`, result widened to f32
//!   - bf16     → `cblas_gemm_bf16bf16f32` (AMX-bf16), full A·Bᵀ, f32 output
//!   - u8 / i8  → `cblas_gemm_s8u8s32` (AMX-int8), centered s8×u8 → s32 → f32
//!
//! The u8 path uses `cblas_gemm_s8u8s32`, whose CBLAS operand convention is
//! **A = u8, B = s8** (despite the name — the `s8u8` refers to MKL's internal
//! data order, not the CBLAS A/B order; verified empirically). So we keep the
//! A operand RAW u8 and CENTER the B operand (`b' = b - 128 ∈ [-128,127]`, a
//! valid s8). MKL then computes
//!   s32[i][j] = Σ A[i][l]·(B[j][l]-128) = Σ A·B - 128·Σ_l A[i][l]
//! and we recover the true integer Gram by adding `+128·rowsum_A[i]` PER OUTPUT
//! ROW i (where `rowsum_A[i] = Σ_l A[i][l]`) in the s32→f32 conversion pass.
//! This is exact for any u8 in [0,255]. See unit tests for the bit-exact check.
//!
//! Threading: PiPNN rayon-parallelizes across leaves/chunks, so MKL must run
//! single-threaded per call. We set `MKL_THREADING_LAYER=SEQUENTIAL` via a
//! process-wide `Once` (env must be set before the first MKL call, which the
//! `Once` guarantees by also calling `mkl_set_num_threads_local(1)`).
//!
//! Linker setup is in `diskann-linalg/build.rs`.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::c_int;

// MKL_INT is `int` for the LP64 interface (libmkl_rt default). MKL_BF16 is
// `unsigned short`. MKL_INT8 = i8, MKL_INT32 = i32.
type MklInt = c_int;
type MklBf16 = u16;

// CBLAS enum constants. We pass them as plain `c_int` (not Rust enums) to the
// extern functions. Passing fieldless Rust enums by value miscompiled the
// multi-enum CBLAS signatures (sgemm/gemm_bf16/gemm_s8u8 silently produced
// zero output while single-enum ssyrk worked); plain `c_int` args match the C
// ABI exactly (verified against a standalone C harness).
const CBLAS_ROW_MAJOR: MklInt = 101;
const CBLAS_NO_TRANS: MklInt = 111;
const CBLAS_TRANS: MklInt = 112; // CblasTrans=112 (was wrongly 121=CblasUpper)
const CBLAS_LOWER: MklInt = 122;
const CBLAS_FIX_OFFSET: MklInt = 173;
// Packed-GEMM (cblas_sgemm_pack / cblas_sgemm_compute) identifiers.
// Values from mkl_cblas.h: CBLAS_IDENTIFIER {CblasAMatrix=161, CblasBMatrix=162},
// CBLAS_STORAGE {CblasPacked=151}. NOT the SYRK/offset enum range.
const CBLAS_A_MATRIX: MklInt = 161;
const CBLAS_B_MATRIX: MklInt = 162;
const CBLAS_PACKED: MklInt = 151;

unsafe extern "C" {
    pub unsafe fn cblas_sgemm(
        layout: MklInt,
        transa: MklInt,
        transb: MklInt,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: f32,
        a: *const f32,
        lda: MklInt,
        b: *const f32,
        ldb: MklInt,
        beta: f32,
        c: *mut f32,
        ldc: MklInt,
    );

    pub unsafe fn cblas_dgemm(
        layout: MklInt,
        transa: MklInt,
        transb: MklInt,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: f64,
        a: *const f64,
        lda: MklInt,
        b: *const f64,
        ldb: MklInt,
        beta: f64,
        c: *mut f64,
        ldc: MklInt,
    );

    pub unsafe fn cblas_ssyrk(
        layout: MklInt,
        uplo: MklInt,
        trans: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: f32,
        a: *const f32,
        lda: MklInt,
        beta: f32,
        c: *mut f32,
        ldc: MklInt,
    );

    pub unsafe fn cblas_dsyrk(
        layout: MklInt,
        uplo: MklInt,
        trans: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: f64,
        a: *const f64,
        lda: MklInt,
        beta: f64,
        c: *mut f64,
        ldc: MklInt,
    );

    // CBLAS convention (despite the name): operand A is u8, operand B is s8 →
    // s32 output. Computes C = alpha·(A-ao)·op(B-bo) + beta·C + co, with co
    // gated by OffsetC. We pass ao=bo=0 and OffsetC=Fix, co=[0], then apply the
    // +128·rowsum correction ourselves in the s32→f32 conversion pass.
    pub unsafe fn cblas_gemm_s8u8s32(
        layout: MklInt,
        transa: MklInt,
        transb: MklInt,
        offsetc: MklInt,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: f32,
        a: *const std::ffi::c_void,
        lda: MklInt,
        ao: i8,
        b: *const std::ffi::c_void,
        ldb: MklInt,
        bo: i8,
        beta: f32,
        c: *mut i32,
        ldc: MklInt,
        co: *const i32,
    );

    // bf16 (A) × bf16 (B) → f32 (C). AMX-bf16 on Granite Rapids.
    pub unsafe fn cblas_gemm_bf16bf16f32(
        layout: MklInt,
        transa: MklInt,
        transb: MklInt,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: f32,
        a: *const MklBf16,
        lda: MklInt,
        b: *const MklBf16,
        ldb: MklInt,
        beta: f32,
        c: *mut f32,
        ldc: MklInt,
    );

    // ── Packed sgemm (cblas_?gemm_pack) ───────────────────────────────────────
    // Pack the B operand once and reuse across many sgemm_compute calls (the
    // partition reuses the leader matrix across every point minibatch). Returns
    // the packed-buffer size in BYTES.
    pub unsafe fn cblas_sgemm_pack_get_size(
        identifier: MklInt,
        m: MklInt,
        n: MklInt,
        k: MklInt,
    ) -> usize;

    pub unsafe fn cblas_sgemm_pack(
        layout: MklInt,
        identifier: MklInt,
        trans: MklInt,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: f32,
        src: *const f32,
        ld: MklInt,
        dest: *mut f32,
    );

    pub unsafe fn cblas_sgemm_compute(
        layout: MklInt,
        transa: MklInt,
        transb: MklInt,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        a: *const f32,
        lda: MklInt,
        b: *const f32,
        ldb: MklInt,
        beta: f32,
        c: *mut f32,
        ldc: MklInt,
    );

    // ── JIT sgemm (mkl_jit_*) ─────────────────────────────────────────────────
    // Generate a specialized kernel for one (m,n,k) shape to skip MKL's per-call
    // dispatch overhead on small GEMMs. Returns: 0=success, 1=no JIT (kernel still
    // usable via the standard path), 2=error.
    pub unsafe fn mkl_cblas_jit_create_sgemm(
        jitter: *mut *mut std::ffi::c_void,
        layout: MklInt,
        transa: MklInt,
        transb: MklInt,
        m: MklInt,
        n: MklInt,
        k: MklInt,
        alpha: f32,
        lda: MklInt,
        ldb: MklInt,
        beta: f32,
        ldc: MklInt,
    ) -> MklInt;

    pub unsafe fn mkl_jit_get_sgemm_ptr(
        jitter: *const std::ffi::c_void,
    ) -> Option<unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, *const f32, *mut f32)>;

    pub unsafe fn mkl_jit_destroy(jitter: *mut std::ffi::c_void) -> MklInt;
}

// MKL must run single-threaded per call: PiPNN rayon-parallelizes across
// leaves/chunks. We control this purely through env vars
// (MKL_THREADING_LAYER=SEQUENTIAL, MKL_NUM_THREADS=1, MKL_INTERFACE_LAYER=LP64)
// which libmkl_rt reads at its first computational entry.
//
// IMPORTANT: we do NOT call any MKL control routine
// (`mkl_set_num_threads`, `mkl_set_threading_layer`, `mkl_set_interface_layer`).
// In oneMKL 2026.0's `libmkl_rt` these SIGSEGV when called from this process
// (confirmed by gdb backtrace; the compute kernels themselves work fine — a
// standalone C harness calling only `cblas_ssyrk` with the env vars set
// returns correct results). Env-only layer selection is the documented and
// crash-free path.
static MKL_PIN: std::sync::Once = std::sync::Once::new();

#[inline]
fn ensure_mkl_single_threaded() {
    MKL_PIN.call_once(|| {
        // Select layers via env (read by libmkl_rt at first MKL call). The
        // `Once` guarantees this runs before any computational MKL routine in
        // this module. If the caller already exported these, set_var is a
        // harmless no-op of the same value.
        // SAFETY: process-wide env mutation; serialized by the Once and done
        // before the first MKL call, so no concurrent MKL reader races it.
        unsafe {
            if std::env::var_os("MKL_THREADING_LAYER").is_none() {
                std::env::set_var("MKL_THREADING_LAYER", "SEQUENTIAL");
            }
            if std::env::var_os("MKL_INTERFACE_LAYER").is_none() {
                std::env::set_var("MKL_INTERFACE_LAYER", "LP64");
            }
            if std::env::var_os("MKL_NUM_THREADS").is_none() {
                std::env::set_var("MKL_NUM_THREADS", "1");
            }
        }
    });
}

// ─── f32 ────────────────────────────────────────────────────────────────────

/// `C = A · Aᵀ` for `m × k` row-major A, LOWER triangle only. MKL ssyrk.
pub fn ssyrk_aat_lower(a: &[f32], m: usize, k: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(c.len(), m * m);
    ensure_mkl_single_threaded();
    // SAFETY: pointers from live slices; lda=k, ldc=m match row-major stride.
    unsafe {
        cblas_ssyrk(
            CBLAS_ROW_MAJOR,
            CBLAS_LOWER,
            CBLAS_NO_TRANS,
            m as MklInt,
            k as MklInt,
            1.0,
            a.as_ptr(),
            k as MklInt,
            0.0,
            c.as_mut_ptr(),
            m as MklInt,
        );
    }
}

/// `C = A · Bᵀ` for `m × k` A and `n × k` B (row-major). Output `m × n`.
///
/// oneMKL 2026.0 rejects `TransB=Trans` (errors "Parameter 3 incorrect") for
/// every A·Bᵀ wrapper in this build's row-major path. We materialize Bᵀ (k×n
/// row-major) and call NoTrans×NoTrans. Transpose cost is O(n·k), small next
/// to the m·n·k GEMM.
pub fn sgemm_abt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    ensure_mkl_single_threaded();
    // C = A · Bᵀ via NoTrans × Trans. CblasTrans=112 is honored for row-major
    // (same fix as the u8 path) — B is passed directly (n×k, ldb=k), NO per-call
    // transpose/alloc (the old materialized Bᵀ was ~786KB alloc + scalar strided
    // transpose PER CALL, which dominated the partition GEMM on high-d f32).
    // SAFETY: A m×k (lda=k), B n×k (ldb=k, op(B)=Bᵀ), C m×n (ldc=n).
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as MklInt,
            n as MklInt,
            k as MklInt,
            1.0,
            a.as_ptr(),
            k as MklInt,
            b.as_ptr(),
            k as MklInt,
            0.0,
            c.as_mut_ptr(),
            n as MklInt,
        );
    }
}


// ─── Packed sgemm (pack B once, reuse across minibatches) ────────────────────

/// A pre-packed B operand for `C = A · Bᵀ` reused across many A minibatches.
///
/// The partition pass multiplies every point minibatch against the SAME leader
/// matrix B (n×k). Packing B once into MKL's internal blocked layout and issuing
/// `cblas_sgemm_compute` per minibatch removes the repacking MKL otherwise does
/// on every plain `cblas_sgemm` call.
///
/// `pack` packs with `CblasTrans` so `op(B) = Bᵀ`; `compute` then runs
/// `NoTrans × Packed` to yield `A · Bᵀ` (m×n). The pack size is computed for the
/// `max_m` you intend to compute with; per MKL docs a smaller actual `m` reuses
/// the same packed buffer.
pub struct SgemmPackedB {
    /// 64-byte-aligned scratch holding the packed B. `data_off` is the f32 index
    /// of the aligned start; MKL writes/reads `[data_off..]`.
    buf: Vec<f32>,
    data_off: usize,
    n: usize,
    k: usize,
}

impl SgemmPackedB {
    /// Pack `b` (n×k row-major) for later `C = A·Bᵀ`, `m` up to `max_m`.
    pub fn pack(b: &[f32], n: usize, k: usize, max_m: usize) -> Self {
        debug_assert_eq!(b.len(), n * k);
        ensure_mkl_single_threaded();
        // SAFETY: pure size query, no pointers dereferenced.
        let bytes = unsafe {
            cblas_sgemm_pack_get_size(
                CBLAS_B_MATRIX,
                max_m as MklInt,
                n as MklInt,
                k as MklInt,
            )
        };
        // Round size up to whole f32s, plus 16 f32 (64 B) of head slack so we can
        // land the MKL pointer on a 64-byte boundary regardless of Vec alignment.
        let n_f32 = bytes.div_ceil(4) + 16;
        let mut buf = vec![0.0f32; n_f32];
        let addr = buf.as_ptr() as usize;
        let aligned = (addr + 63) & !63usize;
        let data_off = (aligned - addr) / 4;
        // SAFETY: op(B)=Bᵀ via CblasTrans; src is n×k row-major (ld=k); dest is the
        // aligned slot, sized by pack_get_size for (max_m,n,k). alpha=1.
        unsafe {
            cblas_sgemm_pack(
                CBLAS_ROW_MAJOR,
                CBLAS_B_MATRIX,
                CBLAS_TRANS,
                max_m as MklInt,
                n as MklInt,
                k as MklInt,
                1.0,
                b.as_ptr(),
                k as MklInt,
                buf[data_off..].as_mut_ptr(),
            );
        }
        SgemmPackedB { buf, data_off, n, k }
    }

    /// `C = A · Bᵀ` for `a` (m×k row-major) against the packed B. `c` is m×n.
    pub fn compute(&self, a: &[f32], m: usize, c: &mut [f32]) {
        debug_assert_eq!(a.len(), m * self.k);
        debug_assert_eq!(c.len(), m * self.n);
        ensure_mkl_single_threaded();
        // SAFETY: A m×k (lda=k); B is CblasPacked (the trans/op was baked in at
        // pack time, so transb=Packed and ldb is ignored — we pass k harmlessly);
        // C m×n (ldc=n). beta=0 overwrites C.
        unsafe {
            cblas_sgemm_compute(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_PACKED,
                m as MklInt,
                self.n as MklInt,
                self.k as MklInt,
                a.as_ptr(),
                self.k as MklInt,
                self.buf[self.data_off..].as_ptr(),
                self.k as MklInt,
                0.0,
                c.as_mut_ptr(),
                self.n as MklInt,
            );
        }
    }
}

// ─── JIT sgemm (per-shape generated kernel, cached per thread) ───────────────

/// Cached JIT sgemm kernel + its jitter handle for one (m,n,k) shape.
struct JitKernel {
    jitter: *mut std::ffi::c_void,
    kernel: unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, *const f32, *mut f32),
}

impl Drop for JitKernel {
    fn drop(&mut self) {
        // SAFETY: jitter came from a successful mkl_jit_create_sgemm; destroyed once.
        unsafe {
            mkl_jit_destroy(self.jitter);
        }
    }
}

thread_local! {
    /// Per-thread cache of JIT sgemm kernels keyed by (m,n,k). The partition/leaf
    /// reuse a tiny set of shapes, so this stays small. Thread-local because the
    /// generated kernel and its jitter are not shared across threads.
    static JIT_CACHE: std::cell::RefCell<std::collections::HashMap<(usize, usize, usize), JitKernel>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Get-or-create the JIT kernel for (m,n,k) doing `C = A·Bᵀ` (NoTrans×Trans,
/// alpha=1, beta=0). Returns `(jitter, kernel)` or `None` if MKL won't JIT it.
fn get_or_create_jit(
    m: usize,
    n: usize,
    k: usize,
) -> Option<(
    *mut std::ffi::c_void,
    unsafe extern "C" fn(*mut std::ffi::c_void, *const f32, *const f32, *mut f32),
)> {
    JIT_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(jk) = cache.get(&(m, n, k)) {
            return Some((jk.jitter, jk.kernel));
        }
        let mut jitter: *mut std::ffi::c_void = std::ptr::null_mut();
        // C = A·Bᵀ: NoTrans × Trans, lda=k, ldb=k (B is n×k), ldc=n.
        // SAFETY: out-param jitter; all dims/leading-dims valid for the call.
        let status = unsafe {
            mkl_cblas_jit_create_sgemm(
                &mut jitter as *mut *mut std::ffi::c_void,
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as MklInt,
                n as MklInt,
                k as MklInt,
                1.0,
                k as MklInt,
                k as MklInt,
                0.0,
                n as MklInt,
            )
        };
        // status: 0=JIT kernel, 1=no JIT (standard kernel still returned via the
        // ptr), 2=error. Only an error means we have nothing usable.
        if status == 2 {
            return None;
        }
        // SAFETY: jitter is valid after a non-error create.
        let ptr = unsafe { mkl_jit_get_sgemm_ptr(jitter) };
        match ptr {
            Some(kf) => {
                cache.insert((m, n, k), JitKernel { jitter, kernel: kf });
                Some((jitter, kf))
            }
            None => {
                // SAFETY: clean up the jitter we will not cache.
                unsafe {
                    mkl_jit_destroy(jitter);
                }
                None
            }
        }
    })
}

/// `C = A · Bᵀ` for f32 A (m×k), B (n×k) via a JIT-generated kernel specialized
/// to this (m,n,k) shape (NoTrans × Trans, alpha=1, beta=0). Falls back to the
/// standard `sgemm_abt` if MKL declines to JIT this shape.
pub fn jit_sgemm_abt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    ensure_mkl_single_threaded();
    match get_or_create_jit(m, n, k) {
        Some((jitter, kf)) => {
            // SAFETY: kernel matches (m,n,k); a/b/c sized to it. The kernel reads
            // op(A)=A (m×k) and op(B)=Bᵀ (B passed n×k) and writes C m×n.
            unsafe {
                kf(jitter, a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
            }
        }
        None => sgemm_abt(a, m, k, b, n, c),
    }
}

// ─── f64 (widened to f32 output) ─────────────────────────────────────────────

/// `C = A · Aᵀ` for `m × k` row-major f64 A, LOWER triangle only.
/// Result widened to f32 in `c`. Uses a scratch f64 buffer for the lower tri.
pub fn dsyrk_aat_lower_to_f32(a: &[f64], m: usize, k: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(c.len(), m * m);
    ensure_mkl_single_threaded();
    let mut tmp = vec![0.0f64; m * m];
    // SAFETY: pointers from live slices; row-major lda=k, ldc=m.
    unsafe {
        cblas_dsyrk(
            CBLAS_ROW_MAJOR,
            CBLAS_LOWER,
            CBLAS_NO_TRANS,
            m as MklInt,
            k as MklInt,
            1.0,
            a.as_ptr(),
            k as MklInt,
            0.0,
            tmp.as_mut_ptr(),
            m as MklInt,
        );
    }
    for i in 0..m {
        for j in 0..=i {
            c[i * m + j] = tmp[i * m + j] as f32;
        }
    }
}

/// `C = A · Bᵀ` for f64 A (m×k), B (n×k). Output widened to f32 (m×n).
pub fn dgemm_abt_to_f32(a: &[f64], m: usize, k: usize, b: &[f64], n: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    ensure_mkl_single_threaded();
    let mut tmp = vec![0.0f64; m * n];
    // C = A · Bᵀ via NoTrans × Trans (dgemm honors row-major CblasTrans). B passed
    // directly (n×k, ldb=k, op(B)=Bᵀ); NO per-call transpose/alloc.
    // SAFETY: A m×k (lda=k), B n×k (ldb=k, op(B)=Bᵀ), C m×n (ldc=n).
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as MklInt,
            n as MklInt,
            k as MklInt,
            1.0,
            a.as_ptr(),
            k as MklInt,
            b.as_ptr(),
            k as MklInt,
            0.0,
            tmp.as_mut_ptr(),
            n as MklInt,
        );
    }
    for (dst, &src) in c.iter_mut().zip(tmp.iter()) {
        *dst = src as f32;
    }
}

// ─── bf16 (AMX-bf16) ─────────────────────────────────────────────────────────

/// `C = A · Aᵀ` for `m × k` row-major bf16 A, f32 output. Writes the FULL
/// m×m matrix (bf16 GEMM has no SYRK variant); leaf consumer reads only the
/// lower triangle + diagonal, so the wasted upper-triangle stores are harmless.
pub fn bf16_aat_full(a: &[MklBf16], m: usize, k: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(c.len(), m * m);
    ensure_mkl_single_threaded();
    // C = A · Aᵀ via NoTrans × Trans (CblasTrans=112 honored row-major). A passed
    // directly as both operands (m×k, ldb=k, op(A)=Aᵀ); NO per-call transpose/alloc.
    // SAFETY: A m×k (lda=k), A m×k (ldb=k, op=Aᵀ → k×m), C m×m (ldc=m).
    unsafe {
        cblas_gemm_bf16bf16f32(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as MklInt,
            m as MklInt,
            k as MklInt,
            1.0,
            a.as_ptr(),
            k as MklInt,
            a.as_ptr(),
            k as MklInt,
            0.0,
            c.as_mut_ptr(),
            m as MklInt,
        );
    }
}

/// `C = A · Bᵀ` for bf16 A (m×k), B (n×k), f32 output (m×n).
pub fn bf16_abt(a: &[MklBf16], m: usize, k: usize, b: &[MklBf16], n: usize, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    ensure_mkl_single_threaded();
    // C = A · Bᵀ via NoTrans × Trans (CblasTrans=112 honored row-major, same fix
    // as the u8/f32 paths). B passed directly (n×k, ldb=k, op(B)=Bᵀ); NO per-call
    // transpose/alloc.
    // SAFETY: A m×k (lda=k), B n×k (ldb=k, op(B)=Bᵀ), C m×n (ldc=n).
    unsafe {
        cblas_gemm_bf16bf16f32(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as MklInt,
            n as MklInt,
            k as MklInt,
            1.0,
            a.as_ptr(),
            k as MklInt,
            b.as_ptr(),
            k as MklInt,
            0.0,
            c.as_mut_ptr(),
            n as MklInt,
        );
    }
}

// ─── u8 / i8 (AMX-int8, centered) ────────────────────────────────────────────

thread_local! {
    /// Reusable per-thread Bᵀ transpose scratch for the NoTrans×NoTrans
    /// `cblas_gemm_s8u8s32` call (oneMKL 2026.0 rejects TransB=Trans in
    /// row-major). Avoids a `vec![0i8; k*n]` alloc on every leaf/chunk GEMM
    /// (~1.27M leaves at BigANN 10M → that per-call alloc dominated the leaf
    /// phase and ate the AMX-int8 win). Grows monotonically.
    static BT_SCRATCH: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Compute the per-row sum `colsum[j] = Σ_l x[j*k + l]` over a row-major
/// `n × k` u8 matrix. Used to correct the centered s8×u8 product.
#[inline]
pub fn u8_row_sums(x: &[u8], n: usize, k: usize, out: &mut [i64]) {
    debug_assert_eq!(x.len(), n * k);
    debug_assert_eq!(out.len(), n);
    for j in 0..n {
        let row = &x[j * k..j * k + k];
        let mut s: i64 = 0;
        for &v in row {
            s += v as i64;
        }
        out[j] = s;
    }
}

/// Center a row-major u8 matrix into s8: `b' = b - 128 ∈ [-128, 127]`. This is
/// applied to the **B operand** of `cblas_gemm_s8u8s32` (A stays raw u8).
#[inline]
pub fn u8_center_to_s8(x: &[u8], out: &mut [i8]) {
    debug_assert_eq!(x.len(), out.len());
    for (d, &v) in out.iter_mut().zip(x.iter()) {
        *d = (v as i32 - 128) as i8;
    }
}

/// Raw u8×s8→s32 GEMM with `ao=bo=0`, `OffsetC=Fix, co=0` (no MKL offset).
/// `C = A · Bᵀ` for A (m×k), B (n×k). MKL convention: operand A is u8, operand
/// B is s8.
///
/// IMPORTANT: oneMKL 2026.0's `cblas_gemm_s8u8s32` rejects `TransB=Trans` in
/// row-major (errors "Parameter 3 incorrect"; only NoTrans×NoTrans is honored).
/// So we materialize Bᵀ as a `k×n` row-major scratch and issue NoTrans×NoTrans,
/// which `MKL_VERBOSE` confirms yields the correct s32. The transpose scratch
/// is supplied by the caller (`b_t`, length `k*n`) to avoid per-call alloc.
///
/// Caller supplies raw u8 A and centered s8 B; the `+128·rowsum_a[i]`
/// correction (per output row) is applied separately.
#[inline]
fn u8s8s32_abt_raw(
    a_u8: &[u8],
    m: usize,
    k: usize,
    b_s8: &[i8],
    n: usize,
    b_t: &mut [i8],
    c: &mut [i32],
) {
    debug_assert_eq!(a_u8.len(), m * k);
    debug_assert_eq!(b_s8.len(), n * k);
    debug_assert_eq!(b_t.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    // TransB=Trans IS honored (CblasTrans=112) — op(B)=Bᵀ directly, B=b_s8 (n×k,
    // ldb=k). NO per-leaf transpose. (The prior "rejected" was CBLAS_TRANS being
    // mis-defined as 121=CblasUpper; fixed to 112.)
    let _ = b_t;
    ensure_mkl_single_threaded();
    let co: [i32; 1] = [0];
    // SAFETY: A is m×k (lda=k); B is n×k row-major (ldb=k), TransB → op(B)=Bᵀ (k×n);
    // C is m×n (ldc=n). C = A · Bᵀ. A=u8 (raw), B=s8 (centered).
    unsafe {
        cblas_gemm_s8u8s32(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            CBLAS_FIX_OFFSET,
            m as MklInt,
            n as MklInt,
            k as MklInt,
            1.0,
            a_u8.as_ptr() as *const std::ffi::c_void,
            k as MklInt,
            0,
            b_s8.as_ptr() as *const std::ffi::c_void,
            k as MklInt,
            0,
            0.0,
            c.as_mut_ptr(),
            n as MklInt,
            co.as_ptr(),
        );
    }
}

/// `C = A · Aᵀ` (LOWER triangle + diagonal) for an `m × k` row-major u8 matrix,
/// exact integer Gram converted to f32. `a_u8` is the raw A (operand A = u8),
/// `a_s8` is the centered A (`a-128`, operand B = s8), `rowsum[i] = Σ_l a[i][l]`.
/// MKL computes `s32[i][j] = Σ A[i]·(A[j]-128) = ΣA·A - 128·rowsum[i]`, so we
/// recover `true[i][j] = s32[i][j] + 128·rowsum[i]` (per OUTPUT ROW i) over the
/// lower triangle. The diagonal yields `‖a_i‖²` exactly.
///
/// The s32 scratch (`m × m`) is supplied by the caller to avoid per-call alloc.
pub fn gram_u8_aat_lower(
    a_u8: &[u8],
    a_s8: &[i8],
    m: usize,
    k: usize,
    rowsum: &[i64],
    s32_scratch: &mut [i32],
    c: &mut [f32],
) {
    debug_assert_eq!(a_u8.len(), m * k);
    debug_assert_eq!(a_s8.len(), m * k);
    debug_assert_eq!(rowsum.len(), m);
    debug_assert_eq!(s32_scratch.len(), m * m);
    debug_assert_eq!(c.len(), m * m);
    // Full A·Aᵀ in s32 (u8 A × s8 A). We only read the lower triangle below.
    // B-transpose scratch (k×m) for the NoTrans×NoTrans s8u8s32 call (reused
    // per-thread to avoid a per-leaf alloc).
    BT_SCRATCH.with(|cell| {
        let mut b_t = cell.borrow_mut();
        if b_t.len() < k * m {
            b_t.resize(k * m, 0);
        }
        u8s8s32_abt_raw(a_u8, m, k, a_s8, m, &mut b_t[..k * m], s32_scratch);
    });
    for i in 0..m {
        let base = i * m;
        // f32-broadcast correction: dot <= 255^2*128 = 8.32M and 128*rowsum
        // <= 4.18M, sum < 12.5M < 2^24 → exact in f32. The i64 intermediate
        // (prior code) blocked AVX-512 autovectorization of this triangular
        // loop; the f32 add over the [0..=i] row slice vectorizes 16-wide.
        let corr = (128 * rowsum[i]) as f32; // per-row correction
        let row_s = &s32_scratch[base..base + i + 1];
        let row_c = &mut c[base..base + i + 1];
        for (d, &v) in row_c.iter_mut().zip(row_s.iter()) {
            *d = v as f32 + corr;
        }
    }
}

/// `C = A · Bᵀ` for u8 points A (m×k) and u8 leaders B (n×k), exact integer
/// Gram converted to f32 (m×n full tile). `a_u8` is raw points (operand A=u8),
/// `b_s8` is centered leaders (`b-128`, operand B=s8), `rowsum_a[i] = Σ_l a[i][l]`
/// (POINTS row-sums). MKL computes `s32[i][j] = Σ A[i]·(B[j]-128)`, so
/// `true[i][j] = s32[i][j] + 128·rowsum_a[i]` (per OUTPUT ROW i).
///
/// The s32 scratch (`m × n`) is supplied by the caller.
pub fn gram_u8_abt(
    a_u8: &[u8],
    m: usize,
    k: usize,
    b_s8: &[i8],
    n: usize,
    rowsum_a: &[i64],
    s32_scratch: &mut [i32],
    c: &mut [f32],
) {
    debug_assert_eq!(a_u8.len(), m * k);
    debug_assert_eq!(b_s8.len(), n * k);
    debug_assert_eq!(rowsum_a.len(), m);
    debug_assert_eq!(s32_scratch.len(), m * n);
    debug_assert_eq!(c.len(), m * n);
    // B-transpose scratch (k×n) for the NoTrans×NoTrans s8u8s32 call (reused
    // per-thread to avoid a per-chunk alloc).
    BT_SCRATCH.with(|cell| {
        let mut b_t = cell.borrow_mut();
        if b_t.len() < k * n {
            b_t.resize(k * n, 0);
        }
        u8s8s32_abt_raw(a_u8, m, k, b_s8, n, &mut b_t[..k * n], s32_scratch);
    });
    for i in 0..m {
        let base = i * n;
        // f32-broadcast correction (mirrors gram_u8_aat_lower): dot <= 255^2*128
        // = 8.32M and 128*rowsum <= 4.18M, sum < 12.5M < 2^24 -> exact in f32.
        // The i64 intermediate (prior code) forced the autovectorizer onto the
        // 8-lane vcvtqq2ps path; f32 add of a broadcast corr vectorizes 16-wide.
        let corr = (128 * rowsum_a[i]) as f32; // per-row (points) correction
        let row_s = &s32_scratch[base..base + n];
        let row_c = &mut c[base..base + n];
        for (d, &v) in row_c.iter_mut().zip(row_s.iter()) {
            *d = v as f32 + corr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    // ── Scalar integer reference Gram ────────────────────────────────────────

    fn ref_gram_u8_full(a: &[u8], m: usize, k: usize, b: &[u8], n: usize) -> Vec<i64> {
        let mut c = vec![0i64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0i64;
                for l in 0..k {
                    s += a[i * k + l] as i64 * b[j * k + l] as i64;
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn rand_u8(rng: &mut StdRng, n: usize) -> Vec<u8> {
        (0..n).map(|_| rng.random::<u8>()).collect()
    }

    // ── u8 SYRK (A·Aᵀ lower + diagonal): full [0,255] range, bit-exact ───────

    fn check_u8_syrk(m: usize, k: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let a_u8 = rand_u8(&mut rng, m * k);
        let mut a_s8 = vec![0i8; m * k];
        u8_center_to_s8(&a_u8, &mut a_s8);
        let mut rowsum = vec![0i64; m];
        u8_row_sums(&a_u8, m, k, &mut rowsum);

        let mut s32 = vec![0i32; m * m];
        let mut got = vec![0.0f32; m * m];
        // A operand = raw u8, B operand = centered s8, per-row correction.
        gram_u8_aat_lower(&a_u8, &a_s8, m, k, &rowsum, &mut s32, &mut got);

        let reference = ref_gram_u8_full(&a_u8, m, k, &a_u8, m);
        for i in 0..m {
            for j in 0..=i {
                let exp = reference[i * m + j];
                // u8 SIFT Gram ≤ 255²·k; exact in f32 for k≤512 (< 2^24).
                let got_int = got[i * m + j] as i64;
                assert_eq!(
                    got_int, exp,
                    "u8 SYRK mismatch at ({i},{j}) m={m} k={k}: got {got_int} expected {exp}"
                );
            }
        }
        // Diagonal is ‖a_i‖².
        for i in 0..m {
            let exp = reference[i * m + i];
            assert_eq!(got[i * m + i] as i64, exp, "u8 SYRK diagonal at {i}");
        }
    }

    #[test]
    fn u8_syrk_bit_exact_full_range() {
        check_u8_syrk(64, 96, 1);
        check_u8_syrk(64, 128, 2);
        check_u8_syrk(256, 96, 3);
        check_u8_syrk(256, 128, 4);
        check_u8_syrk(512, 128, 5);
    }

    // Mirror the standalone C harness exactly: m=2,n=2,k=4, A=raw u8, B=s8.
    // C harness produced s32 row0 = [-31070, -58535]. This isolates whether
    // the Rust FFI binding to cblas_gemm_s8u8s32 matches C.
    #[test]
    fn u8_raw_s32_matches_c_harness() {
        let a_u8: [u8; 8] = [200, 10, 255, 0, 128, 128, 128, 128];
        let b_u8: [u8; 8] = [50, 60, 70, 80, 1, 2, 3, 4];
        let mut b_s8 = [0i8; 8];
        u8_center_to_s8(&b_u8, &mut b_s8);

        // Direct N,N call (B pre-transposed to k×n so NoTrans gives A·Bᵀ).
        // b_s8 is 2×4 row-major; its transpose (4×2) for NoTrans GEMM.
        let mut b_t = [0i8; 8];
        for j in 0..2 {
            for l in 0..4 {
                b_t[l * 2 + j] = b_s8[j * 4 + l];
            }
        }
        let mut s32nn = [0i32; 4];
        let co = [0i32; 1];
        ensure_mkl_single_threaded();
        // SAFETY: test-only direct FFI to probe trans-flag acceptance.
        unsafe {
            cblas_gemm_s8u8s32(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                CBLAS_FIX_OFFSET,
                2,
                2,
                4,
                1.0,
                a_u8.as_ptr() as *const std::ffi::c_void,
                4,
                0,
                b_t.as_ptr() as *const std::ffi::c_void,
                2,
                0,
                0.0,
                s32nn.as_mut_ptr(),
                2,
                co.as_ptr(),
            );
        }
        eprintln!("RUST s32 (N,N) = {s32nn:?}  (expect [-31070, -58535, -32256, -64256])");

        let mut s32 = [0i32; 4];
        let mut bt = [0i8; 8];
        u8s8s32_abt_raw(&a_u8, 2, 4, &b_s8, 2, &mut bt, &mut s32);
        eprintln!("RUST s32 (via raw, NoTrans transpose) = {s32:?}  (expects [-31070, -58535, -32256, -64256])");
        assert_eq!(s32[0], -31070, "Rust FFI s32[0] != C harness");
    }

    // ── u8 general A·Bᵀ (partition): full [0,255] both operands, bit-exact ───

    #[test]
    fn u8_abt_bit_exact_full_range() {
        let (m, n, k) = (512usize, 256usize, 128usize);
        let mut rng = StdRng::seed_from_u64(42);
        let a_u8 = rand_u8(&mut rng, m * k); // points (operand A = raw u8)
        let b_u8 = rand_u8(&mut rng, n * k); // leaders
        let mut b_s8 = vec![0i8; n * k];
        u8_center_to_s8(&b_u8, &mut b_s8); // leaders centered (operand B = s8)
        let mut rowsum_a = vec![0i64; m];
        u8_row_sums(&a_u8, m, k, &mut rowsum_a); // POINTS row-sums

        let mut s32 = vec![0i32; m * n];
        let mut got = vec![0.0f32; m * n];
        gram_u8_abt(&a_u8, m, k, &b_s8, n, &rowsum_a, &mut s32, &mut got);

        let reference = ref_gram_u8_full(&a_u8, m, k, &b_u8, n);
        for i in 0..m {
            for j in 0..n {
                let exp = reference[i * n + j];
                let got_int = got[i * n + j] as i64;
                assert_eq!(
                    got_int, exp,
                    "u8 A·Bᵀ mismatch at ({i},{j}): got {got_int} expected {exp}"
                );
            }
        }
    }

    // Edge case: deliberately include 0, 127, 128, 255 to stress the s8 alias.
    #[test]
    fn u8_syrk_boundary_values() {
        let (m, k) = (8usize, 16usize);
        let pattern: [u8; 16] = [0, 1, 127, 128, 129, 254, 255, 200, 64, 192, 255, 0, 128, 100, 50, 255];
        let mut a_u8 = vec![0u8; m * k];
        for i in 0..m {
            for l in 0..k {
                a_u8[i * k + l] = pattern[(i + l) % 16];
            }
        }
        let mut a_s8 = vec![0i8; m * k];
        u8_center_to_s8(&a_u8, &mut a_s8);
        let mut rowsum = vec![0i64; m];
        u8_row_sums(&a_u8, m, k, &mut rowsum);
        let mut s32 = vec![0i32; m * m];
        let mut got = vec![0.0f32; m * m];
        gram_u8_aat_lower(&a_u8, &a_s8, m, k, &rowsum, &mut s32, &mut got);
        let reference = ref_gram_u8_full(&a_u8, m, k, &a_u8, m);
        for i in 0..m {
            for j in 0..=i {
                assert_eq!(got[i * m + j] as i64, reference[i * m + j], "boundary ({i},{j})");
            }
        }
    }

    // ── f32 ssyrk + sgemm vs scalar reference (exact-ish) ────────────────────

    #[test]
    fn f32_ssyrk_matches_reference() {
        let (m, k) = (128usize, 96usize);
        let mut rng = StdRng::seed_from_u64(7);
        let a: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let mut got = vec![0.0f32; m * m];
        ssyrk_aat_lower(&a, m, k, &mut got);
        for i in 0..m {
            for j in 0..=i {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * a[j * k + l];
                }
                let diff = (got[i * m + j] - s).abs();
                assert!(diff < 1e-3, "f32 ssyrk ({i},{j}) got {} exp {s}", got[i * m + j]);
            }
        }
    }

    #[test]
    fn f32_sgemm_abt_matches_reference() {
        let (m, n, k) = (96usize, 64usize, 80usize);
        let mut rng = StdRng::seed_from_u64(11);
        let a: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let mut got = vec![0.0f32; m * n];
        sgemm_abt(&a, m, k, &b, n, &mut got);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[j * k + l];
                }
                let diff = (got[i * n + j] - s).abs();
                assert!(diff < 1e-3, "f32 sgemm ({i},{j})");
            }
        }
    }

    // ── bf16 both ops vs f32 reference within bf16 tolerance ─────────────────

    fn f32_to_bf16(x: f32) -> u16 {
        // Round-to-nearest-even bf16 truncation of the f32 bits.
        let bits = x.to_bits();
        let lsb = (bits >> 16) & 1;
        let round = 0x7fff + lsb;
        ((bits.wrapping_add(round)) >> 16) as u16
    }

    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }

    #[test]
    fn bf16_aat_matches_reference() {
        let (m, k) = (64usize, 128usize);
        let mut rng = StdRng::seed_from_u64(13);
        let af: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let abf: Vec<u16> = af.iter().map(|&v| f32_to_bf16(v)).collect();
        let arec: Vec<f32> = abf.iter().map(|&v| bf16_to_f32(v)).collect();
        let mut got = vec![0.0f32; m * m];
        bf16_aat_full(&abf, m, k, &mut got);
        for i in 0..m {
            for j in 0..=i {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += arec[i * k + l] * arec[j * k + l];
                }
                // bf16 has ~8 bits mantissa; accumulation in f32. Loose tol.
                let tol = 1e-1 * (s.abs() + 1.0);
                let diff = (got[i * m + j] - s).abs();
                assert!(diff < tol, "bf16 aat ({i},{j}) got {} exp {s} diff {diff}", got[i * m + j]);
            }
        }
    }

    #[test]
    fn bf16_abt_matches_reference() {
        let (m, n, k) = (64usize, 48usize, 128usize);
        let mut rng = StdRng::seed_from_u64(17);
        let af: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let bf: Vec<f32> = (0..n * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let abf: Vec<u16> = af.iter().map(|&v| f32_to_bf16(v)).collect();
        let bbf: Vec<u16> = bf.iter().map(|&v| f32_to_bf16(v)).collect();
        let arec: Vec<f32> = abf.iter().map(|&v| bf16_to_f32(v)).collect();
        let brec: Vec<f32> = bbf.iter().map(|&v| bf16_to_f32(v)).collect();
        let mut got = vec![0.0f32; m * n];
        bf16_abt(&abf, m, k, &bbf, n, &mut got);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += arec[i * k + l] * brec[j * k + l];
                }
                let tol = 1e-1 * (s.abs() + 1.0);
                let diff = (got[i * n + j] - s).abs();
                assert!(diff < tol, "bf16 abt ({i},{j}) diff {diff}");
            }
        }
    }

    // ── f64 dsyrk widened to f32 ─────────────────────────────────────────────

    #[test]
    fn f64_dgemm_abt_matches_reference() {
        let (m, n, k) = (48usize, 40usize, 56usize);
        let mut rng = StdRng::seed_from_u64(23);
        let a: Vec<f64> = (0..m * k).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let b: Vec<f64> = (0..n * k).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let mut got = vec![0.0f32; m * n];
        dgemm_abt_to_f32(&a, m, k, &b, n, &mut got);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f64;
                for l in 0..k {
                    s += a[i * k + l] * b[j * k + l];
                }
                let diff = (got[i * n + j] as f64 - s).abs();
                assert!(diff < 1e-4, "f64 dgemm abt ({i},{j}) got {} exp {s}", got[i * n + j]);
            }
        }
    }

    #[test]
    fn f64_dsyrk_matches_reference() {
        let (m, k) = (48usize, 40usize);
        let mut rng = StdRng::seed_from_u64(19);
        let a: Vec<f64> = (0..m * k).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let mut got = vec![0.0f32; m * m];
        dsyrk_aat_lower_to_f32(&a, m, k, &mut got);
        for i in 0..m {
            for j in 0..=i {
                let mut s = 0.0f64;
                for l in 0..k {
                    s += a[i * k + l] * a[j * k + l];
                }
                let diff = (got[i * m + j] as f64 - s).abs();
                assert!(diff < 1e-4, "f64 dsyrk ({i},{j})");
            }
        }
    }

    // ── Packed sgemm: pack B once, A·Bᵀ matches sgemm_abt bit-close ──────────
    #[test]
    fn sgemm_packed_b_matches_sgemm_abt() {
        let (n, k, max_m) = (256usize, 96usize, 512usize);
        let mut rng = StdRng::seed_from_u64(101);
        let b: Vec<f32> = (0..n * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let packed = SgemmPackedB::pack(&b, n, k, max_m);
        // Two minibatch sizes to exercise "smaller m reuses the same packed B".
        for m in [max_m, 64usize, 1usize] {
            let a: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let mut got = vec![0.0f32; m * n];
            packed.compute(&a, m, &mut got);
            let mut reference = vec![0.0f32; m * n];
            sgemm_abt(&a, m, k, &b, n, &mut reference);
            for i in 0..m * n {
                let diff = (got[i] - reference[i]).abs();
                let tol = 1e-3 * (reference[i].abs() + 1.0);
                assert!(diff < tol, "packed vs sgemm_abt at {i} m={m}: got {} exp {} diff {diff}", got[i], reference[i]);
            }
        }
    }

    // ── JIT sgemm: A·Bᵀ matches scalar reference (or clean fallback) ─────────
    #[test]
    fn jit_sgemm_abt_matches_reference() {
        let (m, n, k) = (96usize, 64usize, 80usize);
        let mut rng = StdRng::seed_from_u64(103);
        let a: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let b: Vec<f32> = (0..n * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let mut got = vec![0.0f32; m * n];
        // Call twice to exercise the get-then-cache path.
        jit_sgemm_abt(&a, m, k, &b, n, &mut got);
        jit_sgemm_abt(&a, m, k, &b, n, &mut got);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[j * k + l];
                }
                let diff = (got[i * n + j] - s).abs();
                assert!(diff < 1e-3, "jit sgemm ({i},{j}) got {} exp {s}", got[i * n + j]);
            }
        }
    }
}
