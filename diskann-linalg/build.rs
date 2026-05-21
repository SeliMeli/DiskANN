/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build script: link libxsmm static library when the `libxsmm` feature is enabled.
//!
//! Requires `libxsmm-dev` installed system-wide (provides `/usr/lib/libxsmm.a`,
//! `/usr/lib/libxsmmext.a`, and `/usr/include/libxsmm.h`).

fn main() {
    if std::env::var("CARGO_FEATURE_LIBXSMM").is_ok() {
        // libxsmm static archives shipped by the Ubuntu `libxsmm-dev` package.
        for path in ["/usr/lib", "/usr/lib/x86_64-linux-gnu"] {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={path}");
            }
        }
        // Order matters: xsmmext depends on xsmm, xsmm references BLAS fallback
        // symbols (sgemm_, dgemm_, sgemv_, dgemv_) that are satisfied by
        // libxsmmnoblas (stubs that never get called because we always JIT).
        println!("cargo:rustc-link-lib=static=xsmmext");
        println!("cargo:rustc-link-lib=static=xsmm");
        println!("cargo:rustc-link-lib=static=xsmmnoblas");
        // libxsmm internally needs pthread (worker pool) and libdl (JIT mmap),
        // libm (math), and libstdc++ (C++ helpers in libxsmmext).
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rerun-if-changed=build.rs");
    }
}
