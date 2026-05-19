/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

fn main() {
    if std::env::var("CARGO_FEATURE_MKL_FP16").is_err() {
        return;
    }
    let mkl_root = std::env::var("MKLROOT")
        .unwrap_or_else(|_| "/opt/intel/oneapi/mkl/latest".to_string());
    println!("cargo:rustc-link-search=native={}/lib", mkl_root);
    println!("cargo:rustc-link-search=native={}/lib/intel64", mkl_root);
    // Single-dynamic-library: mkl_rt routes through the right interface/threading.
    // Honors MKL_INTERFACE_LAYER and MKL_THREADING_LAYER env vars at runtime.
    println!("cargo:rustc-link-lib=mkl_rt");
    println!("cargo:rerun-if-env-changed=MKLROOT");
}
