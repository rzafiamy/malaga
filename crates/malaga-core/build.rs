//! Compiles `kernels/malaga.cu` to PTX when the `cuda` feature is enabled.
//! The PTX targets `compute_$CUDA_COMPUTE_CAP` (default 70) and is JIT-compiled
//! by the driver for newer GPUs.

fn main() {
    println!("cargo:rerun-if-changed=kernels/malaga.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("malaga.ptx");
    let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "70".into());
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let status = std::process::Command::new(&nvcc)
        .args(["--ptx", "-O3", "--use_fast_math", "-std=c++17"])
        .arg(format!("-arch=compute_{cap}"))
        .arg("kernels/malaga.cu")
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc} (set NVCC or add CUDA to PATH): {e}"));
    assert!(status.success(), "nvcc failed to compile kernels/malaga.cu");
}
