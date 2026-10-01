//! # malaga-core
//!
//! Fast French/English -> Malagasy translation with Meta's NLLB-200 models
//! running on quantized GGUF weights (CPU, CUDA or Metal through candle).
//!
//! ```no_run
//! use malaga_core::{Device, GenOptions, Translator};
//! let t = Translator::load("models/nllb-200-distilled-600M-q8_0.gguf", &Device::Cpu)?;
//! let mg = t.translate("Bonjour tout le monde.", "fr", "mg", &GenOptions::default())?;
//! # anyhow::Ok(())
//! ```

pub mod config;
pub mod convert;
mod graph;
#[cfg(feature = "cuda")]
mod kernels;
pub mod langs;
pub mod model;
pub mod segment;
#[doc(hidden)]
pub mod testutil;
pub mod translator;

pub use candle_core::Device;
pub use config::Config;
pub use translator::{GenOptions, Translator};

/// Picks the best available device: CUDA, then Metal, then CPU.
pub fn best_device(ordinal: usize) -> anyhow::Result<Device> {
    if candle_core::utils::cuda_is_available() {
        return Ok(Device::new_cuda(ordinal)?);
    }
    if candle_core::utils::metal_is_available() {
        return Ok(Device::new_metal(ordinal)?);
    }
    Ok(Device::Cpu)
}
