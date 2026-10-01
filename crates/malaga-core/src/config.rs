//! Hyper-parameters of an NLLB / M2M100 checkpoint.
//!
//! The same struct is read from a Hugging Face `config.json` (at conversion
//! time) and from the GGUF metadata (at inference time), so both sides always
//! agree on the architecture.

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use serde::{Deserialize, Serialize};

/// GGUF `general.architecture` value written by `malaga convert`.
pub const ARCH: &str = "nllb";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_true")]
    pub scale_embedding: bool,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default = "default_pad")]
    pub pad_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
    #[serde(default = "default_eos")]
    pub decoder_start_token_id: u32,
    #[serde(default = "default_activation")]
    pub activation_function: String,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
}

fn default_true() -> bool {
    true
}
fn default_pad() -> u32 {
    1
}
fn default_eos() -> u32 {
    2
}
fn default_activation() -> String {
    "relu".to_string()
}
fn default_max_length() -> usize {
    200
}

impl Config {
    pub const LAYER_NORM_EPS: f64 = 1e-5;

    pub fn from_hf_json(json: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(json).context("parsing config.json")?;
        anyhow::ensure!(cfg.activation_function == "relu", "unsupported activation {}", cfg.activation_function);
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    /// Metadata entries stored in the GGUF file.
    pub fn to_gguf(&self) -> Vec<(String, Value)> {
        let u = |v: usize| Value::U32(v as u32);
        let k = |s: &str| format!("{ARCH}.{s}");
        vec![
            (k("vocab_size"), u(self.vocab_size)),
            (k("embedding_length"), u(self.d_model)),
            (k("encoder.block_count"), u(self.encoder_layers)),
            (k("decoder.block_count"), u(self.decoder_layers)),
            (k("encoder.attention.head_count"), u(self.encoder_attention_heads)),
            (k("decoder.attention.head_count"), u(self.decoder_attention_heads)),
            (k("encoder.feed_forward_length"), u(self.encoder_ffn_dim)),
            (k("decoder.feed_forward_length"), u(self.decoder_ffn_dim)),
            (k("context_length"), u(self.max_position_embeddings)),
            (k("scale_embedding"), Value::Bool(self.scale_embedding)),
            (k("max_length"), u(self.max_length)),
            ("tokenizer.ggml.bos_token_id".into(), Value::U32(self.bos_token_id)),
            ("tokenizer.ggml.padding_token_id".into(), Value::U32(self.pad_token_id)),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(self.eos_token_id)),
            ("tokenizer.ggml.decoder_start_token_id".into(), Value::U32(self.decoder_start_token_id)),
        ]
    }

    pub fn from_gguf(ct: &Content) -> Result<Self> {
        let arch = get(ct, "general.architecture")?.to_string()?.clone();
        anyhow::ensure!(arch == ARCH, "unsupported architecture '{arch}', expected '{ARCH}'");
        let u = |key: &str| -> Result<usize> { Ok(get(ct, key)?.to_u32()? as usize) };
        let k = |s: &str| format!("{ARCH}.{s}");
        Ok(Self {
            vocab_size: u(&k("vocab_size"))?,
            d_model: u(&k("embedding_length"))?,
            encoder_layers: u(&k("encoder.block_count"))?,
            decoder_layers: u(&k("decoder.block_count"))?,
            encoder_attention_heads: u(&k("encoder.attention.head_count"))?,
            decoder_attention_heads: u(&k("decoder.attention.head_count"))?,
            encoder_ffn_dim: u(&k("encoder.feed_forward_length"))?,
            decoder_ffn_dim: u(&k("decoder.feed_forward_length"))?,
            max_position_embeddings: u(&k("context_length"))?,
            scale_embedding: get(ct, &k("scale_embedding"))?.to_bool()?,
            max_length: u(&k("max_length"))?,
            bos_token_id: u("tokenizer.ggml.bos_token_id")? as u32,
            pad_token_id: u("tokenizer.ggml.padding_token_id")? as u32,
            eos_token_id: u("tokenizer.ggml.eos_token_id")? as u32,
            decoder_start_token_id: u("tokenizer.ggml.decoder_start_token_id")? as u32,
            activation_function: default_activation(),
        })
    }
}

pub(crate) fn get<'a>(ct: &'a Content, key: &str) -> Result<&'a Value> {
    ct.metadata.get(key).with_context(|| format!("missing GGUF metadata key '{key}'"))
}
