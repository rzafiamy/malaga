//! Hugging Face (PyTorch `.bin` or `.safetensors`) -> GGUF conversion.
//!
//! Besides re-encoding the weights, the converter bakes a few inference-time
//! optimisations into the file so that the runtime has less work to do:
//!
//! * `q_proj`, `k_proj`, `v_proj` of every self-attention block are fused into
//!   a single `attn_qkv` matrix (one matmul / kernel launch instead of three).
//! * The cross-attention `k_proj`/`v_proj` are fused into `cross_attn_kv`.
//! * The `1/sqrt(head_dim)` attention scaling is folded into the query weights
//!   and bias, removing one elementwise op per attention call.
//! * The shared embedding is stored once and reused for the LM head.
//! * The tokenizer (`tokenizer.json`) is embedded so a single `.gguf` file is
//!   all that is needed to run or host the model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};

use crate::config::{Config, ARCH};

/// Quantization presets. The `_m` presets keep the most sensitive tensors
/// (LM head / embeddings and FFN down projections) at higher precision,
/// mirroring llama.cpp's `Q4_K_M` / `Q5_K_M` recipes. Every preset offered here
/// scores the same as f32 on FLORES-200 (paired bootstrap, p > 0.05); `q4_0`
/// was removed because it measurably degrades translations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    F32,
    F16,
    Q8_0,
    Q6K,
    Q5KM,
    Q4KM,
}

impl std::str::FromStr for Preset {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" => Self::F32,
            "f16" => Self::F16,
            "q8_0" => Self::Q8_0,
            "q6_k" => Self::Q6K,
            "q5_k_m" => Self::Q5KM,
            "q4_k_m" => Self::Q4KM,
            _ => bail!("unknown preset '{s}' (f32, f16, q8_0, q6_k, q5_k_m, q4_k_m)"),
        })
    }
}

impl Preset {
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q6K => "q6_k",
            Self::Q5KM => "q5_k_m",
            Self::Q4KM => "q4_k_m",
        }
    }

    fn linear(&self) -> GgmlDType {
        match self {
            Self::F32 => GgmlDType::F32,
            Self::F16 => GgmlDType::F16,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q6K => GgmlDType::Q6K,
            Self::Q5KM => GgmlDType::Q5K,
            Self::Q4KM => GgmlDType::Q4K,
        }
    }

    fn ffn_down(&self) -> GgmlDType {
        match self {
            Self::Q5KM | Self::Q4KM => GgmlDType::Q6K,
            p => p.linear(),
        }
    }

    fn embedding(&self) -> GgmlDType {
        match self {
            Self::Q5KM | Self::Q4KM => GgmlDType::Q6K,
            p => p.linear(),
        }
    }
}

/// Loads every tensor of a Hugging Face checkpoint directory on the CPU.
pub(crate) fn load_checkpoint(dir: &Path) -> Result<HashMap<String, Tensor>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            name.ends_with(".safetensors") || (name.starts_with("pytorch_model") && name.ends_with(".bin"))
        })
        .collect();
    files.sort();
    // Prefer safetensors when both are present.
    if files.iter().any(|p| p.extension().is_some_and(|e| e == "safetensors")) {
        files.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
    }
    anyhow::ensure!(!files.is_empty(), "no weights found in {}", dir.display());

    let mut out = HashMap::new();
    for f in files {
        tracing::info!("reading {}", f.display());
        if f.extension().is_some_and(|e| e == "safetensors") {
            out.extend(candle_core::safetensors::load(&f, &Device::Cpu)?);
        } else {
            out.extend(candle_core::pickle::read_all(&f)?);
        }
    }
    Ok(out)
}

struct Writer {
    preset: Preset,
    tensors: Vec<(String, QTensor)>,
    bytes: usize,
}

impl Writer {
    fn add(&mut self, name: String, t: &Tensor, dtype: GgmlDType) -> Result<()> {
        let t = t.to_dtype(DType::F32)?.contiguous()?;
        let dtype = fallback_dtype(dtype, t.dims());
        let q = QTensor::quantize(&t, dtype).with_context(|| format!("quantizing {name}"))?;
        tracing::debug!("{name:40} {:?} {dtype:?}", t.dims());
        self.bytes += q.storage_size_in_bytes();
        self.tensors.push((name, q));
        Ok(())
    }

    /// Norm weights and biases are tiny and precision sensitive: always f32.
    fn add_f32(&mut self, name: String, t: &Tensor) -> Result<()> {
        self.add(name, t, GgmlDType::F32)
    }

    fn linear(&self) -> GgmlDType {
        self.preset.linear()
    }
}

/// k-quants need rows that are a multiple of 256 elements, legacy quants a
/// multiple of 32. Fall back to a compatible type when that does not hold.
fn fallback_dtype(dtype: GgmlDType, dims: &[usize]) -> GgmlDType {
    let row = *dims.last().unwrap_or(&0);
    if dims.len() < 2 {
        return GgmlDType::F32;
    }
    if row.is_multiple_of(dtype.block_size()) {
        dtype
    } else if row.is_multiple_of(32) {
        GgmlDType::Q8_0
    } else {
        GgmlDType::F16
    }
}

pub struct ConvertOptions {
    pub preset: Preset,
    pub model_name: String,
    /// `(language, corpus)` pairs: for each target language, the tokens seen in
    /// the corpus form a reduced LM head (vocabulary shortlist).
    pub shortlists: Vec<(String, PathBuf)>,
    /// Minimum corpus frequency for a token to enter a shortlist.
    pub shortlist_min_count: usize,
}

/// Token ids seen at least `min_count` times when tokenizing `corpus`, plus the
/// special tokens. Sorted.
fn shortlist_ids(tokenizer: &tokenizers::Tokenizer, corpus: &Path, min_count: usize, vocab: usize) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(corpus).with_context(|| format!("reading {}", corpus.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut counts = vec![0usize; vocab];
    for chunk in lines.chunks(4096) {
        let enc = tokenizer.encode_batch(chunk.to_vec(), false).map_err(anyhow::Error::msg)?;
        for e in enc {
            for &id in e.get_ids() {
                if let Some(c) = counts.get_mut(id as usize) {
                    *c += 1;
                }
            }
        }
    }
    // <s>, <pad>, </s>, <unk> are always kept.
    let ids: Vec<u32> = (0..vocab as u32).filter(|&i| i < 4 || counts[i as usize] >= min_count).collect();
    Ok(ids)
}

/// Converts a Hugging Face NLLB checkpoint directory into a single GGUF file.
pub fn convert(hf_dir: &Path, out: &Path, opts: &ConvertOptions) -> Result<()> {
    let cfg_json = std::fs::read_to_string(hf_dir.join("config.json"))?;
    let mut cfg = Config::from_hf_json(&cfg_json)?;
    if let Ok(gen) = std::fs::read_to_string(hf_dir.join("generation_config.json")) {
        let gen: serde_json::Value = serde_json::from_str(&gen)?;
        if let Some(m) = gen.get("max_length").and_then(|v| v.as_u64()) {
            cfg.max_length = m as usize;
        }
    }
    let tokenizer_json = std::fs::read_to_string(hf_dir.join("tokenizer.json"))
        .context("tokenizer.json is required (download it from the model repo)")?;
    let mut w = load_checkpoint(hf_dir)?;
    let mut take = |name: &str| -> Result<Tensor> { w.remove(name).with_context(|| format!("missing tensor {name}")) };

    let mut wr = Writer { preset: opts.preset, tensors: vec![], bytes: 0 };
    let enc_scale = ((cfg.d_model / cfg.encoder_attention_heads) as f64).powf(-0.5);
    let dec_scale = ((cfg.d_model / cfg.decoder_attention_heads) as f64).powf(-0.5);

    let shared = take("model.shared.weight").or_else(|_| take("model.encoder.embed_tokens.weight"))?;
    wr.add("token_embd.weight".into(), &shared, opts.preset.embedding())?;

    let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(anyhow::Error::msg)?;
    let mut shortlist_meta = vec![];
    for (lang, corpus) in &opts.shortlists {
        let code = crate::langs::resolve(lang);
        anyhow::ensure!(tokenizer.token_to_id(&code).is_some(), "unknown language {lang}");
        let mut ids = shortlist_ids(&tokenizer, corpus, opts.shortlist_min_count, cfg.vocab_size)?;
        // The language token is fed to the decoder (forced BOS): keep its embedding.
        let lang_id = tokenizer.token_to_id(&code).unwrap_or_default();
        if let Err(pos) = ids.binary_search(&lang_id) {
            ids.insert(pos, lang_id);
        }
        tracing::info!("shortlist {code}: {} / {} tokens", ids.len(), cfg.vocab_size);
        let idx = Tensor::new(ids.as_slice(), &Device::Cpu)?;
        wr.add(format!("lm_head.{code}.weight"), &shared.index_select(&idx, 0)?, opts.preset.embedding())?;
        shortlist_meta
            .push((format!("{ARCH}.shortlist.{code}"), Value::Array(ids.into_iter().map(Value::U32).collect())));
    }

    for (side, n_layers, s) in [("encoder", cfg.encoder_layers, enc_scale), ("decoder", cfg.decoder_layers, dec_scale)]
    {
        let short = &side[..3];
        for i in 0..n_layers {
            let p = format!("model.{side}.layers.{i}");
            let o = format!("{short}.blk.{i}");

            // Fused, pre-scaled self-attention QKV.
            let q_w = (take(&format!("{p}.self_attn.q_proj.weight"))? * s)?;
            let q_b = (take(&format!("{p}.self_attn.q_proj.bias"))? * s)?;
            let k_w = take(&format!("{p}.self_attn.k_proj.weight"))?;
            let k_b = take(&format!("{p}.self_attn.k_proj.bias"))?;
            let v_w = take(&format!("{p}.self_attn.v_proj.weight"))?;
            let v_b = take(&format!("{p}.self_attn.v_proj.bias"))?;
            let lin = wr.linear();
            wr.add(format!("{o}.attn_qkv.weight"), &Tensor::cat(&[q_w, k_w, v_w], 0)?, lin)?;
            wr.add_f32(format!("{o}.attn_qkv.bias"), &Tensor::cat(&[q_b, k_b, v_b], 0)?)?;
            wr.add(format!("{o}.attn_out.weight"), &take(&format!("{p}.self_attn.out_proj.weight"))?, lin)?;
            wr.add_f32(format!("{o}.attn_out.bias"), &take(&format!("{p}.self_attn.out_proj.bias"))?)?;
            wr.add_f32(format!("{o}.attn_norm.weight"), &take(&format!("{p}.self_attn_layer_norm.weight"))?)?;
            wr.add_f32(format!("{o}.attn_norm.bias"), &take(&format!("{p}.self_attn_layer_norm.bias"))?)?;

            if side == "decoder" {
                let q_w = (take(&format!("{p}.encoder_attn.q_proj.weight"))? * s)?;
                let q_b = (take(&format!("{p}.encoder_attn.q_proj.bias"))? * s)?;
                let k_w = take(&format!("{p}.encoder_attn.k_proj.weight"))?;
                let k_b = take(&format!("{p}.encoder_attn.k_proj.bias"))?;
                let v_w = take(&format!("{p}.encoder_attn.v_proj.weight"))?;
                let v_b = take(&format!("{p}.encoder_attn.v_proj.bias"))?;
                wr.add(format!("{o}.cross_attn_q.weight"), &q_w, lin)?;
                wr.add_f32(format!("{o}.cross_attn_q.bias"), &q_b)?;
                wr.add(format!("{o}.cross_attn_kv.weight"), &Tensor::cat(&[k_w, v_w], 0)?, lin)?;
                wr.add_f32(format!("{o}.cross_attn_kv.bias"), &Tensor::cat(&[k_b, v_b], 0)?)?;
                wr.add(
                    format!("{o}.cross_attn_out.weight"),
                    &take(&format!("{p}.encoder_attn.out_proj.weight"))?,
                    lin,
                )?;
                wr.add_f32(format!("{o}.cross_attn_out.bias"), &take(&format!("{p}.encoder_attn.out_proj.bias"))?)?;
                wr.add_f32(
                    format!("{o}.cross_attn_norm.weight"),
                    &take(&format!("{p}.encoder_attn_layer_norm.weight"))?,
                )?;
                wr.add_f32(format!("{o}.cross_attn_norm.bias"), &take(&format!("{p}.encoder_attn_layer_norm.bias"))?)?;
            }

            wr.add(format!("{o}.ffn_up.weight"), &take(&format!("{p}.fc1.weight"))?, lin)?;
            wr.add_f32(format!("{o}.ffn_up.bias"), &take(&format!("{p}.fc1.bias"))?)?;
            let down = opts.preset.ffn_down();
            wr.add(format!("{o}.ffn_down.weight"), &take(&format!("{p}.fc2.weight"))?, down)?;
            wr.add_f32(format!("{o}.ffn_down.bias"), &take(&format!("{p}.fc2.bias"))?)?;
            wr.add_f32(format!("{o}.ffn_norm.weight"), &take(&format!("{p}.final_layer_norm.weight"))?)?;
            wr.add_f32(format!("{o}.ffn_norm.bias"), &take(&format!("{p}.final_layer_norm.bias"))?)?;
        }
        wr.add_f32(format!("{short}.output_norm.weight"), &take(&format!("model.{side}.layer_norm.weight"))?)?;
        wr.add_f32(format!("{short}.output_norm.bias"), &take(&format!("model.{side}.layer_norm.bias"))?)?;
    }

    let mut meta: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String(ARCH.into())),
        ("general.name".into(), Value::String(opts.model_name.clone())),
        ("general.file_type".into(), Value::String(opts.preset.name().into())),
        ("general.license".into(), Value::String("cc-by-nc-4.0".into())),
        ("general.source.url".into(), Value::String(format!("https://huggingface.co/{}", opts.model_name))),
        ("tokenizer.ggml.model".into(), Value::String("nllb".into())),
        ("tokenizer.huggingface.json".into(), Value::String(tokenizer_json)),
    ];
    meta.extend(cfg.to_gguf());
    meta.extend(shortlist_meta);

    tracing::info!(
        "writing {} tensors ({:.1} MiB) to {}",
        wr.tensors.len(),
        wr.bytes as f64 / (1024.0 * 1024.0),
        out.display()
    );
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = wr.tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tmp = out.with_extension("gguf.partial");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    candle_core::quantized::gguf_file::write(&mut f, &meta_refs, &tensor_refs)?;
    drop(f);
    std::fs::rename(&tmp, out)?;
    Ok(())
}
