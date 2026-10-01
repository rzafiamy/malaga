//! Hugging Face VITS (MMS-TTS) -> GGUF conversion.
//!
//! Only what inference needs is written, with the constant work baked in:
//!
//! * training-only tensors are dropped: the posterior encoder, the duration
//!   predictor's `post_*` flows and its flow 1 (skipped by the reverse pass) —
//!   about 23 % of the checkpoint;
//! * weight norm (`weight_g`, `weight_v`) is folded into plain weights;
//! * Q, K, V are fused and Q is pre-scaled by `1/sqrt(head_dim)`;
//! * the embedding is pre-scaled by `sqrt(hidden_size)`;
//! * the spline width/height logits are pre-divided by `sqrt(filter_channels)`;
//! * HiFi-GAN's `/ num_kernels` resblock average is folded into the next
//!   upsampler (and `conv_post`): leaky ReLU is positively homogeneous.
//!
//! The character vocabulary and the full HF config ride along as metadata, so
//! the `.gguf` is self-contained.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Tensor};

use crate::convert::{load_checkpoint, Preset};
use crate::vits::{VitsConfig, VITS_ARCH};

/// Training-only parameters.
fn unused(name: &str) -> bool {
    name.starts_with("posterior_encoder.")
        || name.starts_with("duration_predictor.post_")
        || name.starts_with("duration_predictor.flows.1.")
}

/// HF prefix -> short GGUF prefix (GGUF names are limited to 64 bytes).
fn rename(name: &str) -> String {
    for (from, to) in [
        ("text_encoder.encoder.layers.", "enc."),
        ("text_encoder.project.", "enc.proj."),
        ("duration_predictor.", "dp."),
        ("flow.flows.", "flow."),
        ("decoder.", "dec."),
    ] {
        if let Some(rest) = name.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    name.to_string()
}

/// `g * v / ||v||`, the norm taken over every axis but the first.
fn take(w: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    Ok(w.remove(name).with_context(|| format!("missing tensor {name}"))?.to_dtype(DType::F32)?)
}

fn fold_weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let n = v.dims()[0];
    let norm = v.sqr()?.reshape((n, ()))?.sum_keepdim(1)?.sqrt()?;
    let shape: Vec<usize> = std::iter::once(n).chain(std::iter::repeat_n(1, v.rank() - 1)).collect();
    Ok(v.broadcast_mul(&g.reshape(shape.as_slice())?)?.broadcast_div(&norm.reshape(shape.as_slice())?)?)
}

pub fn convert_vits(hf_dir: &Path, out: &Path, preset: Preset, model_name: &str) -> Result<()> {
    let dtype = match preset {
        Preset::F32 => GgmlDType::F32,
        Preset::F16 => GgmlDType::F16,
        p => bail!("VITS supports the f32 and f16 presets (got {}): it is a convolutional model", p.name()),
    };
    let cfg_json = std::fs::read_to_string(hf_dir.join("config.json"))?;
    let cfg = VitsConfig::from_hf_json(&cfg_json)?;

    // VitsTokenizer: vocab.json + tokenizer_config.json.
    let vocab: HashMap<String, u32> =
        serde_json::from_str(&std::fs::read_to_string(hf_dir.join("vocab.json")).context("vocab.json is required")?)?;
    let tok_cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(hf_dir.join("tokenizer_config.json")).unwrap_or("{}".into()))?;
    let flag = |k: &str| tok_cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    if flag("phonemize") {
        bail!("this checkpoint needs an espeak phonemizer, which malaga does not implement");
    }
    if flag("is_uroman") {
        tracing::warn!("checkpoint expects uroman-romanized input: non-Latin text will be dropped");
    }
    let mut tokens = vec![String::new(); vocab.len()];
    for (t, &id) in &vocab {
        *tokens.get_mut(id as usize).context("non-contiguous vocab ids")? = t.clone();
    }
    anyhow::ensure!(tokens.len() == cfg.vocab_size, "vocab.json has {} entries, config {}", tokens.len(), cfg.vocab_size);

    let mut w = load_checkpoint(hf_dir)?;
    let total: usize = w.values().map(|t| t.elem_count()).sum();

    // Weight norm.
    let gs: Vec<String> = w.keys().filter(|k| k.ends_with(".weight_g")).cloned().collect();
    for g in gs {
        let base = g.trim_end_matches("_g").to_string();
        let gt = w.remove(&g).unwrap_or_else(|| unreachable!());
        let vt = w.remove(&format!("{base}_v")).with_context(|| format!("missing {base}_v"))?;
        w.insert(base, fold_weight_norm(&gt.to_dtype(DType::F32)?, &vt.to_dtype(DType::F32)?)?);
    }
    w.retain(|k, _| !unused(k));

    let mut out_t: Vec<(String, Tensor)> = vec![];
    // Text encoder.
    let c = cfg.hidden_size;
    let emb = (take(&mut w, "text_encoder.embed_tokens.weight")? * (c as f64).sqrt())?;
    out_t.push(("enc.token_embd".into(), emb));
    let scale = (cfg.head_dim() as f64).powf(-0.5);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("text_encoder.encoder.layers.{i}.attention");
        let o = format!("enc.{i}");
        let qw = (take(&mut w, &format!("{p}.q_proj.weight"))? * scale)?;
        let qb = (take(&mut w, &format!("{p}.q_proj.bias"))? * scale)?;
        let (kw, kb) = (take(&mut w, &format!("{p}.k_proj.weight"))?, take(&mut w, &format!("{p}.k_proj.bias"))?);
        let (vw, vb) = (take(&mut w, &format!("{p}.v_proj.weight"))?, take(&mut w, &format!("{p}.v_proj.bias"))?);
        out_t.push((format!("{o}.attn_qkv.weight"), Tensor::cat(&[qw, kw, vw], 0)?));
        out_t.push((format!("{o}.attn_qkv.bias"), Tensor::cat(&[qb, kb, vb], 0)?));
        out_t.push((format!("{o}.attn_out.weight"), take(&mut w, &format!("{p}.out_proj.weight"))?));
        out_t.push((format!("{o}.attn_out.bias"), take(&mut w, &format!("{p}.out_proj.bias"))?));
        out_t.push((format!("{o}.rel_k"), take(&mut w, &format!("{p}.emb_rel_k"))?.squeeze(0)?));
        out_t.push((format!("{o}.rel_v"), take(&mut w, &format!("{p}.emb_rel_v"))?.squeeze(0)?));
        let p = format!("text_encoder.encoder.layers.{i}");
        for (from, to) in [
            ("layer_norm", "attn_norm"),
            ("final_layer_norm", "ffn_norm"),
            ("feed_forward.conv_1", "ffn_up"),
            ("feed_forward.conv_2", "ffn_down"),
        ] {
            for s in ["weight", "bias"] {
                out_t.push((format!("{o}.{to}.{s}"), take(&mut w, &format!("{p}.{from}.{s}"))?));
            }
        }
    }

    // Spline logits: widths and heights are divided by sqrt(filter_channels).
    let bins = cfg.duration_predictor_flow_bins;
    let inv = 1.0 / (cfg.hidden_size as f64).sqrt();
    let row_scale: Vec<f32> = (0..3 * bins - 1).map(|r| if r < 2 * bins { inv as f32 } else { 1.0 }).collect();
    for i in 2..=cfg.duration_predictor_num_flows {
        for s in ["weight", "bias"] {
            let name = format!("duration_predictor.flows.{i}.conv_proj.{s}");
            let t = take(&mut w, &name)?;
            let sc = Tensor::new(row_scale.as_slice(), t.device())?;
            let sc = if s == "weight" { sc.reshape(((), 1, 1))? } else { sc };
            w.insert(name, t.broadcast_mul(&sc)?);
        }
    }

    // HiFi-GAN: fold the 1/num_kernels average into the next conv.
    let nk = cfg.resblock_kernel_sizes.len() as f64;
    for i in 1..cfg.upsample_rates.len() {
        let name = format!("decoder.upsampler.{i}.weight");
        let t = (take(&mut w, &name)? / nk)?;
        w.insert(name, t);
    }
    let post = (take(&mut w, "decoder.conv_post.weight")? / nk)?;
    w.insert("decoder.conv_post.weight".into(), post);

    // Everything else, renamed.
    let mut rest: Vec<String> = w.keys().cloned().collect();
    rest.sort();
    for k in rest {
        let t = w.remove(&k).unwrap_or_else(|| unreachable!()).to_dtype(DType::F32)?;
        out_t.push((rename(&k), t));
    }

    let mut qs = vec![];
    let mut kept = 0usize;
    let mut bytes = 0usize;
    for (name, t) in out_t {
        anyhow::ensure!(name.len() < 64, "tensor name too long for GGUF: {name}");
        // Norms, biases and other vectors stay f32; so do the tiny relative embeddings.
        let dt = if t.rank() < 2 || name.contains("norm") || name.contains(".rel_") { GgmlDType::F32 } else { dtype };
        let q = QTensor::quantize(&t.contiguous()?, dt).with_context(|| format!("encoding {name}"))?;
        kept += t.elem_count();
        bytes += q.storage_size_in_bytes();
        qs.push((name, q));
    }
    tracing::info!(
        "kept {:.2}M of {:.2}M parameters, {} tensors, {:.1} MiB",
        kept as f64 / 1e6,
        total as f64 / 1e6,
        qs.len(),
        bytes as f64 / (1024.0 * 1024.0)
    );

    let s = |v: &str| Value::String(v.to_string());
    let meta: Vec<(String, Value)> = vec![
        ("general.architecture".into(), s(VITS_ARCH)),
        ("general.name".into(), s(model_name)),
        ("general.file_type".into(), s(preset.name())),
        ("general.license".into(), s("cc-by-nc-4.0")),
        ("general.source.url".into(), Value::String(format!("https://huggingface.co/{model_name}"))),
        ("vits.config".into(), Value::String(serde_json::to_string(&cfg)?)),
        ("vits.language".into(), s(tok_cfg.get("language").and_then(|v| v.as_str()).unwrap_or(""))),
        ("vits.add_blank".into(), Value::Bool(flag("add_blank"))),
        ("tokenizer.ggml.model".into(), s("vits-chars")),
        ("tokenizer.ggml.tokens".into(), Value::Array(tokens.into_iter().map(Value::String).collect())),
    ];
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = qs.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tmp = out.with_extension("gguf.partial");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    candle_core::quantized::gguf_file::write(&mut f, &meta_refs, &tensor_refs)?;
    drop(f);
    std::fs::rename(&tmp, out)?;
    Ok(())
}

/// `model_type` of a Hugging Face checkpoint directory.
pub fn hf_model_type(hf_dir: &Path) -> Result<String> {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(hf_dir.join("config.json"))?)?;
    Ok(cfg.get("model_type").and_then(|v| v.as_str()).unwrap_or_default().to_string())
}
