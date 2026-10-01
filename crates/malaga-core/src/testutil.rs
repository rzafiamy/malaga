//! A tiny random NLLB checkpoint for tests: same architecture and file layout
//! as the real models (config.json, tokenizer.json, safetensors), a few hundred
//! kilobytes, so the whole convert -> load -> translate pipeline can run in CI.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub const LANGS: [&str; 3] = ["fra_Latn", "eng_Latn", "plt_Latn"];
/// Plain words of the tiny vocabulary: `w0` .. `w{WORDS-1}`.
pub const WORDS: usize = 250;

/// Writes a random Hugging Face style NLLB checkpoint into `dir`.
pub fn write_tiny_checkpoint(dir: &Path, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let specials = ["<s>", "<pad>", "</s>", "<unk>"];
    let vocab_size = specials.len() + WORDS + LANGS.len();
    let (d, heads, ffn, layers) = (64usize, 4usize, 128usize, 2usize);

    let config = serde_json::json!({
        "vocab_size": vocab_size, "d_model": d,
        "encoder_layers": layers, "decoder_layers": layers,
        "encoder_attention_heads": heads, "decoder_attention_heads": heads,
        "encoder_ffn_dim": ffn, "decoder_ffn_dim": ffn,
        "max_position_embeddings": 128, "scale_embedding": true,
        "bos_token_id": 0, "pad_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 2,
        "activation_function": "relu", "max_length": 64
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config)?)?;

    let mut vocab = serde_json::Map::new();
    let mut added = vec![];
    for (i, s) in specials.iter().enumerate() {
        vocab.insert(s.to_string(), i.into());
        added.push(serde_json::json!({"id": i, "content": s, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true}));
    }
    for w in 0..WORDS {
        vocab.insert(format!("w{w}"), (specials.len() + w).into());
    }
    for (i, l) in LANGS.iter().enumerate() {
        let id = specials.len() + WORDS + i;
        vocab.insert(l.to_string(), id.into());
        added.push(serde_json::json!({"id": id, "content": l, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true}));
    }
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    });
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_string(&tokenizer)?)?;

    // Deterministic pseudo-random weights (xorshift), no extra dependency.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rand = |shape: &[usize], scale: f32| -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect();
        Ok(Tensor::from_vec(v, shape, &Device::Cpu)?)
    };
    let mut w: HashMap<String, Tensor> = HashMap::new();
    w.insert("model.shared.weight".into(), rand(&[vocab_size, d], 0.5)?);
    let ones = Tensor::ones(d, DType::F32, &Device::Cpu)?;
    let zeros = Tensor::zeros(d, DType::F32, &Device::Cpu)?;
    for side in ["encoder", "decoder"] {
        for l in 0..layers {
            let p = format!("model.{side}.layers.{l}");
            let mut attns = vec!["self_attn"];
            if side == "decoder" {
                attns.push("encoder_attn");
            }
            for a in attns {
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    w.insert(format!("{p}.{a}.{proj}.weight"), rand(&[d, d], 0.3)?);
                    w.insert(format!("{p}.{a}.{proj}.bias"), rand(&[d], 0.1)?);
                }
                w.insert(format!("{p}.{a}_layer_norm.weight"), ones.clone());
                w.insert(format!("{p}.{a}_layer_norm.bias"), zeros.clone());
            }
            w.insert(format!("{p}.fc1.weight"), rand(&[ffn, d], 0.3)?);
            w.insert(format!("{p}.fc1.bias"), rand(&[ffn], 0.1)?);
            w.insert(format!("{p}.fc2.weight"), rand(&[d, ffn], 0.3)?);
            w.insert(format!("{p}.fc2.bias"), rand(&[d], 0.1)?);
            w.insert(format!("{p}.final_layer_norm.weight"), ones.clone());
            w.insert(format!("{p}.final_layer_norm.bias"), zeros.clone());
        }
        w.insert(format!("model.{side}.layer_norm.weight"), ones.clone());
        w.insert(format!("model.{side}.layer_norm.bias"), zeros.clone());
    }
    candle_core::safetensors::save(&w, dir.join("model.safetensors"))?;
    Ok(())
}

/// Sentence of `n` vocabulary words, deterministic in `seed`.
pub fn sentence(n: usize, seed: usize) -> String {
    (0..n).map(|i| format!("w{}", (seed * 31 + i * 17) % WORDS)).collect::<Vec<_>>().join(" ")
}
