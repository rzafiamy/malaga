//! NLLB-200 / M2M100 encoder-decoder transformer on (quantized) GGUF weights.
//!
//! The weights are immutable and can be shared between threads; every
//! per-request piece of state (self-attention KV cache, projected encoder
//! keys/values) lives in [`DecoderState`].

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::kv_cache::KvCache;
use candle_nn::LayerNorm;

use crate::config::Config;

/// Reads tensors from a GGUF file straight to the target device.
pub(crate) struct Weights<'a> {
    pub ct: &'a Content,
    file: std::fs::File,
    device: Device,
}

impl<'a> Weights<'a> {
    pub fn new(ct: &'a Content, path: &Path, device: &Device) -> Result<Self> {
        Ok(Self { ct, file: std::fs::File::open(path)?, device: device.clone() })
    }

    fn qtensor(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        self.ct.tensor(&mut self.file, name, device).with_context(|| format!("loading tensor {name}"))
    }

    /// Small dense tensor (norms, biases): read on the CPU and only upload the f32 result.
    fn dense(&mut self, name: &str) -> Result<Tensor> {
        let dev = self.device.clone();
        Ok(self.qtensor(name, &Device::Cpu)?.dequantize(&Device::Cpu)?.to_device(&dev)?)
    }

    /// Matmul weight. Quantized tensors stay quantized in VRAM; f16/bf16 weights stay
    /// in f16 (instead of candle's default of expanding them to f32).
    fn matmul(&mut self, name: &str) -> Result<QMatMul> {
        let dev = self.device.clone();
        self.matmul_on(name, &dev)
    }

    fn matmul_on(&mut self, name: &str, dev: &Device) -> Result<QMatMul> {
        let dev = dev.clone();
        let cpu = self.qtensor(name, &Device::Cpu)?;
        Ok(match cpu.dtype() {
            GgmlDType::F32 => QMatMul::Tensor(cpu.dequantize(&Device::Cpu)?.to_device(&dev)?),
            GgmlDType::F16 | GgmlDType::BF16 => {
                QMatMul::TensorF16(cpu.dequantize(&Device::Cpu)?.to_dtype(DType::F16)?.to_device(&dev)?)
            }
            _ if dev.is_cpu() => QMatMul::QTensor(Arc::new(cpu)),
            _ => QMatMul::QTensor(Arc::new(self.qtensor(name, &dev)?)),
        })
    }

    fn linear(&mut self, name: &str) -> Result<Linear> {
        Ok(Linear { w: self.matmul(&format!("{name}.weight"))?, b: self.dense(&format!("{name}.bias"))? })
    }

    fn layer_norm(&mut self, name: &str) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.dense(&format!("{name}.weight"))?,
            self.dense(&format!("{name}.bias"))?,
            Config::LAYER_NORM_EPS,
        ))
    }
}

#[derive(Debug, Clone)]
struct Linear {
    w: QMatMul,
    b: Tensor,
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.w.forward(x)?.broadcast_add(&self.b)
    }
}

/// `[b, t, n*h*hd]` -> `[n, b, h, t, hd]` with a single copy kernel.
fn split_heads(x: &Tensor, n: usize, heads: usize) -> Result<Tensor> {
    let (b, t, c) = x.dims3()?;
    let hd = c / (n * heads);
    Ok(x.reshape((b, t, n, heads, hd))?.permute((2, 0, 3, 1, 4))?.contiguous()?)
}

/// `[b, h, t, hd]` -> `[b, t, h*hd]`.
fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (b, h, t, hd) = x.dims4()?;
    Ok(x.transpose(1, 2)?.reshape((b, t, h * hd))?)
}

/// Scaled dot-product attention. The `1/sqrt(hd)` factor is folded into the
/// query projection at conversion time.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let att = q.matmul(&k.t()?)?;
    let att = match mask {
        Some(m) => att.broadcast_add(m)?,
        None => att,
    };
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    Ok(att.matmul(v)?)
}

#[derive(Debug, Clone)]
struct SelfAttention {
    qkv: Linear,
    out: Linear,
    heads: usize,
}

impl SelfAttention {
    /// Returns `[3, b, h, t, hd]` (q, k, v).
    fn qkv(&self, x: &Tensor) -> Result<Tensor> {
        split_heads(&self.qkv.forward(x)?, 3, self.heads)
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, cache: Option<&mut KvCache>) -> Result<Tensor> {
        let qkv = self.qkv(x)?;
        let (q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
        let (k, v) = match cache {
            Some(c) => c.append(&k, &v)?,
            None => (k, v),
        };
        let y = attention(&q, &k, &v, mask)?;
        Ok(self.out.forward(&merge_heads(&y)?)?)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    attn_norm: LayerNorm,
    attn: SelfAttention,
    ffn_norm: LayerNorm,
    up: Linear,
    down: Linear,
}

fn ffn(x: &Tensor, norm: &LayerNorm, up: &Linear, down: &Linear) -> Result<Tensor> {
    Ok(down.forward(&up.forward(&norm.forward(x)?)?.relu()?)?)
}

impl EncoderLayer {
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.attn_norm.forward(x)?, mask, None)?)?;
        Ok((&x + ffn(&x, &self.ffn_norm, &self.up, &self.down)?)?)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    attn_norm: LayerNorm,
    attn: SelfAttention,
    cross_norm: LayerNorm,
    cross_q: Linear,
    cross_kv: Linear,
    cross_out: Linear,
    ffn_norm: LayerNorm,
    up: Linear,
    down: Linear,
    heads: usize,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &Tensor,
        self_mask: Option<&Tensor>,
        cache: &mut KvCache,
        cross: &(Tensor, Tensor),
        cross_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.attn_norm.forward(x)?, self_mask, Some(cache))?)?;
        self.cross_and_ffn(x, cross, cross_mask)
    }

    fn cross_and_ffn(&self, x: Tensor, cross: &(Tensor, Tensor), cross_mask: Option<&Tensor>) -> Result<Tensor> {
        let q = split_heads(&self.cross_q.forward(&self.cross_norm.forward(&x)?)?, 1, self.heads)?.get(0)?;
        let y = attention(&q, &cross.0, &cross.1, cross_mask)?;
        let x = (x + self.cross_out.forward(&merge_heads(&y)?)?)?;
        Ok((&x + ffn(&x, &self.ffn_norm, &self.up, &self.down)?)?)
    }
}

/// A `(keys, values)` pair.
pub type KeyValue = (Tensor, Tensor);

/// Per-request decoder state.
pub struct DecoderState {
    caches: Vec<KvCache>,
    /// Projected encoder keys/values for every decoder layer, `[b, h, s, hd]`.
    cross: Vec<(Tensor, Tensor)>,
    cross_mask: Option<Tensor>,
    pos: usize,
}

impl DecoderState {
    pub fn batch_size(&self) -> Result<usize> {
        Ok(self.cross[0].0.dim(0)?)
    }

    /// Reorders the batch (beam search): every cached tensor is gathered with `idx`.
    pub fn reorder(&mut self, idx: &Tensor) -> Result<()> {
        for c in self.caches.iter_mut() {
            if let (Some(k), Some(v)) = (c.k()?, c.v()?) {
                let (k, v) = (k.contiguous()?.index_select(idx, 0)?, v.contiguous()?.index_select(idx, 0)?);
                let max = c.k_cache().max_seq_len();
                *c = KvCache::new(2, max);
                c.append(&k, &v)?;
            }
        }
        for (k, v) in self.cross.iter_mut() {
            *k = k.index_select(idx, 0)?;
            *v = v.index_select(idx, 0)?;
        }
        if let Some(m) = self.cross_mask.as_mut() {
            *m = m.index_select(idx, 0)?;
        }
        Ok(())
    }
}

/// Reduced LM head for one target language (see `malaga convert --shortlist`).
#[derive(Clone)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct Shortlist {
    pub(crate) head: QMatMul,
    /// Vocabulary id of every row of `head`, `[n]` u32 on the device.
    pub(crate) ids: Tensor,
    /// Same ids on the host (sorted).
    pub(crate) host_ids: Arc<Vec<u32>>,
}

impl Shortlist {
    /// Whether vocabulary id `id` is part of the shortlist.
    pub fn contains(&self, id: u32) -> bool {
        self.row(id).is_some()
    }

    /// Row of `head` holding vocabulary id `id`.
    pub(crate) fn row(&self, id: u32) -> Option<u32> {
        self.host_ids.binary_search(&id).ok().map(|r| r as u32)
    }
}

/// The shared token embedding (`[vocab, d]`, also the tied LM head).
///
/// It is the largest tensor of the model (256k rows). A host copy serves the
/// encoder and source-token lookups; the device copy holds the LM head. With
/// `--fast-vocab` (shortlist) decoding the device copy is never needed, so
/// `MALAGA_LAZY_EMBEDDINGS=1` skips uploading it and saves ~45% of the
/// weights' VRAM.
struct Embedding {
    host: QMatMul,
    device_copy: std::sync::OnceLock<QMatMul>,
    path: std::path::PathBuf,
    info: candle_core::quantized::gguf_file::TensorInfo,
    data_offset: u64,
    device: Device,
}

impl Embedding {
    /// Rows for host-side ids, uploaded to the device as f32 `[n, d]`.
    fn lookup(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(ids, &Device::Cpu)?;
        Ok(self.host.embedding(&ids)?.to_dtype(DType::F32)?.to_device(&self.device)?)
    }

    fn on_device(&self) -> Result<&QMatMul> {
        if self.device.is_cpu() {
            return Ok(&self.host);
        }
        if let Some(m) = self.device_copy.get() {
            return Ok(m);
        }
        tracing::info!("uploading the full token embedding to the device");
        let mut f = std::fs::File::open(&self.path)?;
        let oom_hint = || {
            "not enough GPU memory for the full-vocabulary LM head; free some VRAM, use \
             greedy decoding to a language with a shortlist, or run on CPU"
        };
        let m = match self.info.ggml_dtype {
            // Dense tables: expand on the host, upload once (f16 stays f16).
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                let t = self.info.read(&mut f, self.data_offset, &Device::Cpu)?.dequantize(&Device::Cpu)?;
                let t = if self.info.ggml_dtype == GgmlDType::F32 { t } else { t.to_dtype(DType::F16)? };
                let t = t.to_device(&self.device).with_context(oom_hint)?;
                if t.dtype() == DType::F16 {
                    QMatMul::TensorF16(t)
                } else {
                    QMatMul::Tensor(t)
                }
            }
            _ => QMatMul::QTensor(Arc::new(
                self.info.read(&mut f, self.data_offset, &self.device).with_context(oom_hint)?,
            )),
        };
        let _ = self.device_copy.set(m);
        Ok(self.device_copy.get().expect("just set"))
    }
}

pub struct Nllb {
    pub cfg: Config,
    shortlists: std::collections::HashMap<String, Shortlist>,
    /// Shared token embedding, also used (tied) as the LM head.
    embed: Embedding,
    embed_scale: f64,
    /// Sinusoidal position table, `[max_pos + 2, d]`.
    positions: Tensor,
    encoder: Vec<EncoderLayer>,
    enc_norm: LayerNorm,
    decoder: Vec<DecoderLayer>,
    dec_norm: LayerNorm,
    device: Device,
}

/// fairseq / M2M100 sinusoidal table: `[sin(p*f) | cos(p*f)]`, row `padding_idx` zeroed.
fn sinusoidal_table(n: usize, dim: usize, padding_idx: usize) -> Vec<f32> {
    let half = dim / 2;
    let step = (10000f64).ln() / (half as f64 - 1.0);
    let mut out = vec![0f32; n * dim];
    for p in 0..n {
        if p == padding_idx {
            continue;
        }
        for i in 0..half {
            let a = p as f64 * (-(i as f64) * step).exp();
            out[p * dim + i] = a.sin() as f32;
            out[p * dim + half + i] = a.cos() as f32;
        }
    }
    out
}

impl Nllb {
    pub(crate) fn load(ct: &Content, path: &Path, device: &Device) -> Result<Self> {
        let cfg = Config::from_gguf(ct)?;
        let mut w = Weights::new(ct, path, device)?;
        let embed = Embedding {
            host: w.matmul_on("token_embd.weight", &Device::Cpu)?,
            device_copy: std::sync::OnceLock::new(),
            path: path.to_path_buf(),
            info: {
                let i = ct.tensor_infos.get("token_embd.weight").context("missing token_embd.weight")?;
                candle_core::quantized::gguf_file::TensorInfo {
                    ggml_dtype: i.ggml_dtype,
                    shape: i.shape.clone(),
                    offset: i.offset,
                }
            },
            data_offset: ct.tensor_data_offset,
            device: device.clone(),
        };
        let n_pos = cfg.max_position_embeddings + cfg.pad_token_id as usize + 1;
        let positions = Tensor::from_vec(
            sinusoidal_table(n_pos, cfg.d_model, cfg.pad_token_id as usize),
            (n_pos, cfg.d_model),
            device,
        )?;

        let mut encoder = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            let p = format!("enc.blk.{i}");
            encoder.push(EncoderLayer {
                attn_norm: w.layer_norm(&format!("{p}.attn_norm"))?,
                attn: SelfAttention {
                    qkv: w.linear(&format!("{p}.attn_qkv"))?,
                    out: w.linear(&format!("{p}.attn_out"))?,
                    heads: cfg.encoder_attention_heads,
                },
                ffn_norm: w.layer_norm(&format!("{p}.ffn_norm"))?,
                up: w.linear(&format!("{p}.ffn_up"))?,
                down: w.linear(&format!("{p}.ffn_down"))?,
            });
        }
        let mut decoder = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let p = format!("dec.blk.{i}");
            decoder.push(DecoderLayer {
                attn_norm: w.layer_norm(&format!("{p}.attn_norm"))?,
                attn: SelfAttention {
                    qkv: w.linear(&format!("{p}.attn_qkv"))?,
                    out: w.linear(&format!("{p}.attn_out"))?,
                    heads: cfg.decoder_attention_heads,
                },
                cross_norm: w.layer_norm(&format!("{p}.cross_attn_norm"))?,
                cross_q: w.linear(&format!("{p}.cross_attn_q"))?,
                cross_kv: w.linear(&format!("{p}.cross_attn_kv"))?,
                cross_out: w.linear(&format!("{p}.cross_attn_out"))?,
                ffn_norm: w.layer_norm(&format!("{p}.ffn_norm"))?,
                up: w.linear(&format!("{p}.ffn_up"))?,
                down: w.linear(&format!("{p}.ffn_down"))?,
                heads: cfg.decoder_attention_heads,
            });
        }
        let mut shortlists = std::collections::HashMap::new();
        let prefix = format!("{}.shortlist.", crate::config::ARCH);
        for (key, value) in &ct.metadata {
            if let Some(code) = key.strip_prefix(&prefix) {
                let host_ids = value.to_vec()?.iter().map(|v| v.to_u32()).collect::<candle_core::Result<Vec<u32>>>()?;
                let head = w.matmul(&format!("lm_head.{code}.weight"))?;
                let ids = Tensor::new(host_ids.as_slice(), device)?;
                shortlists.insert(code.to_string(), Shortlist { head, ids, host_ids: Arc::new(host_ids) });
            }
        }
        let enc_norm = w.layer_norm("enc.output_norm")?;
        let dec_norm = w.layer_norm("dec.output_norm")?;
        let embed_scale = if cfg.scale_embedding { (cfg.d_model as f64).sqrt() } else { 1.0 };
        if shortlists.is_empty() || std::env::var_os("MALAGA_LAZY_EMBEDDINGS").is_none() {
            // The exact (default) decoding path needs the full LM head: upload it now.
            // MALAGA_LAZY_EMBEDDINGS defers it, for `--fast-vocab` deployments.
            embed.on_device()?;
        }
        Ok(Self {
            cfg,
            shortlists,
            embed,
            embed_scale,
            positions,
            encoder,
            enc_norm,
            decoder,
            dec_norm,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Vocabulary shortlist for an NLLB language code, if the GGUF has one.
    pub fn shortlist(&self, code: &str) -> Option<&Shortlist> {
        self.shortlists.get(code)
    }

    /// Languages with a vocabulary shortlist.
    pub fn shortlist_langs(&self) -> Vec<&str> {
        self.shortlists.keys().map(|s| s.as_str()).collect()
    }

    /// Unscaled embeddings of host-side token ids, `[n, d]` f32 on the device.
    pub(crate) fn embedding_rows(&self, ids: &[u32]) -> Result<Tensor> {
        self.embed.lookup(ids)
    }

    /// Scaled token embeddings (`[n, d]` f32) plus positions, reshaped to `[b, t, d]`.
    fn add_positions(&self, x: Tensor, pos: &Tensor, (b, t): (usize, usize)) -> Result<Tensor> {
        let p = self.positions.index_select(&pos.flatten_all()?, 0)?;
        Ok(((x * self.embed_scale)? + p)?.reshape((b, t, self.cfg.d_model))?)
    }

    /// Runs the encoder on right-padded `ids` (`[b, s]`, padding = `pad_token_id`) and
    /// projects the cross-attention keys/values of every decoder layer once.
    /// Returns them with the additive padding mask (`[b, 1, 1, s]`, `None` without padding).
    pub fn encode_cross(&self, ids: &[Vec<u32>]) -> Result<(Vec<KeyValue>, Option<Tensor>)> {
        let (flat, pos_t, mask) = self.pad_batch(ids)?;
        let dims = pos_t.dims2()?;
        let mut x = self.add_positions(self.embedding_rows(&flat)?, &pos_t, dims)?;
        for l in &self.encoder {
            x = l.forward(&x, mask.as_ref())?;
        }
        let enc = self.enc_norm.forward(&x)?;
        let mut cross = Vec::with_capacity(self.decoder.len());
        for l in &self.decoder {
            let kv = split_heads(&l.cross_kv.forward(&enc)?, 2, l.heads)?;
            cross.push((kv.get(0)?, kv.get(1)?));
        }
        Ok((cross, mask))
    }

    /// Right-pads a batch: returns host ids, position-table indices and the additive mask.
    fn pad_batch(&self, ids: &[Vec<u32>]) -> Result<(Vec<u32>, Tensor, Option<Tensor>)> {
        let b = ids.len();
        let s = ids.iter().map(|v| v.len()).max().unwrap_or(0);
        let pad = self.cfg.pad_token_id;
        let mut flat = vec![pad; b * s];
        let mut pos = vec![pad; b * s];
        let mut mask = vec![0f32; b * s];
        for (i, row) in ids.iter().enumerate() {
            for j in 0..s {
                if j < row.len() {
                    flat[i * s + j] = row[j];
                    pos[i * s + j] = j as u32 + pad + 1;
                } else {
                    mask[i * s + j] = f32::NEG_INFINITY;
                }
            }
        }
        let padded = ids.iter().any(|r| r.len() < s);
        let mask = if padded { Some(Tensor::from_vec(mask, (b, 1, 1, s), &self.device)?) } else { None };
        Ok((flat, Tensor::from_vec(pos, (b, s), &self.device)?, mask))
    }

    /// Encodes the source and prepares a dynamic decoder state.
    pub fn encode(&self, ids: &[Vec<u32>], max_new_tokens: usize) -> Result<DecoderState> {
        let (cross, cross_mask) = self.encode_cross(ids)?;
        let max_len = max_new_tokens + 2;
        Ok(DecoderState {
            caches: (0..self.decoder.len()).map(|_| KvCache::new(2, max_len)).collect(),
            cross,
            cross_mask,
            pos: 0,
        })
    }

    /// Feeds `ids` (`[b, t]`) to the decoder and returns the logits of the last position, `[b, vocab]`.
    pub fn decode(&self, ids: &Tensor, st: &mut DecoderState) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let base = st.pos as u32 + self.cfg.pad_token_id + 1;
        let pos =
            Tensor::arange(base, base + t as u32, &self.device)?.unsqueeze(0)?.broadcast_as((b, t))?.contiguous()?;
        let self_mask = if t > 1 {
            let past = st.pos;
            let m: Vec<f32> = (0..t)
                .flat_map(|i| (0..past + t).map(move |j| if j > past + i { f32::NEG_INFINITY } else { 0. }))
                .collect();
            Some(Tensor::from_vec(m, (1, 1, t, past + t), &self.device)?)
        } else {
            None
        };
        let emb = self.embed.on_device()?.embedding(&ids.flatten_all()?)?.to_dtype(DType::F32)?;
        let mut x = self.add_positions(emb, &pos, (b, t))?;
        for (l, (cache, cross)) in self.decoder.iter().zip(st.caches.iter_mut().zip(st.cross.iter())) {
            x = l.forward(&x, self_mask.as_ref(), cache, cross, st.cross_mask.as_ref())?;
        }
        st.pos += t;
        self.lm_head(&x.narrow(1, t - 1, 1)?.squeeze(1)?)
    }

    pub(crate) fn lm_head(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.embed.on_device()?.forward(&self.dec_norm.forward(x)?)?)
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn decoder_layers(&self) -> usize {
        self.decoder.len()
    }

    /// One greedy decoding step that only reads and writes the fixed buffers of
    /// `st`: no host transfer and no shape change between steps, so it can be
    /// recorded once as a CUDA graph and replayed for every token. Uses the
    /// fused kernels of `kernels/malaga.cu` (~19 launches per layer).
    #[cfg(feature = "cuda")]
    pub(crate) fn step_static(&self, st: &crate::graph::StaticState) -> Result<()> {
        use crate::kernels as k;
        let eps = Config::LAYER_NORM_EPS as f32;
        let norm = |n: &LayerNorm| -> Result<(Tensor, Tensor)> {
            Ok((n.weight().clone(), n.bias().context("layer norm without bias")?.clone()))
        };
        // Input embedding: with a shortlist, the previous token is a shortlist row
        // or a source token, both already on the device; otherwise use the full table.
        let (emb, src) = match &st.shortlist {
            Some(sl) => (sl.head.embedding(&st.sl_row)?, Some((&st.src_emb, &st.src_row))),
            None => (self.embed.on_device()?.embedding(&st.ids.flatten_all()?)?, None),
        };
        let emb = emb.to_dtype(DType::F32)?;
        let (g, b) = norm(&self.decoder[0].attn_norm)?;
        let base = self.cfg.pad_token_id + 1;
        let (mut x, mut h) =
            k::embed_pos_ln(&emb, src, &self.positions, &st.step, base, self.embed_scale as f32, &g, &b, eps)?;
        for (i, l) in self.decoder.iter().enumerate() {
            let (kc, vc) = &st.kv[i];
            let q = k::qkv_decode(&l.attn.qkv.w.forward(&h)?, &l.attn.qkv.b, kc, vc, &st.step, l.heads)?;
            let a = k::attn_decode(&q, None, kc, vc, None, Some(&st.step))?;
            let (g, b) = norm(&l.cross_norm)?;
            (x, h) = k::bias_residual_ln(&l.attn.out.w.forward(&a)?, &l.attn.out.b, &x, &g, &b, eps)?;

            let (ck, cv) = &st.cross[i];
            let q = l.cross_q.w.forward(&h)?;
            let a = k::attn_decode(&q, Some(&l.cross_q.b), ck, cv, Some(&st.lens), None)?;
            let (g, b) = norm(&l.ffn_norm)?;
            (x, h) = k::bias_residual_ln(&l.cross_out.w.forward(&a)?, &l.cross_out.b, &x, &g, &b, eps)?;

            let u = k::bias_relu(&l.up.w.forward(&h)?, &l.up.b)?;
            let next = self.decoder.get(i + 1).map(|n| &n.attn_norm).unwrap_or(&self.dec_norm);
            let (g, b) = norm(next)?;
            (x, h) = k::bias_residual_ln(&l.down.w.forward(&u)?, &l.down.b, &x, &g, &b, eps)?;
        }
        // `h` is already dec_norm(x).
        let buffers = k::GreedyBuffers {
            finished: &st.finished,
            ids: &st.ids,
            out: &st.out,
            step: &st.step,
            eos: self.cfg.eos_token_id,
            pad: self.cfg.pad_token_id,
            rows: st.shortlist.as_ref().map(|_| (&st.sl_row, &st.src_row)),
        };
        match &st.shortlist {
            // Shortlist rows plus the tokens of the source sentence (names,
            // numbers... that the model copies over).
            Some(sl) => {
                let main = sl.head.forward(&h)?;
                let src = k::row_dots(&h, &st.src_emb)?;
                let logits = k::Logits { main: &main, vocab_map: Some(&sl.ids), src: Some((&src, &st.src_ids)) };
                k::greedy_select(&logits, &buffers)?;
            }
            None => {
                let main = self.embed.on_device()?.forward(&h)?;
                k::greedy_select(&k::Logits { main: &main, vocab_map: None, src: None }, &buffers)?;
            }
        }
        Ok(())
    }

    /// Log-softmax helper used by beam search.
    pub fn log_probs(logits: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::log_softmax(logits, D::Minus1)?)
    }
}
