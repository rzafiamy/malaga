//! VITS text-to-speech (Meta MMS-TTS, e.g. `facebook/mms-tts-mlg` for Malagasy)
//! on GGUF weights written by `malaga convert` (see `vits_convert.rs`).
//!
//! Only the inference path is kept: text encoder → stochastic duration
//! predictor (reverse flows) → length regulator → prior flows (reverse) →
//! HiFi-GAN. The converter already folded weight norm, the attention and
//! embedding scales, the spline-bin scaling and HiFi-GAN's resblock averaging
//! into the weights, so none of that appears here.
//!
//! One sentence is synthesized at a time (batch 1, no padding masks): the
//! model was trained without punctuation, so cutting at sentence ends and
//! inserting real silence is what restores the pauses.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
pub use candle_core::DType;
use candle_core::{Device, Tensor};
use candle_nn::ops::{leaky_relu, layer_norm, sigmoid, softmax_last_dim};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::config::get;

/// GGUF `general.architecture` value of a converted VITS checkpoint.
pub const VITS_ARCH: &str = "vits";

/// The subset of the Hugging Face `VitsConfig` the inference path needs. Stored
/// verbatim (as JSON) in the GGUF under `vits.config`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VitsConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub window_size: usize,
    pub ffn_dim: usize,
    pub ffn_kernel_size: usize,
    pub flow_size: usize,
    pub layer_norm_eps: f64,
    pub hidden_act: String,
    pub use_stochastic_duration_prediction: bool,
    pub duration_predictor_kernel_size: usize,
    pub duration_predictor_flow_bins: usize,
    pub duration_predictor_tail_bound: f32,
    pub duration_predictor_num_flows: usize,
    pub depth_separable_channels: usize,
    pub depth_separable_num_layers: usize,
    pub prior_encoder_num_flows: usize,
    pub prior_encoder_num_wavenet_layers: usize,
    pub wavenet_kernel_size: usize,
    pub wavenet_dilation_rate: usize,
    pub upsample_initial_channel: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub leaky_relu_slope: f64,
    pub sampling_rate: u32,
    pub noise_scale: f32,
    pub noise_scale_duration: f32,
    pub speaking_rate: f32,
    #[serde(default = "one")]
    pub num_speakers: usize,
}

fn one() -> usize {
    1
}

impl VitsConfig {
    pub fn from_hf_json(json: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(json).context("parsing VITS config.json")?;
        anyhow::ensure!(cfg.hidden_act == "relu", "unsupported activation {}", cfg.hidden_act);
        anyhow::ensure!(cfg.use_stochastic_duration_prediction, "only the stochastic duration predictor is supported");
        anyhow::ensure!(cfg.num_speakers <= 1, "multi-speaker VITS checkpoints are not supported");
        anyhow::ensure!(cfg.depth_separable_channels == 2, "unsupported depth_separable_channels");
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Waveform samples per spectrogram frame (256 for MMS).
    pub fn hop(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}

/// Character tokenizer of `VitsTokenizer` (no uroman / phonemizer: MMS
/// Malagasy uses the raw Latin alphabet).
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    vocab: HashMap<char, u32>,
    add_blank: bool,
}

impl CharTokenizer {
    /// Whether position `i` of an encoded sequence holds a character (not a blank).
    fn is_char(&self, i: usize) -> bool {
        !self.add_blank || i % 2 == 1
    }

    fn is_vowel_id(&self, id: u32) -> bool {
        self.vocab.iter().any(|(c, &v)| v == id && matches!(c, 'a' | 'e' | 'i' | 'o' | 'y' | 'à' | 'ì' | 'ò' | 'ô' | 'ỳ'))
    }

    fn space_id(&self) -> Option<u32> {
        self.vocab.get(&' ').copied()
    }
}

impl CharTokenizer {
    pub fn new(tokens: &[String], add_blank: bool) -> Result<Self> {
        let mut vocab = HashMap::new();
        for (id, t) in tokens.iter().enumerate() {
            let mut chars = t.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => {
                    vocab.insert(c, id as u32);
                }
                _ => bail!("VITS token {t:?} is not a single character"),
            }
        }
        Ok(Self { vocab, add_blank })
    }

    /// Text the model will actually read: NFC, lowercase, characters outside
    /// the vocabulary mapped to their base letter when that one exists
    /// (`é` → `e`) and dropped otherwise (punctuation, digits), whitespace
    /// collapsed. `transformers` only lowercases and drops, which turns
    /// decomposed input (`o` + U+0302) or French accents into silence.
    pub fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut space = true; // no leading space
        for c in text.nfc().flat_map(char::to_lowercase) {
            let c = if c.is_whitespace() { ' ' } else { c };
            let c = if self.vocab.contains_key(&c) {
                Some(c)
            } else {
                c.to_string().nfd().next().filter(|b| *b != c && self.vocab.contains_key(b))
            };
            match c {
                Some(' ') if space => {}
                Some(c) => {
                    space = c == ' ';
                    out.push(c);
                }
                None => {}
            }
        }
        out.trim_end().to_string()
    }

    /// Token ids, with the blank (id 0) interspersed when `add_blank` is set.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let ids: Vec<u32> = self.normalize(text).chars().filter_map(|c| self.vocab.get(&c).copied()).collect();
        if !self.add_blank {
            return ids;
        }
        let mut out = vec![0u32; ids.len() * 2 + 1];
        for (i, id) in ids.into_iter().enumerate() {
            out[2 * i + 1] = id;
        }
        out
    }
}

/// Sampling knobs; the defaults come from the checkpoint's config.
#[derive(Debug, Clone)]
pub struct SynthOptions {
    /// Prior noise (expressiveness). 0 is deterministic and flatter.
    pub noise_scale: f32,
    /// Duration noise (rhythm variation). 0 gives the mean durations.
    pub noise_scale_duration: f32,
    /// > 1 speaks faster.
    pub speaking_rate: f32,
    /// Silence inserted between sentences.
    pub sentence_pause_ms: u32,
    /// Silence at commas, brackets, colons… (the model never saw punctuation).
    pub phrase_pause_ms: u32,
    /// Minimum vowel length. MMS swallows ~40 % of vowels (≤ 32 ms), which
    /// sounds choppy; flooring them keeps every syllable audible.
    pub vowel_floor_ms: u32,
    /// Minimum length of a word-final vowel (Malagasy words end in weak vowels
    /// that the model clips most).
    pub final_vowel_floor_ms: u32,
    /// Malagasy text normalization (numbers, units, symbols, foreign words).
    pub normalize: bool,
    /// Fixed seed for reproducible output; random otherwise.
    pub seed: Option<u64>,
}

impl SynthOptions {
    pub fn from_config(cfg: &VitsConfig) -> Self {
        Self {
            noise_scale: cfg.noise_scale,
            noise_scale_duration: cfg.noise_scale_duration,
            speaking_rate: cfg.speaking_rate,
            sentence_pause_ms: 250,
            phrase_pause_ms: 120,
            vowel_floor_ms: 45,
            final_vowel_floor_ms: 65,
            normalize: true,
            seed: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Weights

struct Loader<'a> {
    ct: &'a Content,
    file: std::fs::File,
    device: Device,
    dtype: DType,
}

impl Loader<'_> {
    /// f32 tensor on the CPU (for the host-side spline / affine flow).
    fn host(&mut self, name: &str) -> Result<Tensor> {
        let q = self.ct.tensor(&mut self.file, name, &Device::Cpu).with_context(|| format!("loading tensor {name}"))?;
        Ok(q.dequantize(&Device::Cpu)?)
    }

    fn get(&mut self, name: &str) -> Result<Tensor> {
        let dev = self.device.clone();
        Ok(self.host(name)?.to_dtype(self.dtype)?.to_device(&dev)?)
    }

    fn conv(&mut self, prefix: &str, padding: usize, dilation: usize) -> Result<Conv> {
        let w = self.get(&format!("{prefix}.weight"))?;
        let b = match self.ct.tensor_infos.contains_key(&format!("{prefix}.bias")) {
            true => Some(self.get(&format!("{prefix}.bias"))?.reshape((1, (), 1))?),
            false => None,
        };
        Ok(Conv { w, b, padding, dilation })
    }

    fn norm(&mut self, prefix: &str) -> Result<(Tensor, Tensor)> {
        Ok((self.get(&format!("{prefix}.weight"))?, self.get(&format!("{prefix}.bias"))?))
    }
}

/// Conv1d with stride 1 on `[b, c, t]`.
struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    padding: usize,
    dilation: usize,
}

impl Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(&self.w, self.padding, 1, self.dilation, 1)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

/// LayerNorm over the channel axis of `[b, c, t]`.
fn channel_norm(x: &Tensor, (w, b): &(Tensor, Tensor), eps: f32) -> Result<Tensor> {
    Ok(layer_norm(&x.transpose(1, 2)?.contiguous()?, w, b, eps)?.transpose(1, 2)?)
}

// ---------------------------------------------------------------------------
// Text encoder

struct EncoderLayer {
    qkv_w: Tensor,
    qkv_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    rel_k: Tensor, // [2w+1, hd]
    rel_v: Tensor, // [2w+1, hd]
    attn_norm: (Tensor, Tensor),
    ffn_up: Conv,
    ffn_down: Conv,
    ffn_norm: (Tensor, Tensor),
}

struct TextEncoder {
    embed: Tensor,
    layers: Vec<EncoderLayer>,
    proj: Conv,
}

/// Index tensors turning the windowed relative-position products into
/// absolute `[h, t, t]` (keys) and back (values), for one sequence length.
struct RelIndex {
    /// `[h, t, t]` into `[h, t, 2w+2]` (last column is zero).
    k: Tensor,
    /// `[h, t, 2w+1]` into `[h, t, t+1]` (last column is zero).
    v: Tensor,
}

impl RelIndex {
    fn new(t: usize, w: usize, h: usize, dev: &Device) -> Result<Self> {
        let n = 2 * w + 1;
        let mut k = vec![n as u32; t * t];
        let mut v = vec![t as u32; t * n];
        for i in 0..t {
            for o in -(w as isize)..=(w as isize) {
                let j = i as isize + o;
                if (0..t as isize).contains(&j) {
                    k[i * t + j as usize] = (o + w as isize) as u32;
                    v[i * n + (o + w as isize) as usize] = j as u32;
                }
            }
        }
        let k = Tensor::from_vec(k, (1, t, t), dev)?.broadcast_as((h, t, t))?.contiguous()?;
        let v = Tensor::from_vec(v, (1, t, n), dev)?.broadcast_as((h, t, n))?.contiguous()?;
        Ok(Self { k, v })
    }
}

impl TextEncoder {
    /// `ids` → (prior means, prior log-std, hidden states), all `[1, c, t]`.
    fn forward(&self, ids: &Tensor, cfg: &VitsConfig) -> Result<(Tensor, Tensor, Tensor)> {
        let (c, h, hd) = (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim());
        let eps = cfg.layer_norm_eps as f32;
        let t = ids.dim(0)?;
        let rel = RelIndex::new(t, cfg.window_size, h, ids.device())?;
        // [t, c] (batch 1 kept implicit through the encoder).
        let mut x = self.embed.index_select(ids, 0)?;
        for l in &self.layers {
            let qkv = x.matmul(&l.qkv_w.t()?)?.broadcast_add(&l.qkv_b)?;
            let heads = |i: usize| -> Result<Tensor> {
                Ok(qkv.narrow(1, i * c, c)?.reshape((t, h, hd))?.transpose(0, 1)?.contiguous()?)
            };
            let (q, k, v) = (heads(0)?, heads(1)?, heads(2)?); // q is pre-scaled
            let mut scores = q.matmul(&k.t()?)?;
            // Relative keys: q · rel_k for the 2w+1 offsets, scattered onto the band.
            let rk = q.broadcast_matmul(&l.rel_k.t()?)?.pad_with_zeros(2, 0, 1)?;
            scores = (scores + rk.gather(&rel.k, 2)?)?;
            let probs = softmax_last_dim(&scores)?;
            let mut out = probs.matmul(&v)?;
            // Relative values: band of the probabilities · rel_v.
            let band = probs.pad_with_zeros(2, 0, 1)?.gather(&rel.v, 2)?;
            out = (out + band.broadcast_matmul(&l.rel_v)?)?;
            let out = out.transpose(0, 1)?.reshape((t, c))?;
            let out = out.matmul(&l.out_w.t()?)?.broadcast_add(&l.out_b)?;
            x = layer_norm(&(x + out)?, &l.attn_norm.0, &l.attn_norm.1, eps)?;

            let y = x.t()?.unsqueeze(0)?; // [1, c, t]
            let y = l.ffn_up.forward(&y)?.relu()?;
            let y = l.ffn_down.forward(&y)?.squeeze(0)?.t()?;
            x = layer_norm(&(x + y)?, &l.ffn_norm.0, &l.ffn_norm.1, eps)?;
        }
        let hidden = x.t()?.unsqueeze(0)?.contiguous()?; // [1, c, t]
        let stats = self.proj.forward(&hidden)?;
        let f = cfg.flow_size;
        Ok((stats.narrow(1, 0, f)?, stats.narrow(1, f, f)?, hidden))
    }
}

// ---------------------------------------------------------------------------
// Stochastic duration predictor (reverse direction only)

/// Dilated depthwise-separable conv stack (`VitsDilatedDepthSeparableConv`).
struct Dds {
    dilated: Vec<(Tensor, Tensor, usize)>, // depthwise weight [c, k] , bias [1, c, 1], dilation
    pointwise: Vec<Conv>,
    norms_1: Vec<(Tensor, Tensor)>,
    norms_2: Vec<(Tensor, Tensor)>,
    kernel: usize,
    eps: f32,
}

impl Dds {
    fn load(w: &mut Loader, p: &str, cfg: &VitsConfig) -> Result<Self> {
        let k = cfg.duration_predictor_kernel_size;
        let mut s = Self {
            dilated: vec![],
            pointwise: vec![],
            norms_1: vec![],
            norms_2: vec![],
            kernel: k,
            eps: cfg.layer_norm_eps as f32,
        };
        for i in 0..cfg.depth_separable_num_layers {
            let dil = k.pow(i as u32);
            let dw = w.get(&format!("{p}.convs_dilated.{i}.weight"))?.squeeze(1)?;
            let db = w.get(&format!("{p}.convs_dilated.{i}.bias"))?.reshape((1, (), 1))?;
            s.dilated.push((dw, db, dil));
            s.pointwise.push(w.conv(&format!("{p}.convs_pointwise.{i}"), 0, 1)?);
            s.norms_1.push(w.norm(&format!("{p}.norms_1.{i}"))?);
            s.norms_2.push(w.norm(&format!("{p}.norms_2.{i}"))?);
        }
        Ok(s)
    }

    /// Depthwise conv as `k` shifted multiply-adds: candle runs `groups = c`
    /// as `c` separate convolutions.
    fn depthwise(&self, x: &Tensor, w: &Tensor, b: &Tensor, dil: usize) -> Result<Tensor> {
        let t = x.dim(2)?;
        let pad = (self.kernel * dil - dil) / 2;
        let xp = x.pad_with_zeros(2, pad, pad)?;
        let mut y = b.broadcast_as(x.shape())?.contiguous()?;
        for j in 0..self.kernel {
            let wj = w.narrow(1, j, 1)?.unsqueeze(0)?; // [1, c, 1]
            y = (y + xp.narrow(2, j * dil, t)?.broadcast_mul(&wj)?)?;
        }
        Ok(y)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.dilated.len() {
            let (w, b, d) = &self.dilated[i];
            let h = self.depthwise(&x, w, b, *d)?;
            let h = channel_norm(&h, &self.norms_1[i], self.eps)?.gelu_erf()?;
            let h = self.pointwise[i].forward(&h)?;
            let h = channel_norm(&h, &self.norms_2[i], self.eps)?.gelu_erf()?;
            x = (x + h)?;
        }
        Ok(x)
    }
}

struct ConvFlow {
    pre: Conv,
    dds: Dds,
    proj: Conv,
}

enum DurationFlow {
    Conv(ConvFlow),
    /// `(translate, log_scale)` per channel, on the host.
    Affine([f32; 2], [f32; 2]),
}

struct DurationPredictor {
    pre: Conv,
    dds: Dds,
    proj: Conv,
    /// Already in reverse application order.
    flows: Vec<DurationFlow>,
}

impl DurationPredictor {
    fn load(w: &mut Loader, cfg: &VitsConfig) -> Result<Self> {
        let pre = w.conv("dp.conv_pre", 0, 1)?;
        let dds = Dds::load(w, "dp.conv_dds", cfg)?;
        let proj = w.conv("dp.conv_proj", 0, 1)?;
        // transformers: reversed(flows)[:-2] + [flows[0]] — flow 1 is never used.
        let mut flows = vec![];
        for i in (2..=cfg.duration_predictor_num_flows).rev() {
            let p = format!("dp.flows.{i}");
            flows.push(DurationFlow::Conv(ConvFlow {
                pre: w.conv(&format!("{p}.conv_pre"), 0, 1)?,
                dds: Dds::load(w, &format!("{p}.conv_dds"), cfg)?,
                proj: w.conv(&format!("{p}.conv_proj"), 0, 1)?,
            }));
        }
        let v = |t: Tensor| -> Result<[f32; 2]> {
            let v = t.flatten_all()?.to_vec1::<f32>()?;
            Ok([v[0], v[1]])
        };
        flows.push(DurationFlow::Affine(v(w.host("dp.flows.0.translate")?)?, v(w.host("dp.flows.0.log_scale")?)?));
        Ok(Self { pre, dds, proj, flows })
    }

    /// Log-durations per token, on the host.
    fn forward(&self, hidden: &Tensor, noise: [Vec<f32>; 2], cfg: &VitsConfig) -> Result<Vec<f32>> {
        let x = self.pre.forward(hidden)?;
        let x = self.dds.forward(&x)?;
        let cond = self.proj.forward(&x)?;
        let t = hidden.dim(2)?;
        let bins = cfg.duration_predictor_flow_bins;
        let [mut a, mut b] = noise;
        for flow in &self.flows {
            std::mem::swap(&mut a, &mut b); // torch.flip over the 2 channels
            match flow {
                DurationFlow::Affine(tr, ls) => {
                    a.iter_mut().for_each(|v| *v = (*v - tr[0]) * (-ls[0]).exp());
                    b.iter_mut().for_each(|v| *v = (*v - tr[1]) * (-ls[1]).exp());
                }
                DurationFlow::Conv(f) => {
                    let first = Tensor::from_slice(&a, (1, 1, t), cond.device())?.to_dtype(cond.dtype())?;
                    let h = f.pre.forward(&first)?;
                    let h = f.dds.forward(&(h + &cond)?)?;
                    // [1, 3k-1, t] -> host [t][3k-1]
                    let p = f.proj.forward(&h)?.squeeze(0)?.t()?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    for (v, p) in b.iter_mut().zip(&p) {
                        *v = rq_spline_inverse(*v, &p[..bins], &p[bins..2 * bins], &p[2 * bins..], cfg.duration_predictor_tail_bound);
                    }
                }
            }
        }
        Ok(a)
    }
}

/// Inverse of the unconstrained rational-quadratic spline of the duration
/// flows (identity outside `[-tail, tail]`), for one scalar. Widths and
/// heights are already divided by `sqrt(filter_channels)` (folded at convert).
fn rq_spline_inverse(x: f32, uw: &[f32], uh: &[f32], ud: &[f32], tail: f32) -> f32 {
    const MIN_W: f32 = 1e-3;
    const MIN_H: f32 = 1e-3;
    const MIN_D: f32 = 1e-3;
    if !(-tail..=tail).contains(&x) {
        return x;
    }
    let k = uw.len();
    let knots = |u: &[f32], min: f32| -> Vec<f32> {
        let m = u.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = u.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        let mut cum = vec![-tail];
        let mut acc = 0.0f32;
        for v in &e {
            acc += min + (1.0 - min * k as f32) * v / s;
            cum.push(2.0 * tail * acc - tail);
        }
        cum[k] = tail;
        cum
    };
    let cw = knots(uw, MIN_W);
    let ch = knots(uh, MIN_H);
    // Boundary derivatives are pinned to 1 (softplus(log(e^(1-min)-1)) + min).
    let softplus = |v: f32| if v > 20.0 { v } else { v.exp().ln_1p() };
    let deriv = |i: usize| if i == 0 || i == k { 1.0 } else { MIN_D + softplus(ud[i - 1]) };

    let mut bin = ch[..k].iter().take_while(|&&c| x >= c).count().saturating_sub(1);
    if x >= ch[k] + 1e-6 {
        bin = k - 1;
    }
    let bin = bin.min(k - 1);
    let (w, h) = (cw[bin + 1] - cw[bin], ch[bin + 1] - ch[bin]);
    let delta = h / w;
    let (d0, d1) = (deriv(bin), deriv(bin + 1));
    let i1 = d0 + d1 - 2.0 * delta;
    let i2 = x - ch[bin];
    let i3 = i2 * i1;
    let a = h * (delta - d0) + i3;
    let b = h * d0 - i3;
    let c = -delta * i2;
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let root = (2.0 * c) / (-b - disc.sqrt());
    root * w + cw[bin]
}

// ---------------------------------------------------------------------------
// Prior flows (reverse) and HiFi-GAN

struct WaveNet {
    in_layers: Vec<Conv>,
    res_skip: Vec<Conv>,
    hidden: usize,
}

impl WaveNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.hidden;
        let mut x = x.clone();
        let mut out: Option<Tensor> = None;
        let n = self.in_layers.len();
        for i in 0..n {
            let h = self.in_layers[i].forward(&x)?;
            let acts = (h.narrow(1, 0, c)?.tanh()? * sigmoid(&h.narrow(1, c, c)?)?)?;
            let rs = self.res_skip[i].forward(&acts)?;
            let skip = if i < n - 1 {
                x = (x + rs.narrow(1, 0, c)?)?;
                rs.narrow(1, c, c)?
            } else {
                rs
            };
            out = Some(match out {
                Some(o) => (o + skip)?,
                None => skip,
            });
        }
        out.context("empty WaveNet")
    }
}

struct Coupling {
    pre: Conv,
    wavenet: WaveNet,
    post: Conv,
}

struct HifiGan {
    conv_pre: Conv,
    /// (weight, bias, stride, padding); weights pre-divided by the number of
    /// resblock kernels of the previous stage.
    ups: Vec<(Tensor, Tensor, usize, usize)>,
    resblocks: Vec<Vec<(Conv, Conv)>>,
    conv_post: Conv,
    slope: f64,
}

/// HiFi-GAN is purely convolutional: decoding windows of `chunk` frames with
/// `context` frames of real neighbours on each side gives the same samples as
/// one pass, with VRAM bounded by the window instead of the sentence (im2col
/// expands the 64 k-sample stages by the kernel size, ~75 MB per second of
/// audio in f32 without this).
const DEC_CHUNK: usize = 384;
const DEC_CONTEXT: usize = 24;

impl HifiGan {
    fn forward_chunked(&self, z: &Tensor, hop: usize) -> Result<Tensor> {
        let chunk = std::env::var("MALAGA_VITS_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(DEC_CHUNK);
        let ctx = std::env::var("MALAGA_VITS_CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(DEC_CONTEXT);
        let frames = z.dim(2)?;
        if frames <= chunk + 2 * ctx {
            return self.forward(z);
        }
        let mut parts = vec![];
        let mut start = 0;
        while start < frames {
            let len = chunk.min(frames - start);
            let (a, b) = (start.saturating_sub(ctx), (start + len + ctx).min(frames));
            let y = self.forward(&z.narrow(2, a, b - a)?)?;
            parts.push(y.narrow(2, (start - a) * hop, len * hop)?);
            start += len;
        }
        Ok(Tensor::cat(&parts, 2)?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_pre.forward(x)?;
        let nk = self.resblocks.len() / self.ups.len();
        for (i, (w, b, stride, pad)) in self.ups.iter().enumerate() {
            x = leaky_relu(&x, self.slope)?;
            // padding = 0 hits candle's fast col2im path; trim afterwards.
            let y = x.conv_transpose1d(w, 0, 0, *stride, 1, 1)?;
            let len = y.dim(2)? - 2 * pad;
            x = y.narrow(2, *pad, len)?.broadcast_add(b)?;
            let mut acc: Option<Tensor> = None;
            for rb in &self.resblocks[i * nk..(i + 1) * nk] {
                let mut h = x.clone();
                for (c1, c2) in rb {
                    let r = leaky_relu(&h, self.slope)?;
                    let r = c1.forward(&r)?;
                    let r = leaky_relu(&r, self.slope)?;
                    h = (c2.forward(&r)? + h)?;
                }
                acc = Some(match acc {
                    Some(a) => (a + h)?,
                    None => h,
                });
            }
            x = acc.context("no resblocks")?; // the 1/nk is in the next conv
        }
        let x = leaky_relu(&x, 0.01)?; // F.leaky_relu default slope
        Ok(self.conv_post.forward(&x)?.tanh()?)
    }
}

/// Debug: `MALAGA_VITS_DUMP=dir` writes intermediate tensors as raw f32.
fn stage(dev: &Device, name: &str, t0: &mut std::time::Instant) {
    if std::env::var_os("MALAGA_VITS_PROFILE").is_some() {
        let _ = dev.synchronize();
        tracing::info!("{name:>8}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
        *t0 = std::time::Instant::now();
    }
}

fn dump(name: &str, t: &Tensor) {
    if let Ok(dir) = std::env::var("MALAGA_VITS_DUMP") {
        if let Ok(v) = t.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()) {
            let b: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            let _ = std::fs::write(Path::new(&dir).join(format!("{name}.f32")), b);
        }
    }
}

// ---------------------------------------------------------------------------

pub struct Vits {
    pub cfg: VitsConfig,
    tokenizer: CharTokenizer,
    name: String,
    device: Device,
    dtype: DType,
    enc: TextEncoder,
    dp: DurationPredictor,
    flows: Vec<Coupling>,
    dec: HifiGan,
    /// Malagasy front-end (only for Malagasy checkpoints).
    norm: Option<crate::mg_norm::MgNormalizer>,
}

/// Small deterministic normal sampler (xorshift64* + Box-Muller), so a seed
/// gives the same audio on every device.
struct Normal(u64);

impl Normal {
    fn new(seed: Option<u64>) -> Self {
        let s = seed.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
        });
        Self(s.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let (u1, u2) = (self.uniform(), self.uniform());
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32 * scale
            })
            .collect()
    }
}

impl Vits {
    /// Loads a GGUF written by `malaga convert` from a VITS checkpoint. `dtype`
    /// is the compute type (F32, or F16 on CUDA for the conv-heavy decoder).
    pub fn load(path: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let path = path.as_ref();
        let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let ct = Content::read(&mut f).with_context(|| format!("reading GGUF {}", path.display()))?;
        Self::from_content(&ct, path, device, dtype)
    }

    pub(crate) fn from_content(ct: &Content, path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let arch = get(ct, "general.architecture")?.to_string()?;
        anyhow::ensure!(arch == VITS_ARCH, "not a VITS model (architecture '{arch}')");
        let cfg = VitsConfig::from_hf_json(get(ct, "vits.config")?.to_string()?)?;
        let tokens: Vec<String> = get(ct, "tokenizer.ggml.tokens")?
            .to_vec()?
            .iter()
            .map(|v| v.to_string().cloned())
            .collect::<std::result::Result<_, _>>()?;
        let add_blank = get(ct, "vits.add_blank")?.to_bool()?;
        let tokenizer = CharTokenizer::new(&tokens, add_blank)?;
        let language = ct.metadata.get("vits.language").and_then(|v| v.to_string().ok().cloned()).unwrap_or_default();
        let norm = matches!(language.as_str(), "mlg" | "plt" | "mg").then(crate::mg_norm::MgNormalizer::default);
        let name = ct.metadata.get("general.name").and_then(|v| v.to_string().ok().cloned()).unwrap_or_default();

        let mut w = Loader { ct, file: std::fs::File::open(path)?, device: device.clone(), dtype };
        let mut layers = vec![];
        let fk = cfg.ffn_kernel_size;
        for i in 0..cfg.num_hidden_layers {
            let p = format!("enc.{i}");
            layers.push(EncoderLayer {
                qkv_w: w.get(&format!("{p}.attn_qkv.weight"))?,
                qkv_b: w.get(&format!("{p}.attn_qkv.bias"))?,
                out_w: w.get(&format!("{p}.attn_out.weight"))?,
                out_b: w.get(&format!("{p}.attn_out.bias"))?,
                rel_k: w.get(&format!("{p}.rel_k"))?,
                rel_v: w.get(&format!("{p}.rel_v"))?,
                attn_norm: w.norm(&format!("{p}.attn_norm"))?,
                ffn_up: w.conv(&format!("{p}.ffn_up"), (fk - 1) / 2, 1)?,
                ffn_down: w.conv(&format!("{p}.ffn_down"), (fk - 1) / 2, 1)?,
                ffn_norm: w.norm(&format!("{p}.ffn_norm"))?,
            });
        }
        anyhow::ensure!(fk % 2 == 1, "even ffn_kernel_size is not supported");
        let enc = TextEncoder { embed: w.get("enc.token_embd")?, layers, proj: w.conv("enc.proj", 0, 1)? };
        let dp = DurationPredictor::load(&mut w, &cfg)?;

        let mut flows = vec![];
        let (k, dr) = (cfg.wavenet_kernel_size, cfg.wavenet_dilation_rate);
        for i in 0..cfg.prior_encoder_num_flows {
            let p = format!("flow.{i}");
            let mut in_layers = vec![];
            let mut res_skip = vec![];
            for j in 0..cfg.prior_encoder_num_wavenet_layers {
                let d = dr.pow(j as u32);
                in_layers.push(w.conv(&format!("{p}.wavenet.in_layers.{j}"), (k * d - d) / 2, d)?);
                res_skip.push(w.conv(&format!("{p}.wavenet.res_skip_layers.{j}"), 0, 1)?);
            }
            flows.push(Coupling {
                pre: w.conv(&format!("{p}.conv_pre"), 0, 1)?,
                wavenet: WaveNet { in_layers, res_skip, hidden: cfg.hidden_size },
                post: w.conv(&format!("{p}.conv_post"), 0, 1)?,
            });
        }

        let mut ups = vec![];
        let mut resblocks = vec![];
        for (i, (&r, &ks)) in cfg.upsample_rates.iter().zip(&cfg.upsample_kernel_sizes).enumerate() {
            let wt = w.get(&format!("dec.upsampler.{i}.weight"))?;
            let b = w.get(&format!("dec.upsampler.{i}.bias"))?.reshape((1, (), 1))?;
            ups.push((wt, b, r, (ks - r) / 2));
            for (j, (&rk, dils)) in cfg.resblock_kernel_sizes.iter().zip(&cfg.resblock_dilation_sizes).enumerate() {
                let p = format!("dec.resblocks.{}", i * cfg.resblock_kernel_sizes.len() + j);
                let mut convs = vec![];
                for (m, &d) in dils.iter().enumerate() {
                    convs.push((
                        w.conv(&format!("{p}.convs1.{m}"), (rk * d - d) / 2, d)?,
                        w.conv(&format!("{p}.convs2.{m}"), (rk - 1) / 2, 1)?,
                    ));
                }
                resblocks.push(convs);
            }
        }
        let dec = HifiGan {
            conv_pre: w.conv("dec.conv_pre", 3, 1)?,
            ups,
            resblocks,
            conv_post: w.conv("dec.conv_post", 3, 1)?,
            slope: cfg.leaky_relu_slope,
        };
        Ok(Self { cfg, tokenizer, name, device: device.clone(), dtype, enc, dp, flows, dec, norm })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn sample_rate(&self) -> u32 {
        self.cfg.sampling_rate
    }

    pub fn tokenizer(&self) -> &CharTokenizer {
        &self.tokenizer
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Adds user lexicon entries (`word<TAB>respelling`) to the Malagasy front-end.
    pub fn load_lexicon(&mut self, path: &Path) -> Result<usize> {
        match &mut self.norm {
            Some(n) => n.load_lexicon(path),
            None => bail!("this model has no Malagasy text front-end"),
        }
    }

    /// The text the model will read, after normalization (for debugging).
    pub fn spoken_text(&self, text: &str, opts: &SynthOptions) -> String {
        match (&self.norm, opts.normalize) {
            (Some(n), true) => n.normalize(text),
            _ => text.to_string(),
        }
    }

    /// Synthesizes a document: text normalization, then one VITS pass per
    /// phrase (sentences split at `,`), joined with real silences. Returns mono
    /// f32 samples in [-1, 1].
    pub fn synthesize(&self, text: &str, opts: &SynthOptions) -> Result<Vec<f32>> {
        let mut rng = Normal::new(opts.seed);
        let silence = |ms: u32| vec![0f32; (self.cfg.sampling_rate as u64 * ms as u64 / 1000) as usize];
        let (sentence_pause, phrase_pause) = (silence(opts.sentence_pause_ms), silence(opts.phrase_pause_ms));
        let text = self.spoken_text(text, opts);
        let mut out = vec![];
        for seg in crate::segment::segment(&text) {
            let crate::segment::Segment::Text(sentence) = seg else { continue };
            let mut new_sentence = true;
            for phrase in sentence.split(',') {
                let ids = self.tokenizer.encode(phrase);
                if ids.len() <= 1 {
                    continue; // nothing pronounceable
                }
                if !out.is_empty() {
                    out.extend_from_slice(if new_sentence { &sentence_pause } else { &phrase_pause });
                }
                new_sentence = false;
                out.extend(self.synthesize_ids(&ids, opts, &mut rng)?);
            }
        }
        Ok(out)
    }

    /// Frames per token: ceil(exp(log_dur) / rate), with the vowel floors.
    fn durations(&self, ids: &[u32], log_dur: &[f32], opts: &SynthOptions) -> Vec<usize> {
        let length_scale = 1.0 / opts.speaking_rate.max(0.05);
        let frame_ms = self.cfg.hop() as f32 * 1000.0 / self.cfg.sampling_rate as f32;
        let floor = |ms: u32| (ms as f32 / frame_ms).round() as usize;
        let (vf, ff) = (floor(opts.vowel_floor_ms), floor(opts.final_vowel_floor_ms));
        let step = if self.tokenizer.add_blank { 2 } else { 1 };
        let space = self.tokenizer.space_id();
        log_dur
            .iter()
            .enumerate()
            .map(|(i, ld)| {
                let d = (ld.exp() * length_scale).ceil().max(0.0) as usize;
                if !self.tokenizer.is_char(i) || !self.tokenizer.is_vowel_id(ids[i]) {
                    return d;
                }
                let next = ids.get(i + step).copied();
                let word_final = next.is_none() || next == space;
                d.max(if word_final { ff } else { vf })
            })
            .collect()
    }

    fn synthesize_ids(&self, ids: &[u32], opts: &SynthOptions, rng: &mut Normal) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let t = ids.len();
        let mut t0 = std::time::Instant::now();
        let ids_t = Tensor::new(ids, &self.device)?;
        let (m, logs, hidden) = self.enc.forward(&ids_t, cfg)?;
        stage(&self.device, "encoder", &mut t0);
        dump("hidden", &hidden);
        dump("means", &m);

        let noise = [rng.vec(t, opts.noise_scale_duration), rng.vec(t, opts.noise_scale_duration)];
        let log_dur = self.dp.forward(&hidden, noise, cfg)?;
        stage(&self.device, "duration", &mut t0);
        dump("log_dur", &Tensor::new(log_dur.as_slice(), &Device::Cpu)?);
        // Length regulator: frame -> token index.
        let mut index = vec![];
        for (i, d) in self.durations(ids, &log_dur, opts).into_iter().enumerate() {
            index.extend(std::iter::repeat_n(i as u32, d));
        }
        if index.is_empty() {
            index.push(0);
        }
        let frames = index.len();
        let index = Tensor::from_vec(index, frames, &self.device)?;
        let m = m.index_select(&index, 2)?;
        let logs = logs.index_select(&index, 2)?;
        let eps = Tensor::from_vec(rng.vec(cfg.flow_size * frames, opts.noise_scale), (1, cfg.flow_size, frames), &self.device)?
            .to_dtype(self.dtype)?;
        let mut z = (m + (eps * logs.exp()?)?)?;

        // Reverse coupling flows; torch.flip over channels before each.
        let flip = Tensor::from_vec((0..cfg.flow_size as u32).rev().collect::<Vec<_>>(), cfg.flow_size, &self.device)?;
        let half = cfg.flow_size / 2;
        for f in self.flows.iter().rev() {
            z = z.index_select(&flip, 1)?;
            let first = z.narrow(1, 0, half)?;
            let h = f.pre.forward(&first)?;
            let h = f.wavenet.forward(&h)?;
            let mean = f.post.forward(&h)?;
            z = Tensor::cat(&[&first, &(z.narrow(1, half, half)? - mean)?], 1)?;
        }
        stage(&self.device, "flows", &mut t0);
        dump("z", &z);
        let wav = self.dec.forward_chunked(&z, cfg.hop())?;
        stage(&self.device, "hifigan", &mut t0);
        Ok(wav.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
    }
}

/// 16-bit PCM mono WAV.
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    b
}

/// Architecture of a GGUF file, without loading its tensors.
pub fn gguf_architecture(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let ct = Content::read(&mut f).with_context(|| format!("reading GGUF {}", path.display()))?;
    match ct.metadata.get("general.architecture") {
        Some(Value::String(s)) => Ok(s.clone()),
        _ => bail!("{} has no general.architecture", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> CharTokenizer {
        let v = ["a", "|", "n", "i", "y", "o", "r", "t", "m", "e", "h", "s", "k", "f", "z", "d", "l", "'", "v", "p", "b", "j", "-", "g", "à", "ỳ", "ô", "ò", "ì", " "];
        CharTokenizer::new(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>(), true).unwrap()
    }

    #[test]
    fn normalization_matches_mms_and_recovers_accents() {
        let t = tok();
        assert_eq!(t.normalize("Manao ahoana ianao?"), "manao ahoana ianao");
        // Decomposed ô (o + U+0302) is recomposed, French é falls back to e.
        assert_eq!(t.normalize("To\u{302}ko  Fêté!"), "tôko fete");
        assert_eq!(t.encode("na"), vec![0, 2, 0, 0, 0]);
    }

    #[test]
    fn spline_inverse_is_identity_for_flat_params() {
        // Equal bins and unit derivatives make the spline the identity.
        let z = [0f32; 10];
        let d = [((1.0f32 - 1e-3).exp() - 1.0).ln(); 9];
        for x in [-4.0f32, -0.3, 0.0, 1.7, 4.9] {
            assert!((rq_spline_inverse(x, &z, &z, &d, 5.0) - x).abs() < 1e-3, "{x}");
        }
    }
}
