//! High level translation API: tokenization, segmentation, batching and decoding.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::Content;
use candle_core::{DType, Device, Tensor, D};
use tokenizers::Tokenizer;

use crate::config::get;
use crate::graph::GraphCache;
use crate::model::{DecoderState, Nllb};
use crate::segment::{segment, Segment};

#[derive(Debug, Clone)]
pub struct GenOptions {
    /// 1 = greedy decoding (fastest, Hugging Face's default for NLLB).
    pub beam_size: usize,
    /// Hard cap on generated tokens per sentence. `None` = `2 * src_len + 16`,
    /// bounded by the model's `max_length`.
    pub max_new_tokens: Option<usize>,
    /// Beam search length penalty (score = logprob / len^penalty).
    pub length_penalty: f32,
    /// Greedy decoding only checks for end-of-sequence every `sync_every` steps,
    /// letting the CPU queue kernels ahead of the GPU instead of waiting on it.
    pub sync_every: usize,
    /// Maximum number of sentences decoded together.
    pub max_batch: usize,
    /// Opt-in speed mode: restrict the output vocabulary to the target
    /// language's shortlist (plus the source tokens) when the model has one.
    /// Faster and lighter on VRAM, but NOT exact: ~3% of FLORES fr->mg
    /// sentences change because a rarer token (loanwords, acronyms, numbers)
    /// is missing from the shortlist. Off by default (GPU greedy only).
    pub shortlist: bool,
    /// GPU greedy decoding through fused kernels and CUDA graphs. `false`
    /// forces the generic candle implementation, kept as the reference the
    /// fused path is tested against.
    pub fused: bool,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            beam_size: 1,
            max_new_tokens: None,
            length_penalty: 1.0,
            sync_every: 4,
            max_batch: 32,
            shortlist: false,
            fused: true,
        }
    }
}

pub struct Translator {
    model: Nllb,
    tokenizer: Tokenizer,
    name: String,
    graphs: GraphCache,
}

impl Translator {
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let ct = Content::read(&mut f).with_context(|| format!("reading GGUF {}", path.display()))?;
        let tok_json = get(&ct, "tokenizer.huggingface.json")?.to_string()?;
        let tokenizer = Tokenizer::from_bytes(tok_json.as_bytes()).map_err(anyhow::Error::msg)?;
        let name =
            ct.metadata.get("general.name").and_then(|v| v.to_string().ok().cloned()).unwrap_or_else(|| "nllb".into());
        let model = Nllb::load(&ct, path, device)?;
        Ok(Self { model, tokenizer, name, graphs: GraphCache::new() })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn model(&self) -> &Nllb {
        &self.model
    }

    /// Token id of an NLLB language code (`fra_Latn`) or alias (`fr`).
    pub fn lang_id(&self, lang: &str) -> Result<u32> {
        let code = crate::langs::resolve(lang);
        self.tokenizer.token_to_id(&code).with_context(|| format!("unknown language '{lang}' (NLLB code '{code}')"))
    }

    /// Source token ids: `[src_lang] tokens [eos]`.
    pub fn encode(&self, text: &str, src: u32) -> Result<Vec<u32>> {
        let enc = self.tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
        let max = self.model.cfg.max_position_embeddings - 2;
        let mut ids = Vec::with_capacity(enc.len() + 2);
        ids.push(src);
        ids.extend(enc.get_ids().iter().take(max).copied());
        ids.push(self.model.cfg.eos_token_id);
        Ok(ids)
    }

    /// Translates a whole document: lines are kept, each line is split into
    /// sentences and all sentences are decoded in length-sorted batches.
    pub fn translate(&self, text: &str, src: &str, tgt: &str, opts: &GenOptions) -> Result<String> {
        Ok(self.translate_docs(&[text], src, tgt, opts)?.remove(0))
    }

    /// Like [`Self::translate`] for several documents at once: the sentences of
    /// all documents share the same batches.
    pub fn translate_docs(&self, docs: &[&str], src: &str, tgt: &str, opts: &GenOptions) -> Result<Vec<String>> {
        let segs: Vec<Vec<Segment>> = docs.iter().map(|d| segment(d)).collect();
        let sentences: Vec<&str> = segs
            .iter()
            .flatten()
            .filter_map(|s| match s {
                Segment::Text(t) => Some(*t),
                Segment::Sep(_) => None,
            })
            .collect();
        let mut translated = self.translate_batch(&sentences, src, tgt, opts)?.into_iter();
        Ok(segs
            .into_iter()
            .zip(docs)
            .map(|(segs, doc)| {
                let mut out = String::with_capacity(doc.len() * 2);
                for s in segs {
                    match s {
                        Segment::Text(_) => out.push_str(&translated.next().unwrap_or_default()),
                        Segment::Sep(sep) => out.push_str(sep),
                    }
                }
                out
            })
            .collect())
    }

    /// Translates independent sentences. Output order matches input order.
    pub fn translate_batch(&self, texts: &[&str], src: &str, tgt: &str, opts: &GenOptions) -> Result<Vec<String>> {
        let tgt_code = crate::langs::resolve(tgt);
        let (src, tgt) = (self.lang_id(src)?, self.lang_id(tgt)?);
        let encoded = texts.iter().map(|t| self.encode(t, src)).collect::<Result<Vec<_>>>()?;
        // Sort by length so that each batch carries as little padding as possible.
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(encoded[i].len()));
        let mut out = vec![String::new(); texts.len()];
        let mut max_batch = opts.max_batch.max(1);
        let mut done = 0;
        while done < order.len() {
            let chunk = &order[done..(done + max_batch).min(order.len())];
            let batch: Vec<Vec<u32>> = chunk.iter().map(|&i| encoded[i].clone()).collect();
            let ids = match self.generate_to(&batch, tgt, &tgt_code, opts) {
                Ok(ids) => ids,
                // Sharing the GPU with other models: free our cached buffers and
                // retry with smaller batches instead of failing the request.
                Err(e) if is_oom(&e) && max_batch > 1 && !format!("{e:#}").contains("full-vocabulary") => {
                    max_batch /= 2;
                    tracing::warn!("out of GPU memory, retrying with batches of {max_batch}");
                    self.graphs.clear();
                    continue;
                }
                Err(e) => return Err(e),
            };
            for (&i, ids) in chunk.iter().zip(ids) {
                out[i] = self.tokenizer.decode(&ids, true).map_err(anyhow::Error::msg)?;
            }
            done += chunk.len();
        }
        Ok(out)
    }

    /// Generates target token ids (without special tokens) for already tokenized sources.
    pub fn generate(&self, src: &[Vec<u32>], tgt_lang: u32, opts: &GenOptions) -> Result<Vec<Vec<u32>>> {
        let code = self.tokenizer.id_to_token(tgt_lang).unwrap_or_default();
        self.generate_to(src, tgt_lang, &code, opts)
    }

    fn generate_to(&self, src: &[Vec<u32>], tgt_lang: u32, tgt_code: &str, opts: &GenOptions) -> Result<Vec<Vec<u32>>> {
        if src.is_empty() {
            return Ok(vec![]);
        }
        // Each sentence gets its own length budget, so that the output of a
        // sentence never depends on the other sentences of its batch.
        let cap = self.model.cfg.max_length.min(self.model.cfg.max_position_embeddings - 2);
        let budget = |len: usize| opts.max_new_tokens.unwrap_or(2 * len + 16).clamp(1, cap);
        let max_new = src.iter().map(|s| budget(s.len())).max().unwrap_or(1);
        let mut out = if opts.beam_size <= 1 && opts.fused && self.graphs.supports(self.model.device()) {
            let (cross, _) = self.model.encode_cross(src)?;
            // The shortlist must contain the forced start tokens (decoder start and
            // target language), which `malaga convert` guarantees.
            let start = [self.model.cfg.decoder_start_token_id, tgt_lang];
            let shortlist = self
                .model
                .shortlist(tgt_code)
                .filter(|s| opts.shortlist && start.iter().all(|&t| s.contains(t)))
                .map(|s| (tgt_code, s));
            self.graphs.greedy(&self.model, &cross, src, tgt_lang, shortlist, max_new, opts.sync_every.max(1))?
        } else {
            let mut st = self.model.encode(src, max_new)?;
            if opts.beam_size <= 1 {
                self.greedy(&mut st, src.len(), tgt_lang, max_new, opts.sync_every.max(1))?
            } else {
                self.beam_search(st, src.len(), tgt_lang, max_new, opts)?
            }
        };
        for (row, s) in out.iter_mut().zip(src) {
            row.truncate(budget(s.len()));
        }
        Ok(out)
    }

    fn start_tokens(&self, b: usize, tgt_lang: u32) -> Result<Tensor> {
        let start = [self.model.cfg.decoder_start_token_id, tgt_lang];
        Ok(Tensor::new(&start, self.model.device())?.unsqueeze(0)?.repeat((b, 1))?)
    }

    fn greedy(
        &self,
        st: &mut DecoderState,
        b: usize,
        tgt_lang: u32,
        max_new: usize,
        sync_every: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let dev = self.model.device();
        let cfg = &self.model.cfg;
        let eos = Tensor::full(cfg.eos_token_id, b, dev)?;
        let pad = Tensor::full(cfg.pad_token_id, b, dev)?;
        let mut finished = Tensor::zeros(b, DType::U8, dev)?;
        let mut logits = self.model.decode(&self.start_tokens(b, tgt_lang)?, st)?;
        let mut steps = Vec::with_capacity(max_new);
        for step in 1..=max_new {
            // Everything stays on the device: no host round-trip per token.
            let next = finished.where_cond(&pad, &logits.argmax(D::Minus1)?)?;
            finished = finished.maximum(&next.eq(&eos)?)?;
            steps.push(next.clone());
            if step == max_new || (step % sync_every == 0 && finished.min(0)?.to_scalar::<u8>()? == 1) {
                break;
            }
            logits = self.model.decode(&next.unsqueeze(1)?, st)?;
        }
        let tokens = Tensor::stack(&steps, 1)?.to_vec2::<u32>()?;
        Ok(tokens.into_iter().map(|row| self.strip(row)).collect())
    }

    fn strip(&self, row: Vec<u32>) -> Vec<u32> {
        let eos = self.model.cfg.eos_token_id;
        row.into_iter().take_while(|&t| t != eos).collect()
    }

    /// Batched beam search. Top-k candidates are selected on the device; only
    /// `b * k * k` scores are copied back per step.
    fn beam_search(
        &self,
        mut st: DecoderState,
        b: usize,
        tgt_lang: u32,
        max_new: usize,
        opts: &GenOptions,
    ) -> Result<Vec<Vec<u32>>> {
        let dev = self.model.device().clone();
        let k = opts.beam_size;
        let eos = self.model.cfg.eos_token_id;
        let expand: Vec<u32> = (0..b as u32).flat_map(|i| std::iter::repeat_n(i, k)).collect();
        st.reorder(&Tensor::new(expand.as_slice(), &dev)?)?;

        // Alive beams per sentence: (score, tokens). Only beam 0 starts alive.
        let mut alive: Vec<Vec<(f32, Vec<u32>)>> = vec![vec![(0.0, vec![])]; b];
        let mut done: Vec<Vec<(f32, Vec<u32>)>> = vec![vec![]; b];
        let mut is_done = vec![false; b];
        let mut logits = self.model.decode(&self.start_tokens(b * k, tgt_lang)?, &mut st)?;
        let penalty = |len: usize| (len.max(1) as f32).powf(opts.length_penalty);

        for step in 1..=max_new {
            let (vals, idx) = topk(&Nllb::log_probs(&logits)?, k)?;
            let (vals, idx) = (vals.to_vec2::<f32>()?, idx.to_vec2::<u32>()?);
            let mut reorder = Vec::with_capacity(b * k);
            let mut next = Vec::with_capacity(b * k);
            for s in 0..b {
                let mut cands: Vec<(f32, usize, u32)> = Vec::with_capacity(k * k);
                for (bi, (score, _)) in alive[s].iter().enumerate() {
                    let row = s * k + bi;
                    for j in 0..k {
                        cands.push((score + vals[row][j], bi, idx[row][j]));
                    }
                }
                cands.sort_by(|a, c| c.0.total_cmp(&a.0));
                let mut new_alive = Vec::with_capacity(k);
                for (score, bi, tok) in cands {
                    if new_alive.len() == k {
                        break;
                    }
                    let mut toks = alive[s][bi].1.clone();
                    if tok == eos || step == max_new {
                        if tok != eos {
                            toks.push(tok);
                        }
                        let len = toks.len() + 1;
                        if !is_done[s] {
                            done[s].push((score / penalty(len), toks));
                        }
                        continue;
                    }
                    toks.push(tok);
                    reorder.push((s * k + bi) as u32);
                    next.push(tok);
                    new_alive.push((score, toks));
                }
                // Stop when the best finished hypothesis can no longer be beaten.
                let best_done = done[s].iter().map(|d| d.0).fold(f32::NEG_INFINITY, f32::max);
                let best_alive = new_alive.first().map(|a| a.0 / penalty(step + 1)).unwrap_or(f32::NEG_INFINITY);
                if done[s].len() >= k && best_done >= best_alive {
                    is_done[s] = true;
                }
                // Keep the batch rectangular: pad with copies of the first beam.
                while new_alive.len() < k {
                    reorder.push((s * k) as u32);
                    next.push(eos);
                    new_alive.push((f32::NEG_INFINITY, vec![]));
                }
                alive[s] = new_alive;
            }
            if step == max_new || is_done.iter().all(|&d| d) {
                break;
            }
            st.reorder(&Tensor::new(reorder.as_slice(), &dev)?)?;
            let next = Tensor::new(next.as_slice(), &dev)?.unsqueeze(1)?;
            logits = self.model.decode(&next, &mut st)?;
        }
        Ok(done
            .into_iter()
            .zip(alive)
            .map(|(d, a)| {
                let d = if d.is_empty() { a } else { d };
                d.into_iter().max_by(|x, y| x.0.total_cmp(&y.0)).map(|x| x.1).unwrap_or_default()
            })
            .collect())
    }
}

fn is_oom(e: &anyhow::Error) -> bool {
    format!("{e:?}").contains("OUT_OF_MEMORY")
}

/// Top-`k` values/indices along the last dim via `k` argmax passes (`k` is small).
fn topk(x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let n = x.dim(D::Minus1)?;
    let cols = Tensor::arange(0u32, n as u32, x.device())?.unsqueeze(0)?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.broadcast_as(x.shape())?;
    let mut x = x.clone();
    let (mut vals, mut idxs) = (Vec::with_capacity(k), Vec::with_capacity(k));
    for i in 0..k {
        let idx = x.argmax_keepdim(D::Minus1)?;
        vals.push(x.gather(&idx, D::Minus1)?);
        if i + 1 < k {
            x = cols.broadcast_eq(&idx)?.where_cond(&neg_inf, &x)?;
        }
        idxs.push(idx);
    }
    Ok((Tensor::cat(&vals, D::Minus1)?, Tensor::cat(&idxs, D::Minus1)?))
}
