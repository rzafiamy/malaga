//! Greedy GPU decoding with fixed-shape buffers, replayed as CUDA graphs.
//!
//! Even with fused kernels a decoder step is ~250 launches; issued one by one
//! from the CPU they cost ~3 µs each, more than the GPU needs to run them. Here
//! every step reads and writes the same buffers (token, position, KV caches and
//! outputs live on the device), so the step is captured once into a CUDA graph
//! and replayed with a single launch per generated token.
//!
//! Shapes are bucketed (batch rounded up to a power of two with idle rows,
//! source length and output budget to powers of two) so that a handful of
//! graphs serve every request. Each graph pins ~32 MiB of graph memory: the
//! cache is bounded by a byte budget (`MALAGA_GRAPH_CACHE_MB`, default 512) and,
//! if the GPU runs out of memory (e.g. next to other models), graphs are
//! dropped and decoding falls back to running the same fused step eagerly.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::model::{Nllb, Shortlist};

/// Device buffers read and written by `Nllb::step_static`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct StaticState {
    pub b: usize,
    pub m: usize,
    /// Current input token, `[b, 1]` u32.
    pub ids: Tensor,
    /// Current decoder position (0-based), `[1]` u32, incremented on the device.
    pub step: Tensor,
    /// Self-attention caches per layer, `[b, h, m, hd]` f16.
    pub kv: Vec<(Tensor, Tensor)>,
    /// Encoder keys/values per layer, `[b, h, s, hd]` f16.
    pub cross: Vec<(Tensor, Tensor)>,
    /// Source lengths, `[b]` u32.
    pub lens: Tensor,
    pub finished: Tensor,
    /// Generated tokens, `[b, m]`; column `i` is produced by step `i`.
    pub out: Tensor,
    /// Reduced LM head of the target language.
    pub shortlist: Option<Shortlist>,
    /// Source token ids `[b, s]` and their embeddings `[b, s, d]`, scored next
    /// to the shortlist so that copied tokens stay reachable.
    pub src_ids: Tensor,
    pub src_emb: Tensor,
    /// Where the current input token's embedding is: shortlist row `[b]` u32, or
    /// source position `[b]` u32 (`u32::MAX` when not from the source).
    pub sl_row: Tensor,
    pub src_row: Tensor,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl StaticState {
    fn new(model: &Nllb, b: usize, s: usize, m: usize, shortlist: Option<Shortlist>) -> Result<Self> {
        let cfg = &model.cfg;
        let dev = model.device();
        let h = cfg.decoder_attention_heads;
        let hd = cfg.d_model / h;
        let zeros = |shape: (usize, usize, usize, usize)| Tensor::zeros(shape, DType::F16, dev);
        let mut kv = Vec::new();
        let mut cross = Vec::new();
        for _ in 0..model.decoder_layers() {
            kv.push((zeros((b, h, m, hd))?, zeros((b, h, m, hd))?));
            cross.push((zeros((b, h, s, hd))?, zeros((b, h, s, hd))?));
        }
        Ok(Self {
            b,
            m,
            ids: Tensor::zeros((b, 1), DType::U32, dev)?,
            step: Tensor::zeros(1, DType::U32, dev)?,
            kv,
            cross,
            lens: Tensor::zeros(b, DType::U32, dev)?,
            finished: Tensor::zeros(b, DType::U8, dev)?,
            out: Tensor::zeros((b, m), DType::U32, dev)?,
            shortlist,
            src_ids: Tensor::zeros((b, s), DType::U32, dev)?,
            src_emb: Tensor::zeros((b, s, cfg.d_model), DType::F32, dev)?,
            sl_row: Tensor::zeros(b, DType::U32, dev)?,
            src_row: Tensor::full(u32::MAX, b, dev)?,
        })
    }

    fn bytes(&self) -> usize {
        let kv: usize = self.kv.iter().map(|(k, _)| 2 * k.elem_count()).sum();
        let cross: usize = self.cross.iter().map(|(k, _)| 2 * k.elem_count()).sum();
        2 * (kv + cross) + 4 * self.src_emb.elem_count()
    }

    /// Loads a new batch: encoder K/V, source tokens, start token. Rows past
    /// `src.len()` are idle padding rows, marked finished from the start.
    fn reset(&self, model: &Nllb, cross: &[(Tensor, Tensor)], src: &[Vec<u32>], start: u32) -> Result<()> {
        let dev = self.ids.device();
        let n = src.len();
        for ((dk, dv), (k, v)) in self.cross.iter().zip(cross) {
            dk.narrow(0, 0, n)?.slice_set(&k.to_dtype(DType::F16)?, 2, 0)?;
            dv.narrow(0, 0, n)?.slice_set(&v.to_dtype(DType::F16)?, 2, 0)?;
        }
        if self.shortlist.is_some() {
            let s = self.src_ids.dim(1)?;
            let pad = model.cfg.pad_token_id;
            let ids: Vec<u32> = (0..self.b)
                .flat_map(|i| (0..s).map(move |j| src.get(i).and_then(|r| r.get(j)).copied().unwrap_or(pad)))
                .collect();
            let emb = model.embedding_rows(&ids)?.reshape(self.src_emb.shape())?;
            self.src_emb.slice_set(&emb, 0, 0)?;
            self.src_ids.slice_set(&Tensor::from_vec(ids, (self.b, s), dev)?, 0, 0)?;
        }
        let lens: Vec<u32> = (0..self.b).map(|i| src.get(i).map_or(1, |r| r.len() as u32)).collect();
        self.lens.slice_set(&Tensor::new(lens.as_slice(), dev)?, 0, 0)?;
        self.set_ids(start)?;
        self.step.slice_set(&Tensor::zeros(1, DType::U32, dev)?, 0, 0)?;
        self.reset_finished(src.len())
    }

    fn reset_finished(&self, active: usize) -> Result<()> {
        let f: Vec<u8> = (0..self.b).map(|i| u8::from(i >= active)).collect();
        Ok(self.finished.slice_set(&Tensor::new(f.as_slice(), self.ids.device())?, 0, 0)?)
    }

    /// Sets the next input token of every row.
    fn set_ids(&self, id: u32) -> Result<()> {
        let dev = self.ids.device();
        self.ids.slice_set(&Tensor::full(id, (self.b, 1), dev)?, 0, 0)?;
        if let Some(sl) = &self.shortlist {
            let row = sl.row(id).ok_or_else(|| anyhow::anyhow!("token {id} missing from the shortlist"))?;
            self.sl_row.slice_set(&Tensor::full(row, self.b, dev)?, 0, 0)?;
            self.src_row.slice_set(&Tensor::full(u32::MAX, self.b, dev)?, 0, 0)?;
        }
        Ok(())
    }

    fn all_finished(&self) -> Result<bool> {
        Ok(self.finished.min(0)?.to_scalar::<u8>()? == 1)
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn bucket(n: usize, min: usize, max: usize) -> usize {
    n.max(min).next_power_of_two().min(max).max(n)
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::cuda::cudarc::driver::{sys, CudaGraph};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    /// Graph memory pinned by one captured decoder step (allocation granularity).
    const GRAPH_BYTES: usize = 32 << 20;

    struct Graph(CudaGraph);
    // SAFETY: the graph is only launched while holding the cache mutex, and
    // cudarc binds the CUDA context to the calling thread before launching.
    unsafe impl Send for Graph {}

    /// (batch, source, output, shortlist language)
    type Key = (usize, usize, usize, Option<String>);

    struct Entry {
        key: Key,
        st: StaticState,
        graph: Option<Graph>,
        last_used: u64,
    }

    impl Entry {
        fn bytes(&self) -> usize {
            self.st.bytes() + if self.graph.is_some() { GRAPH_BYTES } else { 0 }
        }
    }

    pub(crate) struct GraphCache {
        entries: Mutex<(u64, Vec<Entry>)>,
        max_bytes: usize,
        /// Record steps as CUDA graphs (otherwise the fused static step runs
        /// eagerly). Turned off for good after an out-of-memory error.
        capture: AtomicBool,
    }

    impl GraphCache {
        pub fn new() -> Self {
            let max_mb = std::env::var("MALAGA_GRAPH_CACHE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(512usize);
            let capture = std::env::var("MALAGA_NO_CUDA_GRAPH").map(|v| v.is_empty() || v == "0").unwrap_or(true);
            Self { entries: Mutex::new((0, vec![])), max_bytes: max_mb << 20, capture: AtomicBool::new(capture) }
        }

        pub fn supports(&self, dev: &Device) -> bool {
            dev.is_cuda()
        }

        /// Frees every cached graph and buffer, and returns graph memory to the driver.
        pub fn clear(&self) {
            let mut guard = self.entries.lock().unwrap();
            let had_graphs = guard.1.iter().any(|e| e.graph.is_some());
            guard.1.clear();
            if had_graphs {
                trim_graph_memory();
            }
        }

        /// Greedy decoding of an encoded batch. Returns token ids without specials.
        #[allow(clippy::too_many_arguments)]
        pub fn greedy(
            &self,
            model: &Nllb,
            cross: &[(Tensor, Tensor)],
            src: &[Vec<u32>],
            tgt_lang: u32,
            shortlist: Option<(&str, &Shortlist)>,
            max_new: usize,
            sync_every: usize,
        ) -> Result<Vec<Vec<u32>>> {
            match self.run(model, cross, src, tgt_lang, shortlist, max_new, sync_every) {
                Err(e) if format!("{e:?}").contains("OUT_OF_MEMORY") && self.capture.load(Ordering::Relaxed) => {
                    tracing::warn!("out of GPU memory: dropping CUDA graphs, decoding without them from now on");
                    self.capture.store(false, Ordering::Relaxed);
                    self.clear();
                    self.run(model, cross, src, tgt_lang, shortlist, max_new, sync_every)
                }
                r => r,
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn run(
            &self,
            model: &Nllb,
            cross: &[(Tensor, Tensor)],
            src: &[Vec<u32>],
            tgt_lang: u32,
            shortlist: Option<(&str, &Shortlist)>,
            max_new: usize,
            sync_every: usize,
        ) -> Result<Vec<Vec<u32>>> {
            let cfg = &model.cfg;
            let capture = self.capture.load(Ordering::Relaxed);
            let n = src.len();
            let s = cross[0].0.dim(2)?;
            let max_m = cfg.max_position_embeddings;
            // Without graphs there is no point in padding the batch.
            let b = if capture { n.next_power_of_two() } else { n };
            let key = (b, bucket(s, 16, max_m), bucket(max_new + 2, 32, max_m), shortlist.map(|(c, _)| c.to_string()));
            let mut guard = self.entries.lock().unwrap();
            let (clock, entries) = &mut *guard;
            *clock += 1;
            let idx = match entries.iter().position(|e| e.key == key) {
                Some(i) => i,
                None => {
                    let st = StaticState::new(model, key.0, key.1, key.2, shortlist.map(|(_, s)| s.clone()))?;
                    entries.push(Entry { key: key.clone(), st, graph: None, last_used: 0 });
                    evict(entries, self.max_bytes, &key);
                    entries.iter().position(|e| e.key == key).expect("just inserted")
                }
            };
            let e = &mut entries[idx];
            e.last_used = *clock;
            let st = &e.st;
            let dev = model.device().as_cuda_device()?;
            let _cache = dev.enable_cuda_graph_htod_cache();

            // Step 0 consumes the decoder start token; its output is replaced by
            // the forced target-language token. Both eager steps also warm the
            // parameter caches that graph capture relies on.
            st.reset(model, cross, src, cfg.decoder_start_token_id)?;
            model.step_static(st)?;
            st.set_ids(tgt_lang)?;
            st.reset_finished(n)?;
            model.step_static(st)?;
            let mut steps = 2;
            let max_steps = (max_new + 1).min(st.m);

            if capture && e.graph.is_none() && steps < max_steps {
                let stream = dev.cuda_stream();
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let captured = model.step_static(st);
                let graph = stream
                    .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
                captured?;
                let graph = graph.ok_or_else(|| anyhow::anyhow!("empty CUDA graph"))?;
                graph.upload()?;
                e.graph = Some(Graph(graph));
                tracing::debug!("captured decoder graph for {key:?}");
            }
            while steps < max_steps {
                if steps % sync_every == 0 && st.all_finished()? {
                    break;
                }
                match &e.graph {
                    Some(Graph(g)) => g.launch()?,
                    None => model.step_static(st)?,
                }
                steps += 1;
            }
            let out = st.out.narrow(0, 0, n)?.narrow(1, 1, steps - 1)?.to_vec2::<u32>()?;
            let eos = cfg.eos_token_id;
            Ok(out.into_iter().map(|r| r.into_iter().take_while(|&t| t != eos).collect()).collect())
        }
    }

    /// Drops least recently used entries (never `keep`) until under budget.
    fn evict(entries: &mut Vec<Entry>, max_bytes: usize, keep: &Key) {
        let mut dropped_graph = false;
        loop {
            let total: usize = entries.iter().map(Entry::bytes).sum::<usize>() + GRAPH_BYTES;
            if total <= max_bytes || entries.len() <= 1 {
                break;
            }
            let victim = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| &e.key != keep)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i);
            match victim {
                Some(i) => dropped_graph |= entries.remove(i).graph.is_some(),
                None => break,
            }
        }
        if dropped_graph {
            trim_graph_memory();
        }
    }

    /// Returns the memory of destroyed graphs to the driver.
    fn trim_graph_memory() {
        let mut dev: sys::CUdevice = 0;
        // SAFETY: plain driver calls on the current context's device.
        unsafe {
            if sys::cuCtxGetDevice(&mut dev) == sys::CUresult::CUDA_SUCCESS {
                sys::cuDeviceGraphMemTrim(dev);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda::GraphCache;

/// Stub used when CUDA support is not compiled in.
#[cfg(not(feature = "cuda"))]
pub(crate) struct GraphCache;

#[cfg(not(feature = "cuda"))]
impl GraphCache {
    pub fn new() -> Self {
        Self
    }

    pub fn supports(&self, _dev: &Device) -> bool {
        false
    }

    pub fn clear(&self) {}

    #[allow(clippy::too_many_arguments)]
    pub fn greedy(
        &self,
        _model: &Nllb,
        _cross: &[(Tensor, Tensor)],
        _src: &[Vec<u32>],
        _tgt_lang: u32,
        _shortlist: Option<(&str, &Shortlist)>,
        _max_new: usize,
        _sync_every: usize,
    ) -> Result<Vec<Vec<u32>>> {
        anyhow::bail!("GPU decoding requires the `cuda` feature")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn buckets() {
        assert_eq!(super::bucket(5, 16, 1024), 16);
        assert_eq!(super::bucket(17, 16, 1024), 32);
        assert_eq!(super::bucket(64, 32, 1024), 64);
        assert_eq!(super::bucket(900, 32, 512), 900);
    }
}
