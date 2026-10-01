//! Launch wrappers for the fused kernels of `kernels/malaga.cu`.

use std::sync::RwLockReadGuard;

use candle_core::cuda::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda::{CudaDType, CudaDevice, CudaStorage, WrapErr};
use candle_core::{bail, CpuStorage, CustomOp1, Layout, Result, Shape, Storage, Tensor};
use half::f16;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/malaga.ptx"));
const NULL: u64 = 0;

/// Locked view on a contiguous CUDA tensor, kept alive for the launch.
struct Arg<'a> {
    guard: RwLockReadGuard<'a, Storage>,
    layout: &'a Layout,
}

impl<'a> Arg<'a> {
    fn new(t: &'a Tensor) -> Result<Self> {
        if !t.is_contiguous() {
            bail!("fused kernel expects contiguous inputs, got {:?}", t.layout())
        }
        let (guard, layout) = t.storage_and_layout();
        Ok(Self { guard, layout })
    }

    fn view<T: CudaDType>(&self) -> Result<CudaView<'_, T>> {
        match &*self.guard {
            Storage::Cuda(c) => Ok(c.as_cuda_slice::<T>()?.slice(self.layout.start_offset()..)),
            _ => bail!("fused kernel expects CUDA tensors"),
        }
    }
}

fn cuda_dev(t: &Tensor) -> Result<CudaDevice> {
    Ok(t.device().as_cuda_device()?.clone())
}

fn cfg(grid: (u32, u32), block: u32, shared: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (block, 1, 1), shared_mem_bytes: shared }
}

/// Threads per row of the LayerNorm kernels (`LN_THREADS` in malaga.cu); each
/// thread holds up to `LN_VPT` = 4 values, so `d <= 2048`.
const LN_THREADS: u32 = 512;
const LN_MAX_D: usize = 2048;

/// A kernel producing one new f32 tensor, run through candle's custom-op
/// machinery so that the output is a regular tensor.
struct NewTensor<F: Fn(&CudaDevice) -> Result<CudaStorage>> {
    shape: Shape,
    launch: F,
}

impl<F: Fn(&CudaDevice) -> Result<CudaStorage>> CustomOp1 for NewTensor<F> {
    fn name(&self) -> &'static str {
        "malaga-fused"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("fused kernels are CUDA only")
    }

    fn cuda_fwd(&self, s: &CudaStorage, _: &Layout) -> Result<(CudaStorage, Shape)> {
        Ok(((self.launch)(&s.device)?, self.shape.clone()))
    }
}

fn new_tensor<F: Fn(&CudaDevice) -> Result<CudaStorage>>(anchor: &Tensor, shape: Shape, launch: F) -> Result<Tensor> {
    anchor.apply_op1_no_bwd(&NewTensor { shape, launch })
}

fn alloc(dev: &CudaDevice, n: usize) -> Result<candle_core::cuda::cudarc::driver::CudaSlice<f32>> {
    // SAFETY: every kernel fully overwrites its output.
    unsafe { dev.alloc::<f32>(n) }
}

/// `x = resid + y + bias`, `ln = LayerNorm(x)`. Returns `(x, ln)` shaped like `y`.
pub fn bias_residual_ln(
    y: &Tensor,
    bias: &Tensor,
    resid: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let d = y.dim(candle_core::D::Minus1)?;
    if d > LN_MAX_D {
        bail!("hidden size {d} > {LN_MAX_D} is not supported by the fused kernels")
    }
    let n = y.elem_count() / d;
    let mut shape = vec![2];
    shape.extend_from_slice(y.dims());
    let both = new_tensor(y, shape.into(), |dev| {
        let (ya, ba, ra, ga, be) = (Arg::new(y)?, Arg::new(bias)?, Arg::new(resid)?, Arg::new(gamma)?, Arg::new(beta)?);
        let (yv, bv, rv, gv, bev) =
            (ya.view::<f32>()?, ba.view::<f32>()?, ra.view::<f32>()?, ga.view::<f32>()?, be.view::<f32>()?);
        let out = alloc(dev, 2 * n * d)?;
        {
            let (xo, lo) = (out.slice(..n * d), out.slice(n * d..));
            let f = dev.get_or_load_custom_func("bias_residual_ln", "malaga", PTX)?;
            let di = d as i32;
            let mut b = f.builder();
            b.arg(&yv).arg(&bv).arg(&rv).arg(&gv).arg(&bev).arg(&xo).arg(&lo).arg(&di).arg(&eps);
            unsafe { b.launch(cfg((n as u32, 1), LN_THREADS, 0)) }.w()?;
        }
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    })?;
    Ok((both.get(0)?, both.get(1)?))
}

/// Decoder input embedding: `x = e * scale + pos_table[step + base]`, `ln = LayerNorm(x)`,
/// where `e` is `emb[row]`, or `src_emb[row, src_row[row]]` when `src` is given and
/// `src_row[row] != u32::MAX`.
#[allow(clippy::too_many_arguments)]
pub fn embed_pos_ln(
    emb: &Tensor,
    src: Option<(&Tensor, &Tensor)>,
    pos_table: &Tensor,
    step: &Tensor,
    base: u32,
    scale: f32,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let (n, d) = emb.dims2()?;
    let s = match src {
        Some((e, _)) => e.dim(1)? as i32,
        None => 0,
    };
    let both = new_tensor(emb, (2, n, d).into(), |dev| {
        let (ea, pa, sa, ga, be) =
            (Arg::new(emb)?, Arg::new(pos_table)?, Arg::new(step)?, Arg::new(gamma)?, Arg::new(beta)?);
        let (ev, pv, sv, gv, bev) =
            (ea.view::<f32>()?, pa.view::<f32>()?, sa.view::<u32>()?, ga.view::<f32>()?, be.view::<f32>()?);
        let srca = src.map(|(e, r)| Ok::<_, candle_core::Error>((Arg::new(e)?, Arg::new(r)?))).transpose()?;
        let srcv = srca
            .as_ref()
            .map(|(e, r)| Ok::<_, candle_core::Error>((e.view::<f32>()?, r.view::<u32>()?)))
            .transpose()?;
        let out = alloc(dev, 2 * n * d)?;
        {
            let (xo, lo) = (out.slice(..n * d), out.slice(n * d..));
            let f = dev.get_or_load_custom_func("embed_pos_ln", "malaga", PTX)?;
            let di = d as i32;
            let mut b = f.builder();
            b.arg(&ev);
            match &srcv {
                Some((e, r)) => b.arg(e).arg(r),
                None => b.arg(&NULL).arg(&NULL),
            };
            b.arg(&s).arg(&pv).arg(&sv).arg(&base).arg(&scale).arg(&gv).arg(&bev).arg(&xo).arg(&lo).arg(&di).arg(&eps);
            unsafe { b.launch(cfg((n as u32, 1), LN_THREADS, 0)) }.w()?;
        }
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    })?;
    Ok((both.get(0)?, both.get(1)?))
}

/// `relu(y + bias)`.
pub fn bias_relu(y: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let d = y.dim(candle_core::D::Minus1)?;
    let total = y.elem_count();
    let n = total / d;
    new_tensor(y, y.shape().clone(), |dev| {
        let (ya, ba) = (Arg::new(y)?, Arg::new(bias)?);
        let (yv, bv) = (ya.view::<f32>()?, ba.view::<f32>()?);
        let out = alloc(dev, total)?;
        let f = dev.get_or_load_custom_func("bias_relu", "malaga", PTX)?;
        let (ni, di) = (n as i32, d as i32);
        let mut b = f.builder();
        b.arg(&yv).arg(&bv).arg(&out).arg(&ni).arg(&di);
        let grid = total.div_ceil(256).min(65535) as u32;
        unsafe { b.launch(cfg((grid, 1), 256, 0)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    })
}

/// Splits a raw `[b, 3*H*hd]` projection: returns `q` (`[b, H, hd]`, bias added) and
/// writes `k`, `v` into the f16 `[b, H, m, hd]` caches at slot `*step`.
pub fn qkv_decode(
    qkv: &Tensor,
    bias: &Tensor,
    kc: &Tensor,
    vc: &Tensor,
    step: &Tensor,
    heads: usize,
) -> Result<Tensor> {
    let (b, c) = qkv.dims2()?;
    let hd = c / (3 * heads);
    let m = kc.dim(2)?;
    new_tensor(qkv, (b, heads, hd).into(), |dev| {
        let (qa, ba, ka, va, sa) = (Arg::new(qkv)?, Arg::new(bias)?, Arg::new(kc)?, Arg::new(vc)?, Arg::new(step)?);
        let (qv, bv, kv, vv, sv) =
            (qa.view::<f32>()?, ba.view::<f32>()?, ka.view::<f16>()?, va.view::<f16>()?, sa.view::<u32>()?);
        let out = alloc(dev, b * heads * hd)?;
        let f = dev.get_or_load_custom_func("qkv_decode", "malaga", PTX)?;
        let (hi, hdi, mi) = (heads as i32, hd as i32, m as i32);
        let mut l = f.builder();
        l.arg(&qv).arg(&bv).arg(&out).arg(&kv).arg(&vv).arg(&sv).arg(&hi).arg(&hdi).arg(&mi);
        unsafe { l.launch(cfg(((b * heads) as u32, 1), (hd.div_ceil(32) * 32) as u32, 0)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    })
}

/// Single-query attention. `q`: `[b, H, hd]` f32, `k`/`v`: `[b, H, L, hd]` f16. The number
/// of valid keys is `lens[b]` (cross-attention) or `*step + 1` (self-attention).
/// Returns `[b, H*hd]`.
pub fn attn_decode(
    q: &Tensor,
    q_bias: Option<&Tensor>,
    k: &Tensor,
    v: &Tensor,
    lens: Option<&Tensor>,
    step: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, h, l, hd) = k.dims4()?;
    let q = q.reshape((b, h, hd))?;
    new_tensor(&q, (b, h * hd).into(), |dev| {
        let (qa, ka, va) = (Arg::new(&q)?, Arg::new(k)?, Arg::new(v)?);
        let (qv, kv, vv) = (qa.view::<f32>()?, ka.view::<f16>()?, va.view::<f16>()?);
        let qb = q_bias.map(Arg::new).transpose()?;
        let la = lens.map(Arg::new).transpose()?;
        let sa = step.map(Arg::new).transpose()?;
        let qbv = qb.as_ref().map(|a| a.view::<f32>()).transpose()?;
        let lv = la.as_ref().map(|a| a.view::<u32>()).transpose()?;
        let sv = sa.as_ref().map(|a| a.view::<u32>()).transpose()?;
        let out = alloc(dev, b * h * hd)?;
        let f = dev.get_or_load_custom_func("attn_decode", "malaga", PTX)?;
        let (hi, li, hdi) = (h as i32, l as i32, hd as i32);
        let block = (hd.div_ceil(32) * 32).max(256);
        let shared = 4 * (hd + l + (block / hd).max(1) * hd);
        let mut lb = f.builder();
        lb.arg(&qv);
        match &qbv {
            Some(x) => lb.arg(x),
            None => lb.arg(&NULL),
        };
        lb.arg(&kv).arg(&vv).arg(&out).arg(&hi).arg(&li).arg(&hdi);
        match &lv {
            Some(x) => lb.arg(x),
            None => lb.arg(&NULL),
        };
        match &sv {
            Some(x) => lb.arg(x),
            None => lb.arg(&NULL),
        };
        unsafe { lb.launch(cfg(((b * h) as u32, 1), block as u32, shared as u32)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    })
}

/// `out[b, j] = dot(h[b], emb[b, j])` for `h` `[b, d]` and `emb` `[b, S, d]`.
pub fn row_dots(h: &Tensor, emb: &Tensor) -> Result<Tensor> {
    let (b, s, d) = emb.dims3()?;
    new_tensor(h, (b, s).into(), |dev| {
        let (ha, ea) = (Arg::new(h)?, Arg::new(emb)?);
        let (hv, ev) = (ha.view::<f32>()?, ea.view::<f32>()?);
        let out = alloc(dev, b * s)?;
        let f = dev.get_or_load_custom_func("row_dots", "malaga", PTX)?;
        let (bi, si, di) = (b as i32, s as i32, d as i32);
        let mut l = f.builder();
        l.arg(&hv).arg(&ev).arg(&out).arg(&bi).arg(&si).arg(&di);
        let warps_per_block = 8;
        let grid = (b * s).div_ceil(warps_per_block) as u32;
        unsafe { l.launch(cfg((grid, 1), 32 * warps_per_block as u32, 0)) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    })
}

/// Device buffers updated by [`greedy_select`].
pub struct GreedyBuffers<'a> {
    pub finished: &'a Tensor,
    pub ids: &'a Tensor,
    pub out: &'a Tensor,
    pub step: &'a Tensor,
    pub eos: u32,
    pub pad: u32,
    /// Where the next input embedding lives (shortlist row, source position).
    pub rows: Option<(&'a Tensor, &'a Tensor)>,
}

/// Logits to choose from: `main` (`[b, V1]`, full vocabulary or a shortlist mapped
/// back through `vocab_map`) and optionally `src` (`[b, S]` logits of the source
/// tokens `src_ids`).
pub struct Logits<'a> {
    pub main: &'a Tensor,
    pub vocab_map: Option<&'a Tensor>,
    pub src: Option<(&'a Tensor, &'a Tensor)>,
}

/// Greedy token selection: two-pass parallel argmax, then updates `ids`,
/// `out[:, step]`, `finished` and increments `step`, all on the device.
pub fn greedy_select(logits: &Logits, g: &GreedyBuffers) -> Result<()> {
    const PARTS: usize = 64;
    let (b, v1) = logits.main.dims2()?;
    let v2 = match logits.src {
        Some((l, _)) => l.dim(1)?,
        None => 0,
    };
    let dev = cuda_dev(logits.main)?;
    let chunk = (v1 + v2).div_ceil(PARTS);
    // SAFETY: fully written by argmax_partial.
    let pv = unsafe { dev.alloc::<f32>(b * PARTS)? };
    let pi = unsafe { dev.alloc::<u32>(b * PARTS)? };
    let (v1i, v2i) = (v1 as i32, v2 as i32);
    {
        let la = Arg::new(logits.main)?;
        let lv = la.view::<f32>()?;
        let sa = logits.src.map(|(l, _)| Arg::new(l)).transpose()?;
        let sv = sa.as_ref().map(|a| a.view::<f32>()).transpose()?;
        let f = dev.get_or_load_custom_func("argmax_partial", "malaga", PTX)?;
        let ci = chunk as i32;
        let mut l = f.builder();
        l.arg(&lv).arg(&v1i);
        match &sv {
            Some(x) => l.arg(x),
            None => l.arg(&NULL),
        };
        l.arg(&v2i).arg(&ci).arg(&pv).arg(&pi);
        unsafe { l.launch(cfg((PARTS as u32, b as u32), 256, 0)) }.w()?;
    }
    let (fa, ia, oa, sa) = (Arg::new(g.finished)?, Arg::new(g.ids)?, Arg::new(g.out)?, Arg::new(g.step)?);
    let (fv, iv, ov, sv) = (fa.view::<u8>()?, ia.view::<u32>()?, oa.view::<u32>()?, sa.view::<u32>()?);
    let ma = logits.vocab_map.map(Arg::new).transpose()?;
    let mv = ma.as_ref().map(|a| a.view::<u32>()).transpose()?;
    let ida = logits.src.map(|(_, ids)| Arg::new(ids)).transpose()?;
    let idv = ida.as_ref().map(|a| a.view::<u32>()).transpose()?;
    let m = g.out.dim(1)? as i32;
    let (parts, bi) = (PARTS as i32, b as i32);
    let f = dev.get_or_load_custom_func("greedy_finalize", "malaga", PTX)?;
    let mut l = f.builder();
    l.arg(&pv).arg(&pi).arg(&parts).arg(&bi).arg(&fv).arg(&iv).arg(&ov).arg(&m).arg(&sv).arg(&g.eos).arg(&g.pad);
    match &mv {
        Some(x) => l.arg(x),
        None => l.arg(&NULL),
    };
    l.arg(&v1i);
    match &idv {
        Some(x) => l.arg(x),
        None => l.arg(&NULL),
    };
    l.arg(&v2i);
    let ra = g.rows.map(|(a, b)| Ok::<_, candle_core::Error>((Arg::new(a)?, Arg::new(b)?))).transpose()?;
    let rv = ra.as_ref().map(|(a, b)| Ok::<_, candle_core::Error>((a.view::<u32>()?, b.view::<u32>()?))).transpose()?;
    match &rv {
        Some((a, b)) => l.arg(a).arg(b),
        None => l.arg(&NULL).arg(&NULL),
    };
    let block = (b * 32).clamp(32, 1024) as u32;
    unsafe { l.launch(cfg((1, 1), block, 0)) }.w()?;
    Ok(())
}
