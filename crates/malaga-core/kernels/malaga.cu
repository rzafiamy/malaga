// Fused CUDA kernels for NLLB decoding.
//
// Each kernel replaces a chain of 3-8 generic candle ops with a single launch.
// All tensors are contiguous f32 unless stated otherwise.
#include <stdint.h>
#include <cuda_fp16.h>

#define NEG_INF __int_as_float(0xff800000)
#define NO_ROW 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Block-wide reductions; blockDim.x must be a multiple of 32. `sh` holds 32 floats.
__device__ float block_sum(float v, float *sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
    if (w == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ float block_max(float v, float *sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : NEG_INF;
    if (w == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// Row-wise kernels keep the whole row in registers: LN_THREADS threads, up to
// LN_VPT values each (d <= 2048), so LayerNorm needs only two reductions.
#define LN_THREADS 512
#define LN_VPT 4

// Sum over a LN_THREADS block with one __syncthreads per call.
__device__ __forceinline__ float ln_block_sum(float v, float *sh) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < LN_THREADS / 32; ++i) t += sh[i];
    return t;
}

// v[] holds this thread's elements (index threadIdx.x + k*LN_THREADS).
__device__ __forceinline__ void ln_store(float (&v)[LN_VPT], int d, const float *gamma, const float *beta,
                                         float *out, float eps, float *sh) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < LN_VPT; ++k) s += v[k];
    const float mean = ln_block_sum(s, sh) / d;
    float s2 = 0.f;
#pragma unroll
    for (int k = 0; k < LN_VPT; ++k) {
        const int i = threadIdx.x + k * LN_THREADS;
        const float c = i < d ? v[k] - mean : 0.f;
        s2 += c * c;
    }
    const float rstd = rsqrtf(ln_block_sum(s2, sh) / d + eps);
#pragma unroll
    for (int k = 0; k < LN_VPT; ++k) {
        const int i = threadIdx.x + k * LN_THREADS;
        if (i < d) out[i] = (v[k] - mean) * rstd * gamma[i] + beta[i];
    }
}

// x_out = resid + y + bias ; ln_out = LayerNorm(x_out). One block per row.
extern "C" __global__ void __launch_bounds__(LN_THREADS)
bias_residual_ln(const float *y, const float *bias, const float *resid, const float *gamma, const float *beta,
                 float *x_out, float *ln_out, int d, float eps) {
    __shared__ float sh[32];
    const size_t off = (size_t)blockIdx.x * d;
    float v[LN_VPT];
#pragma unroll
    for (int k = 0; k < LN_VPT; ++k) {
        const int i = threadIdx.x + k * LN_THREADS;
        v[k] = 0.f;
        if (i < d) {
            v[k] = y[off + i] + bias[i] + resid[off + i];
            x_out[off + i] = v[k];
        }
    }
    ln_store(v, d, gamma, beta, ln_out + off, eps, sh);
}

// Decoder input: x = e * scale + pos_table[*step + base] ; ln_out = LayerNorm(x),
// where e is emb[row], or src_emb[row, src_row[row]] when src_row is not null
// and src_row[row] != NO_ROW (the previous token was copied from the source).
extern "C" __global__ void __launch_bounds__(LN_THREADS)
embed_pos_ln(const float *emb, const float *src_emb, const uint32_t *src_row, int S, const float *pos_table,
             const uint32_t *step, uint32_t base, float scale, const float *gamma, const float *beta,
             float *x_out, float *ln_out, int d, float eps) {
    __shared__ float sh[32];
    const int row = blockIdx.x;
    const size_t off = (size_t)row * d;
    const float *pos = pos_table + (size_t)(*step + base) * d;
    if (src_row && src_row[row] != NO_ROW) emb = src_emb + ((size_t)row * S + src_row[row]) * d - off;
    float v[LN_VPT];
#pragma unroll
    for (int k = 0; k < LN_VPT; ++k) {
        const int i = threadIdx.x + k * LN_THREADS;
        v[k] = 0.f;
        if (i < d) {
            v[k] = emb[off + i] * scale + pos[i];
            x_out[off + i] = v[k];
        }
    }
    ln_store(v, d, gamma, beta, ln_out + off, eps, sh);
}

// out = relu(y + bias), y is [n, d].
extern "C" __global__ void bias_relu(const float *y, const float *bias, float *out, int n, int d) {
    const size_t total = (size_t)n * d;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        out[i] = fmaxf(y[i] + bias[i % d], 0.f);
    }
}

// Splits a raw fused projection [b, 3*H*hd] (q|k|v), adds the bias, writes q to
// `q_out` [b, H, hd] and k/v into the f16 caches [b, H, m, hd] at slot *step.
extern "C" __global__ void qkv_decode(const float *qkv, const float *bias, float *q_out, __half *kc, __half *vc,
                                      const uint32_t *step, int H, int hd, int m) {
    const int bh = blockIdx.x, bi = bh / H, h = bh % H;
    const int d = H * hd;
    const float *r = qkv + (size_t)bi * 3 * d;
    const size_t slot = ((size_t)bh * m + *step) * hd;
    for (int t = threadIdx.x; t < hd; t += blockDim.x) {
        const int c = h * hd + t;
        q_out[(size_t)bh * hd + t] = r[c] + bias[c];
        kc[slot + t] = __float2half(r[d + c] + bias[d + c]);
        vc[slot + t] = __float2half(r[2 * d + c] + bias[2 * d + c]);
    }
}

// Single-query attention for one decoding step. One block per (batch, head).
//   q:      [b, H, hd] (+ q_bias [H*hd] if not null)
//   K, V:   [b, H, L, hd] in f16 (hd multiple of 8)
//   length: lens[b] if lens is not null (cross-attention), *step + 1 otherwise.
//   out:    [b, H*hd] (heads already merged)
// The 1/sqrt(hd) scaling is folded into the query projection. Each thread scores
// whole keys (16-byte loads, independent memory requests), then thread groups of
// `hd` threads split the keys for the weighted sum of V.
// Shared memory: (hd + L + groups*hd) floats.
extern "C" __global__ void attn_decode(const float *q, const float *q_bias, const __half *K, const __half *V,
                                       float *out, int H, int L, int hd, const uint32_t *lens,
                                       const uint32_t *step) {
    extern __shared__ float smem[];
    __shared__ float red[32];
    const int bh = blockIdx.x, bi = bh / H, h = bh % H;
    const int n = lens ? (int)lens[bi] : (int)(*step) + 1;
    const int groups = max(1, (int)blockDim.x / hd);
    float *qs = smem;            // hd
    float *sc = smem + hd;       // L
    float *part = sc + L;        // groups * hd
    for (int t = threadIdx.x; t < hd; t += blockDim.x)
        qs[t] = q[(size_t)bh * hd + t] + (q_bias ? q_bias[h * hd + t] : 0.f);
    __syncthreads();

    const __half *Kb = K + (size_t)bh * L * hd;
    const __half *Vb = V + (size_t)bh * L * hd;
    float mx = NEG_INF;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const uint4 *k8 = reinterpret_cast<const uint4 *>(Kb + (size_t)j * hd);
        float acc = 0.f;
#pragma unroll 8
        for (int t = 0; t < hd / 8; ++t) {
            const uint4 raw = k8[t];
            const __half2 *h2 = reinterpret_cast<const __half2 *>(&raw);
            const float *qq = qs + 8 * t;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 f = __half22float2(h2[u]);
                acc += f.x * qq[2 * u] + f.y * qq[2 * u + 1];
            }
        }
        sc[j] = acc;
        mx = fmaxf(mx, acc);
    }
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = __expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);  // also makes the sc writes visible
    const float inv = 1.f / sum;

    const int g = threadIdx.x / hd, t = threadIdx.x % hd;
    if (g < groups) {
        float a0 = 0.f, a1 = 0.f;
        int j = g;
        for (; j + groups < n; j += 2 * groups) {
            a0 += sc[j] * __half2float(Vb[(size_t)j * hd + t]);
            a1 += sc[j + groups] * __half2float(Vb[(size_t)(j + groups) * hd + t]);
        }
        if (j < n) a0 += sc[j] * __half2float(Vb[(size_t)j * hd + t]);
        part[g * hd + t] = a0 + a1;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float o = 0.f;
        for (int k = 0; k < groups; ++k) o += part[k * hd + c];
        out[(size_t)bi * H * hd + h * hd + c] = o * inv;
    }
}

// Logits of the source tokens: out[b, j] = dot(h[b], emb[b, j]), one warp per (b, j).
extern "C" __global__ void row_dots(const float *h, const float *emb, float *out, int b, int S, int d) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= b * S) return;
    const int bi = warp / S;
    const float4 *x = reinterpret_cast<const float4 *>(h + (size_t)bi * d);
    const float4 *e = reinterpret_cast<const float4 *>(emb + (size_t)warp * d);
    float acc = 0.f;
    for (int t = lane; t < d / 4; t += 32) {
        const float4 a = x[t], c = e[t];
        acc += a.x * c.x + a.y * c.y + a.z * c.z + a.w * c.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) out[warp] = acc;
}

// First argmax pass over the virtual row concat(x1[row, :V1], x2[row, :V2])
// (x2 may be null). Block (p, row) reduces elements p*chunk .. (p+1)*chunk.
extern "C" __global__ void argmax_partial(const float *x1, int V1, const float *x2, int V2, int chunk,
                                          float *pv, uint32_t *pi) {
    __shared__ float sv[32];
    __shared__ uint32_t si[32];
    const int row = blockIdx.y, p = blockIdx.x, P = gridDim.x;
    const int V = V1 + (x2 ? V2 : 0);
    const int start = p * chunk, end = min(start + chunk, V);
    const float *r1 = x1 + (size_t)row * V1;
    const float *r2 = x2 ? x2 + (size_t)row * V2 - V1 : nullptr;
    float best = NEG_INF;
    uint32_t bidx = 0xffffffffu;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const float v = i < V1 ? r1[i] : r2[i];
        if (v > best) { best = v; bidx = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, best, o);
        const uint32_t oi = __shfl_xor_sync(0xffffffff, bidx, o);
        if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bidx; }
    __syncthreads();
    if (w == 0) {
        best = lane < (blockDim.x >> 5) ? sv[lane] : NEG_INF;
        bidx = lane < (blockDim.x >> 5) ? si[lane] : 0xffffffffu;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, best, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffff, bidx, o);
            if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
        }
        if (lane == 0) { pv[row * P + p] = best; pi[row * P + p] = bidx; }
    }
}

// Second argmax pass fused with the greedy bookkeeping. Single block, one warp per row.
// The winning index i is mapped back to a vocabulary id:
//   i <  V1: vocab_map ? vocab_map[i] : i      (shortlist or full vocabulary)
//   i >= V1: src_ids[row, i - V1]              (tokens of the source sentence)
// then: tok = finished ? pad : tok ; finished |= tok == eos ; ids[row] = tok ;
// out[row, *step] = tok ; and finally ++*step.
// When sl_row/src_row are not null, they receive where the next input embedding
// lives: shortlist row (i < V1) or source position (i >= V1, else NO_ROW).
extern "C" __global__ void greedy_finalize(const float *pv, const uint32_t *pi, int P, int b, uint8_t *finished,
                                           uint32_t *ids, uint32_t *out, int m, uint32_t *step, uint32_t eos,
                                           uint32_t pad, const uint32_t *vocab_map, int V1,
                                           const uint32_t *src_ids, int V2, uint32_t *sl_row, uint32_t *src_row) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const uint32_t s = *step;
    for (int row = w; row < b; row += nw) {
        float best = NEG_INF;
        uint32_t bidx = 0xffffffffu;
        for (int p = lane; p < P; p += 32) {
            const float v = pv[row * P + p];
            const uint32_t i = pi[row * P + p];
            if (v > best || (v == best && i < bidx)) { best = v; bidx = i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, best, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffff, bidx, o);
            if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
        }
        if (lane == 0) {
            uint32_t tok;
            if ((int)bidx < V1) tok = vocab_map ? vocab_map[bidx] : bidx;
            else tok = src_ids[(size_t)row * V2 + (bidx - V1)];
            const bool done = finished[row];
            if (sl_row) {
                sl_row[row] = (!done && (int)bidx < V1) ? bidx : 0;
                src_row[row] = (!done && (int)bidx >= V1) ? bidx - V1 : NO_ROW;
            }
            if (done) tok = pad;
            if (tok == eos) finished[row] = 1;
            ids[row] = tok;
            out[(size_t)row * m + s] = tok;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) *step = s + 1;
}
