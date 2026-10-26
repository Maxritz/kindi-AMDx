// kernels_hip.h - RDNA2 kernels for the Kindi engine, written for gfx1031.
//
// Design constraints, all measured (HIP-PORT.md):
//   - exact fp32 arithmetic; no int8 activations (dp4a destroyed the ranking: 1 of the top
//     20 rows survived)
//   - weights stay packed in VRAM at 1.75 bits, decoded in registers
//   - wave32 native; width from rows per wave (RPW), never wave64
//   - 64 KB LDS per workgroup, 128 KB per WGP
//
// PTQ1_0 block layout (128 weights in 28 bytes):
//   bytes 0..23  trit bytes, five base-3 digits per byte
//   bytes 24..25 the fourth digit group's high part
//   bytes 26..27 fp16 block scale
//   trit i is in byte j at digit n:
//     i <  80: j = i & 15,        n = i >> 4
//     i < 120: j = 16 + ((i-80)&7), n = (i-80) >> 3
//     else:    j = 24 + ((i-120)&1), n = (i-120) >> 1
//   digit: q = (raw * 3^n) & 0xFF ; trit = (q * 3) >> 8
#pragma once
#include <hip/hip_runtime.h>

#define RPW 4          // rows per wave: 4 on a 32-wide wave, matching one row of a 64-wide

__device__ __forceinline__ float fp16_to_f32(uint bits) {
    uint sign = (bits >> 15) & 1u, exp = (bits >> 10) & 0x1Fu, mant = bits & 0x3FFu;
    if (exp == 0u) { float v = (float)mant / 1024.0f * (1.0f / 16384.0f); return sign ? -v : v; }
    if (exp == 31u) return sign ? -3.0e38f : 3.0e38f;
    // fp16 has a bias of 15, not 16. Using 16 scales every block by 2x, which reads as
    // a ~100% error against the reference and is easy to miss because the numbers stay
    // plausible.
    float v = (1.0f + (float)mant / 1024.0f) * exp2f((float)((int)exp - 15));
    return sign ? -v : v;
}

// Build the byte -> five trit floats table in LDS. One entry per thread; 256 threads cover
// all 256 byte values. 5 KB of the 64 KB available.
__device__ __forceinline__ void build_trit_table(float* tl, int tid) {
    if (tid < 256) {
        uint b = (uint)tid;
        uint p = 1u;
        for (int n = 0; n < 5; ++n) {
            uint tr = (((b * p) & 0xFFu) * 3u) >> 8;
            tl[b * 5u + (uint)n] = (float)((int)tr - 1);
            p *= 3u;
        }
    }
    __syncthreads();
}

// ------------------------------------------------------------------------- PTQ1_0 GEMV
// y[row] = sum_c w[row][c] * x[c], RPW rows per wave, exact fp32.
__global__ void gemv_ptq1(const uint* __restrict__ w,
                          const float* __restrict__ x,
                          float* __restrict__ y,
                          int R, int C)
{
    __shared__ float tl[1280];
    build_trit_table(tl, threadIdx.x);

    int ws = warpSize;
    int nsub = blockDim.x / ws;
    int sub = threadIdx.x / ws;
    int lane = threadIdx.x % ws;

    int row0 = (blockIdx.x * nsub + sub) * RPW;
    int r0 = row0, r1 = row0 + 1, r2 = row0 + 2, r3 = row0 + 3;
    // Clamp, never return: __syncthreads() must be reached by every thread.
    bool k0 = r0 < R, k1 = r1 < R, k2 = r2 < R, k3 = r3 < R;
    if (!k0) r0 = 0;
    if (!k1) r1 = 0;
    if (!k2) r2 = 0;
    if (!k3) r3 = 0;

    int bpr = C >> 7;
    int ntb = bpr * 26;

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int bi = lane; bi < ntb; bi += ws) {
        int blk = bi / 26;
        int off = bi - blk * 26;
        int base = blk << 7;

        int wa = (r0 * bpr + blk) * 28 + off;
        int wb = (r1 * bpr + blk) * 28 + off;
        int wc = (r2 * bpr + blk) * 28 + off;
        int wd = (r3 * bpr + blk) * 28 + off;
        uint ra = (w[wa >> 2] >> (8u * (uint)(wa & 3))) & 0xFFu;
        uint rb = (w[wb >> 2] >> (8u * (uint)(wb & 3))) & 0xFFu;
        uint rc = (w[wc >> 2] >> (8u * (uint)(wc & 3))) & 0xFFu;
        uint rd = (w[wd >> 2] >> (8u * (uint)(wd & 3))) & 0xFFu;

        // scale: bytes 26..27, and a block starts at a multiple of 28 so byte 26 is the
        // high half of word (base/4 + 6)
        float sa = fp16_to_f32(w[((r0 * bpr + blk) * 7) + 6] >> 16);
        float sb = fp16_to_f32(w[((r1 * bpr + blk) * 7) + 6] >> 16);
        float sc = fp16_to_f32(w[((r2 * bpr + blk) * 7) + 6] >> 16);
        float sd = fp16_to_f32(w[((r3 * bpr + blk) * 7) + 6] >> 16);

        int ba = (int)ra * 5, bb = (int)rb * 5, bc = (int)rc * 5, bd = (int)rd * 5;
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;

        if (off < 16) {
            int xi = base + off;
            #pragma unroll
            for (int n = 0; n < 5; ++n) {
                float xv = x[xi + (n << 4)];
                p0 += tl[ba + n] * xv;
                p1 += tl[bb + n] * xv;
                p2 += tl[bc + n] * xv;
                p3 += tl[bd + n] * xv;
            }
        } else if (off < 24) {
            int xi = base + 80 + (off - 16);
            #pragma unroll
            for (int n = 0; n < 5; ++n) {
                float xv = x[xi + (n << 3)];
                p0 += tl[ba + n] * xv;
                p1 += tl[bb + n] * xv;
                p2 += tl[bc + n] * xv;
                p3 += tl[bd + n] * xv;
            }
        } else {
            int xi = base + 120 + (off - 24);
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                float xv = x[xi + (n << 1)];
                p0 += tl[ba + n] * xv;
                p1 += tl[bb + n] * xv;
                p2 += tl[bc + n] * xv;
                p3 += tl[bd + n] * xv;
            }
        }
        a0 += sa * p0; a1 += sb * p1; a2 += sc * p2; a3 += sd * p3;
    }

    #pragma unroll
    for (int o = ws / 2; o > 0; o >>= 1) {
        a0 += __shfl_down(a0, o);
        a1 += __shfl_down(a1, o);
        a2 += __shfl_down(a2, o);
        a3 += __shfl_down(a3, o);
    }
    if (lane == 0) {
        if (k0) y[r0] = a0;
        if (k1) y[r1] = a1;
        if (k2) y[r2] = a2;
        if (k3) y[r3] = a3;
    }
}

// Three PTQ1_0 tensors sharing one input, one launch. Row space is the concatenation
// [0,R0) [R0,R0+R1) [R0+R1,R0+R1+R2). Groups are laid out per tensor so a wave's rows
// always share one weight buffer (per-byte buffer selection cost 0.68x on the Vulkan side).
#define DO_TENSOR(W, Y, G, RR)                                                  \
    {                                                                           \
        int row0 = (G) * RPW;                                                   \
        int r0 = row0, r1 = row0 + 1, r2 = row0 + 2, r3 = row0 + 3;             \
        bool k0 = r0 < (RR), k1 = r1 < (RR), k2 = r2 < (RR), k3 = r3 < (RR);    \
        if (!k0) r0 = 0;                                                        \
        if (!k1) r1 = 0;                                                        \
        if (!k2) r2 = 0;                                                        \
        if (!k3) r3 = 0;                                                        \
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;                           \
        for (int bi = lane; bi < ntb; bi += ws) {                               \
            int blk = bi / 26;                                                  \
            int off = bi - blk * 26;                                            \
            int base = blk << 7;                                                \
            int wa = (r0 * bpr + blk) * 28 + off;                               \
            int wb = (r1 * bpr + blk) * 28 + off;                               \
            int wc = (r2 * bpr + blk) * 28 + off;                               \
            int wd = (r3 * bpr + blk) * 28 + off;                               \
            uint ra = (W[wa >> 2] >> (8u * (uint)(wa & 3))) & 0xFFu;            \
            uint rb = (W[wb >> 2] >> (8u * (uint)(wb & 3))) & 0xFFu;            \
            uint rc = (W[wc >> 2] >> (8u * (uint)(wc & 3))) & 0xFFu;            \
            uint rd = (W[wd >> 2] >> (8u * (uint)(wd & 3))) & 0xFFu;            \
            float sa = fp16_to_f32(W[((r0 * bpr + blk) * 7) + 6] >> 16);        \
            float sb = fp16_to_f32(W[((r1 * bpr + blk) * 7) + 6] >> 16);        \
            float sc = fp16_to_f32(W[((r2 * bpr + blk) * 7) + 6] >> 16);        \
            float sd = fp16_to_f32(W[((r3 * bpr + blk) * 7) + 6] >> 16);        \
            int ba = (int)ra * 5, bb = (int)rb * 5, bc = (int)rc * 5, bd = (int)rd * 5; \
            float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;                       \
            if (off < 16) {                                                     \
                int xi = base + off;                                            \
                _Pragma("unroll")                                               \
                for (int n = 0; n < 5; ++n) {                                   \
                    float xv = x[xi + (n << 4)];                                \
                    p0 += tl[ba + n] * xv; p1 += tl[bb + n] * xv;               \
                    p2 += tl[bc + n] * xv; p3 += tl[bd + n] * xv;               \
                }                                                               \
            } else if (off < 24) {                                              \
                int xi = base + 80 + (off - 16);                                \
                _Pragma("unroll")                                               \
                for (int n = 0; n < 5; ++n) {                                   \
                    float xv = x[xi + (n << 3)];                                \
                    p0 += tl[ba + n] * xv; p1 += tl[bb + n] * xv;               \
                    p2 += tl[bc + n] * xv; p3 += tl[bd + n] * xv;               \
                }                                                               \
            } else {                                                            \
                int xi = base + 120 + (off - 24);                               \
                _Pragma("unroll")                                               \
                for (int n = 0; n < 4; ++n) {                                   \
                    float xv = x[xi + (n << 1)];                                \
                    p0 += tl[ba + n] * xv; p1 += tl[bb + n] * xv;               \
                    p2 += tl[bc + n] * xv; p3 += tl[bd + n] * xv;               \
                }                                                               \
            }                                                                   \
            a0 += sa * p0; a1 += sb * p1; a2 += sc * p2; a3 += sd * p3;         \
        }                                                                       \
        _Pragma("unroll")                                                       \
        for (int o = ws / 2; o > 0; o >>= 1) {                                  \
            a0 += __shfl_down(a0, o); a1 += __shfl_down(a1, o);                 \
            a2 += __shfl_down(a2, o); a3 += __shfl_down(a3, o);                 \
        }                                                                       \
        if (lane == 0) {                                                        \
            if (k0) Y[r0] = a0; if (k1) Y[r1] = a1;                             \
            if (k2) Y[r2] = a2; if (k3) Y[r3] = a3;                             \
        }                                                                       \
    }

__global__ void gemv_ptq1_3(const uint* __restrict__ W0, float* __restrict__ Y0,
                            const uint* __restrict__ W1, float* __restrict__ Y1,
                            const uint* __restrict__ W2, float* __restrict__ Y2,
                            const float* __restrict__ x,
                            int R0, int R1, int R2, int C)
{
    __shared__ float tl[1280];
    build_trit_table(tl, threadIdx.x);

    int ws = warpSize;
    int nsub = blockDim.x / ws;
    int sub = threadIdx.x / ws;
    int lane = threadIdx.x % ws;
    int bpr = C >> 7;
    int ntb = bpr * 26;

    int g0 = (R0 + RPW - 1) / RPW;
    int g1 = (R1 + RPW - 1) / RPW;
    int g2 = (R2 + RPW - 1) / RPW;
    int gid = blockIdx.x * nsub + sub;

    if (gid < g0)                  { DO_TENSOR(W0, Y0, gid, R0) }
    else if (gid < g0 + g1)        { DO_TENSOR(W1, Y1, gid - g0, R1) }
    else if (gid < g0 + g1 + g2)   { DO_TENSOR(W2, Y2, gid - g0 - g1, R2) }
}

// ------------------------------------------------------------------------- embedding
// token_embd is PTQ1_0 with dims [D, VOCAB]: row = token id, cols = D. The Hadamard is
// applied INVERSE here (prism.hadamard.inverse_weight_names), so the stored row is the
// folded embedding and we must un-fold: multiply by signs, then FWHT, then scale.
// The engine instead asks the reference for the primal vector, which is what the GEMV of
// the next layer consumes, so this kernel reproduces the inverse transform.
__global__ void deq_emb(const uint* __restrict__ w, const int* __restrict__ tok,
                        const float* __restrict__ signs, float* __restrict__ out,
                        int D, int HB, float hbscale)
{
    extern __shared__ float xs[];   // HB floats
    // One workgroup per HB block, so every thread is in range and no thread exits
    // before a __syncthreads().
    int blk = blockIdx.x;
    int row = tok[0];
    int nblk = D / HB;
    if (blk >= nblk) return;

    int bpr = D >> 7;
    int c0 = blk * HB;
    // decode this HB-block worth of weights
    for (int i = threadIdx.x; i < HB; i += blockDim.x) {
        int c = c0 + i;
        int bb = c >> 7;
        int blkoff = (row * bpr + bb) * 28;
        int trit_i = c & 127;
        int j, n;
        if (trit_i < 80)        { j = trit_i & 15;              n = trit_i >> 4; }
        else if (trit_i < 120)  { j = 16 + ((trit_i - 80) & 7); n = (trit_i - 80) >> 3; }
        else                    { j = 24 + ((trit_i - 120) & 1); n = (trit_i - 120) >> 1; }
        int bo = blkoff + j;
        uint raw = (w[bo >> 2] >> (8u * (uint)(bo & 3))) & 0xFFu;
        uint p = 1u;
        for (int t = 0; t < n; ++t) p *= 3u;
        uint q = (raw * p) & 0xFFu;
        int trit = (int)((q * 3u) >> 8);
        float sc = fp16_to_f32(w[((row * bpr + bb) * 7) + 6] >> 16);
        // The dequantised weight goes into the transform WITHOUT the signs. The reference
        // (deq_emb5) applies them after the FWHT: h[e] = signs[e] * (fwht(x)[e] * scale).
        // Applying them before the transform gives a different, wrong basis.
        xs[i] = (float)(trit - 1) * sc;
    }
    __syncthreads();

    // FWHT over HB, in place, shared memory
    for (int h = 1; h < HB; h <<= 1) {
        for (int t = threadIdx.x; t < (HB >> 1); t += blockDim.x) {
            int i = (t / h) * 2 * h + (t % h);
            float a = xs[i], b = xs[i + h];
            xs[i] = a + b;
            xs[i + h] = a - b;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < HB; i += blockDim.x)
        out[c0 + i] = signs[c0 + i] * (xs[i] * hbscale);
}

// ------------------------------------------------------------------------- rmsfold
// rmsnorm(h + residual) -> xn (normalised) and xo (folded: signs then FWHT then scale).
// xn feeds the gate scalars, xo feeds the next GEMV.
__global__ void rmsfold(const float* __restrict__ x,   // h (residual target)
                        const float* __restrict__ r,   // residual to add (may be null)
                        const float* __restrict__ wgt, // rms weight
                        const float* __restrict__ signs,
                        float* __restrict__ xn,
                        float* __restrict__ xo,
                        int C, int HB, float eps, float hbscale, bool addres)
{
    extern __shared__ float xs[];
    __shared__ float bscale;
    int ws = warpSize, nsub = blockDim.x / ws;
    int sub = threadIdx.x / ws, lane = threadIdx.x % ws;

    // 1. sum of squares in fp32 (double on the Vulkan side; fp32 with a 512-wide tree is
    //    accurate enough at C=2048 and avoids the fp64 rate on RDNA2)
    float acc = 0.f;
    for (int e = threadIdx.x; e < C; e += blockDim.x) {
        float v = x[e] + (addres ? r[e] : 0.f);
        acc += v * v;
    }
    #pragma unroll
    for (int o = ws / 2; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    __shared__ float part[32];
    if (lane == 0) part[sub] = acc;
    __syncthreads();
    if (sub == 0) {
        float t = (lane < nsub) ? part[lane] : 0.f;
        #pragma unroll
        for (int o = ws / 2; o > 0; o >>= 1) t += __shfl_down(t, o);
        if (lane == 0) bscale = 1.f / sqrtf(t / (float)C + eps);
    }
    __syncthreads();
    float s = bscale;

    int blk = blockIdx.x;
    int c0 = blk * HB;
    for (int i = threadIdx.x; i < HB; i += blockDim.x) {
        int e = c0 + i;
        float v = x[e] + (addres ? r[e] : 0.f);
        float n = (v * s) * wgt[e];
        xn[e] = n;
        xs[i] = signs[e] * n;
    }
    __syncthreads();
    for (int h = 1; h < HB; h <<= 1) {
        for (int t = threadIdx.x; t < (HB >> 1); t += blockDim.x) {
            int i = (t / h) * 2 * h + (t % h);
            float a = xs[i], b = xs[i + h];
            xs[i] = a + b;
            xs[i + h] = a - b;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < HB; i += blockDim.x) xo[c0 + i] = xs[i] * hbscale;
}

// ------------------------------------------------------------------------- gdn_gates
// Reference semantics (k_sgdn_gates), one workgroup per value head:
//   ab = sum_i beta_w[h*D + i] * xn[i]
//   aa = sum_i alpha_w[h*D + i] * xn[i]
//   beta[h] = sigmoid(ab)
//   a  = aa + dt_bias[h]                       <- ADD, not multiply
//   sp = softplus(a)                           <- softplus of a itself
//   g[h] = sp * ssm_a[h]                       <- g is a LOG decay; delta does exp(g)
// GGUF dims for these weights are [D, G], so element (i, h) is at h*D + i.
__global__ void gdn_gates(const float* __restrict__ xn,
                          const float* __restrict__ beta_w,   // [G][D]
                          const float* __restrict__ alpha_w,  // [G][D]
                          const float* __restrict__ dt_bias,  // [G]
                          const float* __restrict__ a_raw,    // [G]
                          float* __restrict__ gb, float* __restrict__ gg,
                          int D, int G)
{
    int h = blockIdx.x;
    float ab = 0.f, aa = 0.f;
    const float* bw = beta_w + (size_t)h * D;
    const float* aw = alpha_w + (size_t)h * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float x = xn[i];
        ab += bw[i] * x;
        aa += aw[i] * x;
    }
    __shared__ float sb[256], sa[256];
    sb[threadIdx.x] = ab;
    sa[threadIdx.x] = aa;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) { sb[threadIdx.x] += sb[threadIdx.x + st];
                                sa[threadIdx.x] += sa[threadIdx.x + st]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gb[h] = 1.f / (1.f + expf(-sb[0]));
        float a = sa[0] + dt_bias[h];
        float sp = (a > 20.f) ? a : log1pf(expf(fminf(a, 20.f)));
        gg[h] = sp * a_raw[h];
    }
}

// ------------------------------------------------------------------------- gdn_conv
// Causal depthwise conv over CONV_CH channels, kernel 4, then SiLU. The weight tensor is
// [4, CONV_CH], so tap t of channel c is at cw[4*c + t] (dims[0]=4 is the fast axis).
// `conv` is the rolling three-token history: conv[c], conv[C+c], conv[2C+c].
__global__ void gdn_conv(const float* __restrict__ qkv,
                         float* __restrict__ conv,        // [3][CH] rolling history
                         const float* __restrict__ cw,    // [CH][4]
                         float* __restrict__ out,         // [CH]
                         int CH)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CH) return;
    float c0 = conv[c], c1 = conv[CH + c], c2 = conv[2 * CH + c];
    float nw = qkv[c];
    float o = c0 * cw[4 * c + 0] + c1 * cw[4 * c + 1]
            + c2 * cw[4 * c + 2] + nw * cw[4 * c + 3];
    conv[c] = c1;
    conv[CH + c] = c2;
    conv[2 * CH + c] = nw;
    out[c] = o / (1.f + expf(-o));
}

// ------------------------------------------------------------------------- l2w
// L2-normalise `rows` rows of width n from x into out. The value part must stay
// un-normalised, so this writes to a separate buffer rather than in place.
__global__ void l2w(const float* __restrict__ x, float* __restrict__ out,
                    int n, int rows)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* p = x + (size_t)r * n;
    float acc = 0.f;
    for (int i = 0; i < n; ++i) acc += p[i] * p[i];
    float s = 1.f / fmaxf(sqrtf(acc), 1e-6f);
    float* o = out + (size_t)r * n;
    for (int i = 0; i < n; ++i) o[i] = p[i] * s;
}

// ------------------------------------------------------------------------- gdn_delta
// The gated delta rule, exactly as k_sgdn_delta:
//   e  = exp(g[h])                       <- g is a log decay
//   sk = sum_i (M[i][j] * e) * ks[i]
//   dd = (v[h*HK + j] - sk) * beta[h]
//   m  = M[i][j] * e + ks[i] * dd ;  M[i][j] = m ;  o += m * qs[i]
//   out[h*HK + j] = o * scale
// One workgroup per value head, HK threads. q, k and v are separate pointers because
// only q and k are L2-normalised; v is the raw conv output.
__global__ void gdn_delta(float* __restrict__ M,
                          const float* __restrict__ q,   // [NK][HK] normalised
                          const float* __restrict__ k,   // [NK][HK] normalised
                          const float* __restrict__ v,   // [NV][HK] raw
                          const float* __restrict__ g,   // [NV] log decay
                          const float* __restrict__ beta,
                          float* __restrict__ out,
                          int HK, int NV, int NK, float scale)
{
    int h = blockIdx.x;
    int j = threadIdx.x;
    if (j >= HK) return;
    int kh = h % NK;
    float e = expf(g[h]);
    float* Mh = M + (size_t)h * HK * HK;

    float sk = 0.f;
    for (int i = 0; i < HK; ++i) sk += (Mh[(size_t)i * HK + j] * e) * k[(size_t)kh * HK + i];
    float dd = (v[(size_t)h * HK + j] - sk) * beta[h];

    float o = 0.f;
    for (int i = 0; i < HK; ++i) {
        float m = Mh[(size_t)i * HK + j] * e + k[(size_t)kh * HK + i] * dd;
        Mh[(size_t)i * HK + j] = m;
        o += m * q[(size_t)kh * HK + i];
    }
    out[(size_t)h * HK + j] = o * scale;
}

// ------------------------------------------------------------------------- rmsw
// Per-row RMS normalisation with a weight, no residual, no fold.
__global__ void rmsw(const float* __restrict__ x, const float* __restrict__ wgt,
                     float* __restrict__ out, int n, int rows, float eps)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* p = x + (size_t)r * n;
    float acc = 0.f;
    for (int i = 0; i < n; ++i) acc += p[i] * p[i];
    float s = 1.f / sqrtf(acc / (float)n + eps);
    float* o = out + (size_t)r * n;
    for (int i = 0; i < n; ++i) o[i] = p[i] * s * wgt[i];
}

// ------------------------------------------------------------------------- smfold
// silu(gate) * up, then signs, then FWHT, then scale. This is the SwiGLU plus the fold in
// ONE pass, because RDNA2 has 64 KB of LDS and the whole HB-block fits.
__global__ void smfold(const float* __restrict__ gate, const float* __restrict__ up,
                       const float* __restrict__ signs,
                       float* __restrict__ out,
                       int HB, float hbscale, int C)
{
    extern __shared__ float xs[];
    int blk = blockIdx.x;
    int nblk = C / HB;
    if (blk >= nblk) return;
    int c0 = blk * HB;
    for (int i = threadIdx.x; i < HB; i += blockDim.x) {
        float g = gate[c0 + i];
        float m = (g / (1.f + expf(-g))) * up[c0 + i];
        xs[i] = signs[c0 + i] * m;
    }
    __syncthreads();
    for (int h = 1; h < HB; h <<= 1) {
        for (int t = threadIdx.x; t < (HB >> 1); t += blockDim.x) {
            int i = (t / h) * 2 * h + (t % h);
            float a = xs[i], b = xs[i + h];
            xs[i] = a + b;
            xs[i + h] = a - b;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < HB; i += blockDim.x) out[c0 + i] = xs[i] * hbscale;
}

// ------------------------------------------------------------------------- rope
// One head per block, HD-wide.
__global__ void rope(float* __restrict__ x, const float* __restrict__ cs,
                     const float* __restrict__ sn, int HD, int pos, int nrot)
{
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (i >= nrot) return;
    float* p = x + (size_t)h * HD;
    float a = p[i], b = p[i + nrot];
    float c = cs[(size_t)pos * nrot + i], s = sn[(size_t)pos * nrot + i];
    p[i] = a * c - b * s;
    p[i + nrot] = a * s + b * c;
}

// ------------------------------------------------------------------------- softmax
__global__ void softmax_rows(float* __restrict__ s, int n)
{
    int r = blockIdx.x;
    float* p = s + (size_t)r * n;
    float mx = -1e30f;
    for (int i = 0; i < n; ++i) mx = fmaxf(mx, p[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { p[i] = expf(p[i] - mx); sum += p[i]; }
    float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) p[i] *= inv;
}

// ------------------------------------------------------------------------- attn_comb
// out[head][i] = sum_j score[head][j] * v[j][i]
__global__ void attn_comb(const float* __restrict__ sc, const float* __restrict__ v,
                          float* __restrict__ out, int npos, int NH, int NHKV,
                          int HD, int G)
{
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (i >= HD) return;
    int kvh = h / G;
    const float* s = sc + (size_t)h * npos;
    float acc = 0.f;
    for (int j = 0; j < npos; ++j)
        acc += s[j] * v[((size_t)j * NHKV + kvh) * HD + i];
    out[(size_t)h * HD + i] = acc;
}

// ------------------------------------------------------------------------- argmax
__global__ void argmax(const float* __restrict__ x, int* __restrict__ out, int n)
{
    __shared__ float bv[256];
    __shared__ int bi[256];
    float best = -1e30f;
    int besti = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (v > best) { best = v; besti = i; }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = besti;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (bv[threadIdx.x + s] > bv[threadIdx.x] ||
                (bv[threadIdx.x + s] == bv[threadIdx.x] && bi[threadIdx.x + s] < bi[threadIdx.x])) {
                bv[threadIdx.x] = bv[threadIdx.x + s];
                bi[threadIdx.x] = bi[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = bi[0];
}

// ------------------------------------------------------------------------- addv
__global__ void addv(float* __restrict__ a, const float* __restrict__ b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// ------------------------------------------------------------------------- cpq
// The fused q projection is [q | gate] per head. Copy the q half out.
__global__ void cpq(const float* __restrict__ qn, float* __restrict__ qr, int HD)
{
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (i >= HD) return;
    qr[(size_t)h * HD + i] = qn[(size_t)h * 2 * HD + i];
}

// ------------------------------------------------------------------------- attn_scores
// score[h][j] = q[h] . K[j][kvh] * scale, for the G query heads sharing KV head kvh.
// One block per (kv head, group of 32 positions); 32 threads cover one head's HD.
__global__ void attn_scores(const float* __restrict__ Kc, const float* __restrict__ q,
                            float* __restrict__ sc, int n, int NHKV, int G, int HD,
                            float scale)
{
    int kvh = blockIdx.x;
    int j = blockIdx.y * 32 + threadIdx.x;
    if (j >= n) return;
    const float* k = Kc + ((size_t)j * NHKV + kvh) * HD;
    for (int g = 0; g < G; ++g) {
        int h = kvh * G + g;
        const float* qh = q + (size_t)h * HD;
        // one thread per position computes the whole head's dot product; HD is 256, which
        // is 256 FMAs per lane. Fine for decode, and it keeps the summation order simple.
        float acc = 0.f;
        for (int i = 0; i < HD; ++i) acc += qh[i] * k[i];
        sc[(size_t)h * n + j] = acc * scale;
    }
}

// ------------------------------------------------------------------------- attn_gate
// att[hh][i] *= sigmoid(gate[hh][i]), where the gate lives in the second half of qn's
// per-head block.
__global__ void attn_gate(float* __restrict__ att, const float* __restrict__ qn, int HD)
{
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (i >= HD) return;
    float g = qn[(size_t)h * 2 * HD + HD + i];
    att[(size_t)h * HD + i] *= 1.f / (1.f + expf(-g));
}

// ------------------------------------------------------------------------- fold_fwht
// signs, then FWHT, then scale. Same as smfold's tail but from a single input.
__global__ void fold_fwht(const float* __restrict__ x, const float* __restrict__ signs,
                          float* __restrict__ out, int HB, float hbscale, int C)
{
    extern __shared__ float xs[];
    int blk = blockIdx.x;
    int nblk = C / HB;
    if (blk >= nblk) return;
    int c0 = blk * HB;
    for (int i = threadIdx.x; i < HB; i += blockDim.x) xs[i] = signs[c0 + i] * x[c0 + i];
    __syncthreads();
    for (int h = 1; h < HB; h <<= 1) {
        for (int t = threadIdx.x; t < (HB >> 1); t += blockDim.x) {
            int i = (t / h) * 2 * h + (t % h);
            float a = xs[i], b = xs[i + h];
            xs[i] = a + b;
            xs[i + h] = a - b;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < HB; i += blockDim.x) out[c0 + i] = xs[i] * hbscale;
}
