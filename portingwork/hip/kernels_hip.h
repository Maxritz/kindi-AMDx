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
    int lane = threadIdx.x % warpSize;
    int ws = warpSize;
    int nsub = blockDim.x / ws;
    int sub = threadIdx.x / ws;
    int blk = blockIdx.x * nsub + sub;      // which HB block of the row
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
        xs[i] = (float)(trit - 1) * sc * signs[c];
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
        out[c0 + i] = xs[i] * hbscale;
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
// beta = sigmoid(beta_w^T xn), gate = softplus-ish from alpha/dt/a, per value head.
// Matches the reference: g = exp(-softplus(a) * dt) with a from alpha*xn, dt from bias.
__global__ void gdn_gates(const float* __restrict__ xn,
                          const float* __restrict__ beta_w,   // [D][G]
                          const float* __restrict__ alpha_w,  // [D][G]
                          const float* __restrict__ dt_bias,  // [G]
                          const float* __restrict__ a_raw,    // [G]
                          float* __restrict__ gb, float* __restrict__ gg,
                          int D, int G)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    float bsum = 0.f, asum = 0.f;
    for (int e = 0; e < D; ++e) {
        bsum += xn[e] * beta_w[e * G + g];
        asum += xn[e] * alpha_w[e * G + g];
    }
    float beta = 1.f / (1.f + expf(-bsum));
    float a = asum + a_raw[g];
    // softplus(a * dt) where dt is the bias term
    float z = a * dt_bias[g];
    float sp = (z > 20.f) ? z : log1pf(expf(z));
    gb[g] = beta;
    gg[g] = expf(-sp);
}

// ------------------------------------------------------------------------- gdn_conv
// Causal depthwise conv over CONV_CH channels, kernel 4, then SiLU.
__global__ void gdn_conv(const float* __restrict__ qkv,   // [pos][CONV_CH] history
                         float* __restrict__ state,        // [3][CONV_CH] rolling history
                         const float* __restrict__ k,      // [4][CONV_CH]
                         float* __restrict__ out,          // [CONV_CH]
                         int CH, int pos)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CH) return;
    float acc = 0.f;
    for (int t = 0; t < 4; ++t) {
        // history is the last three tokens plus the current one
        float v = (t < 3) ? state[t * CH + c] : qkv[c];
        acc += v * k[t * CH + c];
    }
    // roll history: [2] <- [1] <- [0] <- current
    state[2 * CH + c] = state[1 * CH + c];
    state[1 * CH + c] = state[0 * CH + c];
    state[0 * CH + c] = qkv[c];
    float x = acc;
    out[c] = x / (1.f + expf(-x));    // SiLU
}

// ------------------------------------------------------------------------- l2w
// L2-normalise each HK-row of q and k in place.
__global__ void l2w(float* __restrict__ x, int HK, int rows, float eps)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float* p = x + (size_t)r * HK;
    float acc = 0.f;
    for (int i = 0; i < HK; ++i) acc += p[i] * p[i];
    float s = 1.f / sqrtf(acc + eps);
    for (int i = 0; i < HK; ++i) p[i] *= s;
}

// ------------------------------------------------------------------------- gdn_delta
// The gated delta rule: one dispatch per value head, HK columns.
//   M  = decay * M + beta * (v - M^T k) k^T        (state, HK x HK)
//   out = M^T q
// Matches the reference k_sgdn_delta. q, k, v are HK-wide rows of the l2-normalised
// conv output; gb is beta, gg the decay.
__global__ void gdn_delta(float* __restrict__ M,      // [HK][HK] state
                          const float* __restrict__ qk, // q then k, each NK*HK
                          const float* __restrict__ v,  // NV*HK
                          const float* __restrict__ gb,
                          const float* __restrict__ gg,
                          float* __restrict__ out,      // NV*HK
                          int HK, int NV, int NK, float scale)
{
    int vh = blockIdx.x;                    // value head
    int i = threadIdx.x;                    // column within HK
    if (i >= HK) return;
    // grouped: value head vh uses key head vh % NK
    int kh = vh % NK;
    const float* k = qk + (size_t)(NK * HK) + (size_t)kh * HK;
    const float* q = qk + (size_t)kh * HK;
    float* m = M + (size_t)vh * HK * HK;
    float beta = gb[vh];
    float decay = gg[vh];

    // out[i] = sum_j M[j][i] * q[j]   (M stored [row=j][col=i])
    // and the update needs M^T k, i.e. sum_j M[i][j] * k[j] per row i.
    // One thread per column i does the column of the update; the row sums need a
    // reduction, so do it in two passes with a shared scratch of HK floats.
    extern __shared__ float sh[];
    float* mk = sh;                          // M^T k, length HK
    // M^T k: component j = sum_i M[i][j] * k[i]
    float acc = 0.f;
    for (int r = 0; r < HK; ++r) acc += m[r * HK + i] * k[r];
    mk[i] = acc;
    __syncthreads();

    // v - M^T k, scaled by beta
    float vv = v[(size_t)vh * HK + i];
    float err = (vv - mk[i]) * beta;
    // M[j][i] = decay * M[j][i] + err * k[j]
    for (int r = 0; r < HK; ++r) m[r * HK + i] = decay * m[r * HK + i] + err * k[r];

    // out[i] = sum_j M[j][i] * q[j]
    float o = 0.f;
    for (int r = 0; r < HK; ++r) o += m[r * HK + i] * q[r];
    out[(size_t)vh * HK + i] = o * scale;
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
    int nsub = blockDim.x / warpSize;
    int blk = blockIdx.x * nsub + (threadIdx.x / warpSize);
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
