/* HIP (ROCm) or CUDA: KHIP is set only when hipRTC compiles this file. CUDA builds keep their exact
   instructions; the HIP alternatives sit beside them. */
#ifndef KHIP   /* KINDI-PORT-BEGIN */
#if defined(__HIPCC_RTC__) || defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__)
#define KHIP 1
#define KSHFLX(v, o) __shfl_xor((v), (o))
#define KSHFL(v, l) __shfl((v), (l))
#else
#define KHIP 0
#define KSHFLX(v, o) __shfl_xor_sync(0xffffffffu, (v), (o))
#define KSHFL(v, l) __shfl_sync(0xffffffffu, (v), (l))
#endif
#endif         /* KINDI-PORT-END */


/* ---------------------------------------------------------------- prompt chunks: weight-tile unpack + fp32 GEMM */
/* Y[t][r] = sum_k X[t][k] W[r][k] for a chunk of T prompt tokens (X: T rows of K floats, ldx apart; Y: T rows, ldy
   apart). A block computes 128 tokens x 128 rows (256 threads, 8 x 8 each) and walks the columns in steps of at most
   MM_BK. The GEMM itself (mm_core) knows nothing of the weight layout: a layout supplies an unpack stage that fills
   the step's weight tile Bs[k][r] with fp32 values (trit x scale). sm_30 and later, 44.5 KB of shared memory, no
   dp4a. One stage now: the 5-trit lane layout of k_g5n4 (16-row tiles of 28 bytes, byte-position major). */

#define MM_BT 128
#define MM_BR 128
#define MM_BK 40
#define MM_LD 132

struct MMShared {
  float A[MM_BK * MM_LD];          /* A[k][t]: the step's activations, token-minor */
  float B[MM_BK * MM_LD];          /* B[k][r]: the step's weights (trit x scale), row-minor */
  uint4 raw[8 * 28];               /* g5n: the group's raw bytes for the block's 8 row tiles */
  float d[MM_BR];                  /* g5n: the group's row scales */
};

__device__ __forceinline__ void mm_load_a(MMShared& S, const float* __restrict__ X, int ldx, int T, int t0, int k0,
                                          int nk, int t){
  const int nq = nk >> 2;
  for(int i = t; i < MM_BT * nq; i += 256){
    const int tok = i / nq, k4 = i - tok * nq;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if(t0 + tok < T) v = KLDG((const float4*)(X + (long long)(t0 + tok) * ldx + k0) + k4);
    float* a = S.A + (4 * k4) * MM_LD + tok;
    a[0] = v.x; a[MM_LD] = v.y; a[2 * MM_LD] = v.z; a[3 * MM_LD] = v.w;
  }
}

__device__ __forceinline__ void mm_core(MMShared& S, int nk, int tx, int ty, float acc[8][8]){
#pragma unroll 4
  for(int k = 0; k < nk; ++k){
    const float4 a0 = *(const float4*)(S.A + k * MM_LD + tx * 4), a1 = *(const float4*)(S.A + k * MM_LD + 64 + tx * 4);
    const float4 b0 = *(const float4*)(S.B + k * MM_LD + ty * 4), b1 = *(const float4*)(S.B + k * MM_LD + 64 + ty * 4);
    const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
    for(int i = 0; i < 8; ++i)
#pragma unroll
      for(int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
  }
}

__device__ __forceinline__ void mm_store(float* __restrict__ Y, int ldy, int T, int t0, int r0, int tx, int ty,
                                         float acc[8][8], int accumulate){
#pragma unroll
  for(int i = 0; i < 8; ++i){
    const int tok = t0 + (i < 4 ? tx * 4 + i : 64 + tx * 4 + i - 4);
    if(tok >= T) continue;
#pragma unroll
    for(int h = 0; h < 2; ++h){
      float4* yp = (float4*)(Y + (long long)tok * ldy + r0 + h * 64 + ty * 4);
      float4 v = make_float4(acc[i][4 * h], acc[i][4 * h + 1], acc[i][4 * h + 2], acc[i][4 * h + 3]);
      if(accumulate){ const float4 o = *yp; v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w; }
      *yp = v;
    }
  }
}

/* --- unpack stage: 5-trit lane layout (k_g5n4) */
__device__ __forceinline__ void g5n_group_raw(MMShared& S, const uint4* __restrict__ W, int NT16, int m0, int g, int t){
  if(t < 8 * 28){
    const int tile = t / 28, p = t - tile * 28;
    S.raw[t] = KLDG(W + ((long long)g * NT16 + m0 + tile) * 28 + p);
  }
}

__device__ __forceinline__ void g5n_group_scales(MMShared& S, int t){
  if(t < MM_BR){
    const unsigned char* rb = (const unsigned char*)S.raw;
    const int tile = t >> 4, rr = t & 15;
    const unsigned int lo = rb[(tile * 28 + 26) * 16 + rr], hi = rb[(tile * 28 + 27) * 16 + rr];
    S.d[t] = h2f((unsigned short)(lo | (hi << 8)));
  }
}

__device__ __forceinline__ void g5n_unpack(MMShared& S, int st, int t){
  /* step st of a group: st 0..2 = bytes 8 st .. 8 st + 7 (columns 40 st .. 40 st + 39, 5 per byte); st 3 = bytes 24,
     25 (columns 120 .. 127, 4 per byte) */
  const unsigned char* rb = (const unsigned char*)S.raw;
  const int nb = st < 3 ? 8 : 2, p0 = st * 8, nd = st < 3 ? 5 : 4;
  for(int i = t; i < MM_BR * nb; i += 256){
    const int r = i & (MM_BR - 1), pb = i >> 7;
    unsigned int c = rb[((r >> 4) * 28 + p0 + pb) * 16 + (r & 15)];
    const float dr = S.d[r];
    float* b = S.B + (pb * nd) * MM_LD + r;
#pragma unroll
    for(int q = 0; q < 5; ++q){
      if(q < nd){
        const unsigned int c3 = (c * 171u) >> 9;          /* c / 3 for c < 256 */
        b[q * MM_LD] = (float)((int)(c - 3u * c3) - 1) * dr;
        c = c3;
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) k_mm_g5n(const uint4* __restrict__ W, int R, int NG,
    const float* __restrict__ X, int ldx, float* __restrict__ Y, int ldy, int T, int accumulate){
  /* grid (R / 128, ceil(T / 128), splits): block z walks the groups [z NG / splits, ...); with splits > 1 the caller
     passes accumulate = 0 for none (each split writes its own Y + z * T * ldy part) */
  __shared__ MMShared S;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  const int r0 = blockIdx.x * MM_BR, t0 = blockIdx.y * MM_BT, NT16 = R >> 4, m0 = r0 >> 4;
  const int g_lo = (int)((long long)blockIdx.z * NG / gridDim.z), g_hi = (int)((long long)(blockIdx.z + 1) * NG / gridDim.z);
  float acc[8][8];
#pragma unroll
  for(int i = 0; i < 8; ++i)
#pragma unroll
    for(int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  for(int g = g_lo; g < g_hi; ++g){
    __syncthreads();
    g5n_group_raw(S, W, NT16, m0, g, t);
    __syncthreads();
    g5n_group_scales(S, t);
    __syncthreads();
    for(int st = 0; st < 4; ++st){
      const int nk = st < 3 ? 40 : 8;
      if(st) __syncthreads();
      g5n_unpack(S, st, t);
      mm_load_a(S, X, ldx, T, t0, g * 128 + st * 40, nk, t);
      __syncthreads();
      mm_core(S, nk, tx, ty, acc);
    }
  }
  mm_store(Y + (long long)blockIdx.z * T * ldy, ldy, T, t0, r0, tx, ty, acc, accumulate);
}

/* register prefetch: the next step's activations (and at a group's last step, the next group's raw weight bytes)
   are loaded into registers before the current step's products, and stored to shared memory after them */
__device__ __forceinline__ void mm_fetch_a(float4 (&ra)[5], const float* __restrict__ X, int ldx, int T, int t0,
                                           int k0, int nk, int t){
  const int nq = nk >> 2;
#pragma unroll
  for(int j = 0; j < 5; ++j){
    const int i = t + 256 * j;
    ra[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    if(i < MM_BT * nq){
      const int tok = i / nq, k4 = i - tok * nq;
      if(t0 + tok < T) ra[j] = KLDG((const float4*)(X + (long long)(t0 + tok) * ldx + k0) + k4);
    }
  }
}

__device__ __forceinline__ void mm_put_a(MMShared& S, const float4 (&ra)[5], int nk, int t){
  const int nq = nk >> 2;
#pragma unroll
  for(int j = 0; j < 5; ++j){
    const int i = t + 256 * j;
    if(i < MM_BT * nq){
      const int tok = i / nq, k4 = i - tok * nq;
      float* a = S.A + (4 * k4) * MM_LD + tok;
      a[0] = ra[j].x; a[MM_LD] = ra[j].y; a[2 * MM_LD] = ra[j].z; a[3 * MM_LD] = ra[j].w;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 2) k_mm_g5p(const uint4* __restrict__ W, int R, int NG,
    const float* __restrict__ X, int ldx, float* __restrict__ Y, int ldy, int T, int accumulate){
  /* k_mm_g5n with the global loads of step s + 1 issued before the products of step s */
  __shared__ MMShared S;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  const int r0 = blockIdx.x * MM_BR, t0 = blockIdx.y * MM_BT, NT16 = R >> 4, m0 = r0 >> 4;
  const int g_lo = (int)((long long)blockIdx.z * NG / gridDim.z), g_hi = (int)((long long)(blockIdx.z + 1) * NG / gridDim.z);
  float acc[8][8];
#pragma unroll
  for(int i = 0; i < 8; ++i)
#pragma unroll
    for(int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  float4 ra[5];
  uint4 rw = make_uint4(0u, 0u, 0u, 0u);
  if(g_lo < g_hi){
    g5n_group_raw(S, W, NT16, m0, g_lo, t);
    mm_fetch_a(ra, X, ldx, T, t0, g_lo * 128, 40, t);
    __syncthreads();
    g5n_group_scales(S, t);
    __syncthreads();
    g5n_unpack(S, 0, t);
    mm_put_a(S, ra, 40, t);
    __syncthreads();
  }
  for(int g = g_lo; g < g_hi; ++g){
    for(int st = 0; st < 4; ++st){
      const int nk = st < 3 ? 40 : 8;
      const bool more = st < 3 || g + 1 < g_hi;
      const int gn = st < 3 ? g : g + 1, sn = st < 3 ? st + 1 : 0, nkn = sn < 3 ? 40 : 8;
      if(more){
        mm_fetch_a(ra, X, ldx, T, t0, gn * 128 + sn * 40, nkn, t);
        if(st == 3 && t < 8 * 28){
          const int tile = t / 28, p = t - tile * 28;
          rw = KLDG(W + ((long long)gn * NT16 + m0 + tile) * 28 + p);
        }
      }
      mm_core(S, nk, tx, ty, acc);
      if(!more) break;
      __syncthreads();
      if(st == 3){
        if(t < 8 * 28) S.raw[t] = rw;
        __syncthreads();
        g5n_group_scales(S, t);
        __syncthreads();
      }
      g5n_unpack(S, sn, t);
      mm_put_a(S, ra, nkn, t);
      __syncthreads();
    }
  }
  mm_store(Y + (long long)blockIdx.z * T * ldy, ldy, T, t0, r0, tx, ty, acc, accumulate);
}

/* ---------------------------------------------------------------- prompt chunks: multi-token GDN */
/* The GDN steps of T consecutive prompt tokens in one launch each, in token order inside the kernel, with the same
   arithmetic per token as k_gdn_pre and k_gdn_deltan (each block's work never depends on another block's, so token
   order inside a block is the whole ordering). Token t's inputs and outputs are rows t of the chunk buffers. */

extern "C" __global__ void k_gdn_pre_T(const float* __restrict__ xn, int ldxn, const unsigned short* __restrict__ bw,
    const unsigned short* __restrict__ aw, const float* __restrict__ dtb, const float* __restrict__ ssma,
    float* __restrict__ beta, float* __restrict__ g, int ldg, int n,
    const float* __restrict__ qkv, int ldqkv, float* __restrict__ conv, const float* __restrict__ cw,
    float* __restrict__ cs, int ldcs, float* __restrict__ qk, int ldqk, int C, int NV, int NK, int HK, int T){
  /* grid (blocks, T): the gate blocks (b < NV) take token blockIdx.y; the conv blocks run at y = 0 only, over all
     T tokens in order */
  const int b = blockIdx.x, t = threadIdx.x;
  if(b < NV){
    const int tok = blockIdx.y;
    if(tok >= T) return;
    __shared__ float sb[256];
    __shared__ float sa[256];
    const float* x = xn + (long long)tok * ldxn;
    float ab = 0.f, aa = 0.f;
    for(int i = t; i < n; i += 256){
      const float xv = x[i];
      ab += __uint_as_float((unsigned int)bw[(long long)b * n + i] << 16) * xv;
      aa += __uint_as_float((unsigned int)aw[(long long)b * n + i] << 16) * xv;
    }
    sb[t] = ab; sa[t] = aa;
    __syncthreads();
    for(int st = 128; st > 0; st >>= 1){
      if(t < st){ sb[t] += sb[t + st]; sa[t] += sa[t + st]; }
      __syncthreads();
    }
    if(t == 0){
      beta[(long long)tok * ldg + b] = 1.0f / (1.0f + expf(-sb[0]));
      const float a = sa[0] + dtb[b];
      const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
      g[(long long)tok * ldg + b] = sp * ssma[b];
    }
    return;
  }
  if(blockIdx.y) return;
  const bool qkh = b < NV + 2 * NK;
  int c;
  bool act;
  if(qkh){ act = t < HK; c = (b - NV) * HK + t; }
  else { c = 2 * NK * HK + (b - NV - 2 * NK) * 256 + t; act = c < C; if(!act) return; }
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, w0 = 0.f, w1 = 0.f, w2 = 0.f, w3 = 0.f;
  if(act){
    c0 = conv[c]; c1 = conv[C + c]; c2 = conv[2 * C + c];
    w0 = cw[4 * c]; w1 = cw[4 * c + 1]; w2 = cw[4 * c + 2]; w3 = cw[4 * c + 3];
  }
  __shared__ float row[128];
  __shared__ float scl;
  for(int tok = 0; tok < T; ++tok){
    float y = 0.f;
    if(act){
      const float nw = qkv[(long long)tok * ldqkv + c];
      const float o = c0 * w0 + c1 * w1 + c2 * w2 + nw * w3;
      c0 = c1; c1 = c2; c2 = nw;
      y = o / (1.0f + expf(-o));
      cs[(long long)tok * ldcs + c] = y;
    }
    if(!qkh) continue;
    __syncthreads();
    if(act) row[t] = y;
    __syncthreads();
    if(t < 32){
      double acc = 0.0;
      for(int e = t; e < HK; e += 32){ const double v = (double)row[e]; acc += v * v; }
      acc = wsum_d(acc);
      if(t == 0) scl = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
    }
    __syncthreads();
    if(act) qk[(long long)tok * ldqk + (b - NV) * HK + t] = row[t] * scl;
  }
  if(act){ conv[c] = c0; conv[C + c] = c1; conv[2 * C + c] = c2; }
}

extern "C" __global__ void __launch_bounds__(128) k_gdn_deltan_T(float* __restrict__ M, const float* __restrict__ q,
    const float* __restrict__ k, int ldqk, const float* __restrict__ v, int ldv, const float* __restrict__ g,
    const float* __restrict__ beta, int ldg, const float* __restrict__ nw, float* __restrict__ out, int ldo,
    float scale, int n_k_heads, int T){
  /* k_gdn_deltan for T tokens: thread j keeps column j of the head's 128 x 128 state in registers for the chunk */
  const int h = blockIdx.x, j = threadIdx.x;
  __shared__ float ks[128];
  __shared__ float qs[128];
  __shared__ float os[128];
  __shared__ float scl;
  const int kh = h % n_k_heads;
  float* Mh = M + (long long)h * 128 * 128;
  float m[128];
#pragma unroll
  for(int i = 0; i < 128; ++i) m[i] = Mh[i * 128 + j];
  for(int tok = 0; tok < T; ++tok){
    __syncthreads();
    ks[j] = k[(long long)tok * ldqk + kh * 128 + j];
    qs[j] = q[(long long)tok * ldqk + kh * 128 + j];
    __syncthreads();
    const float e = expf(g[(long long)tok * ldg + h]);
    float sk = 0.f;
#pragma unroll
    for(int i = 0; i < 128; ++i) sk += (m[i] * e) * ks[i];
    const float dd = (v[(long long)tok * ldv + h * 128 + j] - sk) * beta[(long long)tok * ldg + h];
    float o = 0.f;
#pragma unroll
    for(int i = 0; i < 128; ++i){
      const float mm = m[i] * e + ks[i] * dd;
      m[i] = mm;
      o += mm * qs[i];
    }
    os[j] = o * scale;
    __syncthreads();
    if(j < 32){
      double acc = 0.0;
      for(int x = j; x < 128; x += 32){ const double vv = (double)os[x]; acc += vv * vv; }
      acc = wsum_d(acc);
      if(j == 0) scl = 1.0f / sqrtf((float)(acc / 128) + 1e-6f);
    }
    __syncthreads();
    out[(long long)tok * ldo + h * 128 + j] = (os[j] * scl) * nw[j];
  }
#pragma unroll
  for(int i = 0; i < 128; ++i) Mh[i * 128 + j] = m[i];
}

extern "C" __global__ void k_sumsplits(const float* __restrict__ p, float* __restrict__ y, int ldy, int R, int T, int sp){
  /* y[t][r] = sum of the sp split parts (each T x R, in split order) */
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)T * R) return;
  const int tok = (int)(i / R), r = (int)(i - (long long)tok * R);
  float a = p[i];
  for(int z = 1; z < sp; ++z) a += p[(long long)z * T * R + i];
  y[(long long)tok * ldy + r] = a;
}

/* ---------------------------------------------------------------- prompt chunks: attention for TS tokens at once */
/* Tokens t0 .. t0+TS-1 of the chunk (positions pos + t0 + t, each seeing the n_t = pos + t0 + t + 1 positions up to
   its own) against the q8_0 cache, as tiled products that read each K/V block once for all TS tokens' G heads.
   The arithmetic is that of the decode kernels, so the result is bit-identical to one token at a time:
   scores (k_attn_sq8t): per 32-value block an exact integer dot product (as floats), times dq * dk, the 8 blocks
     added in order, times 1/16; softmax (k_softmax_rows) unchanged per row;
   values (k_attn_vq8w + k_attn_combn): per 256-position chunk, 4 sums over positions i = 0, 1, 2, 3 mod 4 in
     position order, added (0 + 1) + 2 + 3; the chunks added in order and gated.
   S: scores, row (t, head) at S + (t * NH + head) * ld. AP: chunk parts, (c, t, head) at ((c * TS + t) * NH + head) * 256. */

__device__ __forceinline__ float sbf(unsigned int b){ return (float)(signed char)(b & 255u); }

extern "C" __global__ void __launch_bounds__(256) k_att_qk_T(const uint2* __restrict__ Kq,
    const unsigned short* __restrict__ Kd, const signed char* __restrict__ qq, const float* __restrict__ qd,
    float* __restrict__ S, int ld, int pos, int t0, int TS, int n_kv, int G, int NH){
  /* grid (ceil(nmax / 128), ceil(TS G / 64), n_kv), block 256: 64 query rows (t, j) x 128 positions, 4 rows x 8
     positions (tx * 4 .. + 3 and 64 + tx * 4 .. + 3) per thread */
  __shared__ float Qs[32][68];
  __shared__ float Ks[32][132];
  __shared__ float dqs[64];
  __shared__ float dks[128];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, kv = blockIdx.z;
  const int p0 = blockIdx.x * 128, r0 = blockIdx.y * 64, nrow = TS * G, nmax = pos + t0 + TS;
  const int QL = NH * 256;
  float acc[4][8];
#pragma unroll
  for(int a = 0; a < 4; ++a)
#pragma unroll
    for(int c = 0; c < 8; ++c) acc[a][c] = 0.f;
  const int lr = tid >> 2, lq = tid & 3;
  const int qrow = r0 + lr, qt = qrow / G, qj = qrow - qt * G;
  const bool qok = qrow < nrow;
  const signed char* qsrc = qq + (long long)(t0 + qt) * QL + (kv * G + qj) * 256;
  for(int b = 0; b < 8; ++b){
    __syncthreads();
    uint2 qw = make_uint2(0u, 0u);
    if(qok) qw = *(const uint2*)(qsrc + b * 32 + lq * 8);
#pragma unroll
    for(int e = 0; e < 4; ++e){ Qs[lq * 8 + e][lr] = sbf(qw.x >> (8 * e)); Qs[lq * 8 + 4 + e][lr] = sbf(qw.y >> (8 * e)); }
#pragma unroll
    for(int h = 0; h < 2; ++h){
      const int kp = p0 + h * 64 + lr;
      uint2 kw = make_uint2(0u, 0u);
      if(kp < nmax) kw = KLDG(Kq + (((long long)kp * n_kv + kv) * 256 + b * 32 + lq * 8) / 8);
#pragma unroll
      for(int e = 0; e < 4; ++e){
        Ks[lq * 8 + e][h * 64 + lr] = sbf(kw.x >> (8 * e)); Ks[lq * 8 + 4 + e][h * 64 + lr] = sbf(kw.y >> (8 * e));
      }
    }
    if(tid < 64){
      const int r = r0 + tid, t = r / G, j = r - t * G;
      dqs[tid] = r < nrow ? qd[(long long)(t0 + t) * (QL / 32) + (kv * G + j) * 8 + b] : 0.f;
    } else if(tid < 192){
      const int p = p0 + tid - 64;
      dks[tid - 64] = p < nmax ? h2f(KLDG(Kd + ((long long)p * n_kv + kv) * 8 + b)) : 0.f;
    }
    __syncthreads();
    float part[4][8];
#pragma unroll
    for(int a = 0; a < 4; ++a)
#pragma unroll
      for(int c = 0; c < 8; ++c) part[a][c] = 0.f;
#pragma unroll 8
    for(int i = 0; i < 32; ++i){
      const float4 qv = *(const float4*)&Qs[i][ty * 4];
      const float4 k0 = *(const float4*)&Ks[i][tx * 4], k1 = *(const float4*)&Ks[i][64 + tx * 4];
      const float qa[4] = {qv.x, qv.y, qv.z, qv.w}, ka[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
      for(int a = 0; a < 4; ++a)
#pragma unroll
        for(int c = 0; c < 8; ++c) part[a][c] += ka[c] * qa[a];
    }
#pragma unroll
    for(int a = 0; a < 4; ++a)
#pragma unroll
      for(int c = 0; c < 8; ++c)
        acc[a][c] += part[a][c] * (dqs[ty * 4 + a] * dks[(c < 4 ? tx * 4 + c : 64 + tx * 4 + c - 4)]);
  }
#pragma unroll
  for(int a = 0; a < 4; ++a){
    const int r = r0 + ty * 4 + a;
    if(r >= nrow) continue;
    const int t = r / G, j = r - t * G, n = pos + t0 + t + 1;
    float* srow = S + ((long long)t * NH + kv * G + j) * ld;
#pragma unroll
    for(int c = 0; c < 8; ++c){
      const int p = p0 + (c < 4 ? tx * 4 + c : 64 + tx * 4 + c - 4);
      if(p < n) srow[p] = acc[a][c] * 0.0625f;
    }
  }
}

extern "C" __global__ void k_att_softmax_T(float* __restrict__ S, int ld, int pos, int t0, int NH){
  /* k_softmax_rows for row (token blockIdx.y, head blockIdx.x) of n = pos + t0 + t + 1 positions */
  const int hh = blockIdx.x, t = threadIdx.x, n = pos + t0 + blockIdx.y + 1;
  float* sr = S + ((long long)blockIdx.y * NH + hh) * ld;
  __shared__ float red[256];
  __shared__ float pmax;
  float mx = -1e30f;
  for(int i = t; i < n; i += 256) mx = fmaxf(mx, sr[i]);
  red[t] = mx;
  __syncthreads();
  for(int st = 128; st > 0; st >>= 1){
    if(t < st) red[t] = fmaxf(red[t], red[t + st]);
    __syncthreads();
  }
  if(t == 0) pmax = red[0];
  __syncthreads();
  const float m = pmax;
  float acc0 = 0.0f;
  for(int i = t; i < n; i += 256){ const float e = expf(sr[i] - m); sr[i] = e; acc0 += e; }   /* kept: the same expf */
  __syncthreads();
  red[t] = acc0;
  __syncthreads();
  for(int st = 128; st > 0; st >>= 1){
    if(t < st) red[t] += red[t + st];
    __syncthreads();
  }
  const float tot = red[0];
  __syncthreads();
  for(int i = t; i < n; i += 256) sr[i] = sr[i] / tot;
}

extern "C" __global__ void __launch_bounds__(256) k_att_pv_T(const float* __restrict__ S, int ld,
    const uint2* __restrict__ Vq, const unsigned short* __restrict__ Vd, float* __restrict__ AP, int pos, int t0,
    int TS, int n_kv, int G, int NH){
  /* grid (chunks of 256 positions, ceil(TS G / 64), n_kv * 4 feature quarters), block 256: 64 rows x 64 features,
     4 rows x 4 features per thread, each with the 4 position-class sums of k_attn_vq8w */
  __shared__ float Ps[64][33];
  __shared__ float Vs[32][68];
  const int tid = threadIdx.x, fq = tid & 15, rq = tid >> 4;
  const int c = blockIdx.x, r0 = blockIdx.y * 64, kv = blockIdx.z >> 2, f0 = (blockIdx.z & 3) * 64;
  const int nrow = TS * G, i0 = c * 256, nmax = pos + t0 + TS;
  /* the loader: lane li = position in the step, rows (tid >> 5) + 8 m */
  const int li = tid & 31;
  long long rb[8];
  int nl[8];
#pragma unroll
  for(int m = 0; m < 8; ++m){
    const int rr = r0 + (tid >> 5) + 8 * m;
    nl[m] = 0; rb[m] = 0;
    if(rr < nrow){
      const int t = rr / G, j = rr - t * G;
      nl[m] = pos + t0 + t + 1;
      rb[m] = ((long long)t * NH + kv * G + j) * ld;
    }
  }
  float acc[4][4][4];
#pragma unroll
  for(int g = 0; g < 4; ++g)
#pragma unroll
    for(int a = 0; a < 4; ++a)
#pragma unroll
      for(int e = 0; e < 4; ++e) acc[g][a][e] = 0.f;
  for(int s0 = 0; s0 < 256; s0 += 32){
    if(i0 + s0 >= nmax) break;
    __syncthreads();
    const int p = i0 + s0 + li;
#pragma unroll
    for(int m = 0; m < 8; ++m) Ps[(tid >> 5) + 8 * m][li] = p < nl[m] ? S[rb[m] + p] : 0.f;
    {
      const int i = tid >> 3, f8 = tid & 7, p = i0 + s0 + i;
      float v[8];
#pragma unroll
      for(int e = 0; e < 8; ++e) v[e] = 0.f;
      if(p < nmax){
        const uint2 w = KLDG(Vq + (((long long)p * n_kv + kv) * 256 + f0 + f8 * 8) / 8);
        const float d = h2f(KLDG(Vd + ((long long)p * n_kv + kv) * 8 + (f0 + f8 * 8) / 32));
#pragma unroll
        for(int e = 0; e < 4; ++e){ v[e] = sbf(w.x >> (8 * e)) * d; v[4 + e] = sbf(w.y >> (8 * e)) * d; }
      }
#pragma unroll
      for(int e = 0; e < 8; ++e) Vs[i][f8 * 8 + e] = v[e];
    }
    __syncthreads();
#pragma unroll
    for(int i = 0; i < 32; ++i){
      const float4 v4 = *(const float4*)&Vs[i][fq * 4];
      const float va[4] = {v4.x, v4.y, v4.z, v4.w};
      const float pa[4] = {Ps[rq * 4][i], Ps[rq * 4 + 1][i], Ps[rq * 4 + 2][i], Ps[rq * 4 + 3][i]};
#pragma unroll
      for(int a = 0; a < 4; ++a)
#pragma unroll
        for(int e = 0; e < 4; ++e) acc[i & 3][a][e] += pa[a] * va[e];
    }
  }
#pragma unroll
  for(int a = 0; a < 4; ++a){
    const int rr = r0 + rq * 4 + a;
    if(rr >= nrow) continue;
    const int t = rr / G, j = rr - t * G;
    float* ap = AP + (((long long)c * TS + t) * NH + kv * G + j) * 256 + f0 + fq * 4;
#pragma unroll
    for(int e = 0; e < 4; ++e){
      float x = acc[0][a][e];
      x += acc[1][a][e]; x += acc[2][a][e]; x += acc[3][a][e];
      ap[e] = x;
    }
  }
}

extern "C" __global__ void k_att_comb_T(const float* __restrict__ AP, const float* __restrict__ gatev, int ldg,
                                        float* __restrict__ att, int lda, int pos, int t0, int TS, int NH){
  /* k_attn_combn for token t0 + blockIdx.y: its chunks added in order, times the sigmoid gate */
  const int hh = blockIdx.x, f = threadIdx.x, t = blockIdx.y, n = pos + t0 + t + 1, nch = (n + 255) / 256;
  float acc = AP[((long long)t * NH + hh) * 256 + f];
  for(int c = 1; c < nch; ++c) acc += AP[(((long long)c * TS + t) * NH + hh) * 256 + f];
  const float g = gatev[(long long)(t0 + t) * ldg + hh * 512 + 256 + f];
  att[(long long)(t0 + t) * lda + hh * 256 + f] = acc * (1.0f / (1.0f + expf(-g)));
}

/* ---------------------------------------------------------------- prompt chunks: one-pass attention */
#define KNEGINF __int_as_float(0xff800000)
/* All T tokens of the chunk in one launch, no score matrix: a block takes 64 query rows (t, j) of one K/V head and
   walks the positions in tiles of 64 with a running max and sum (online softmax). The scores are those of
   k_attn_sq8t (per 32-value block an exact integer dot product, times dq * dk, the 8 blocks added in order, times
   1/16); the softmax and the value sums are the same quantities in a different order of additions. The output is
   gated as in k_attn_combn. grid (ceil(T G / 64), n_kv), block 256: 4 rows x 4 positions (scores) and 4 rows x 16
   features (values) per thread. */

extern "C" __global__ void __launch_bounds__(256, 2) k_att_flash_T(const uint2* __restrict__ Kq,
    const unsigned short* __restrict__ Kd, const uint4* __restrict__ Vq, const unsigned short* __restrict__ Vd,
    const signed char* __restrict__ qq, const float* __restrict__ qd, const float* __restrict__ gatev, int ldg,
    float* __restrict__ att, int lda, int pos, int T, int n_kv, int G, int NH, float* __restrict__ PO){
  /* gridDim.z > 1: block z takes position tiles [z ntiles / Z, (z + 1) ntiles / Z) and leaves its unnormalised values,
     max and sum in PO (row (kv, r) of split z at ((z * n_kv + kv) * T G + r) * 260) for k_att_merge_T */
  __shared__ float QKV[2 * 32 * 68];          /* scores: Q[32][68] then K[32][68]; values: V[64][68] */
  __shared__ float Ps[64][68];
  __shared__ float dqs[64];
  __shared__ float dks[64];
  float (*Qs)[68] = (float (*)[68])QKV;
  float (*Ks)[68] = (float (*)[68])(QKV + 32 * 68);
  float (*Vs)[68] = (float (*)[68])QKV;
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, kv = blockIdx.y;
  const int r0 = blockIdx.x * 64, nrow = T * G, QL = NH * 256;
  const int rlast = min(r0 + 63, nrow - 1), nblk = pos + rlast / G + 1;
  /* this thread's 4 rows */
  int nr[4];
#pragma unroll
  for(int a = 0; a < 4; ++a){
    const int r = r0 + ty * 4 + a;
    nr[a] = r < nrow ? pos + r / G + 1 : 0;
  }
  /* loader roles */
  const int lr = tid >> 2, lq = tid & 3;
  const int qrow = r0 + lr, qt = qrow / G, qj = qrow - qt * G;
  const bool qok = qrow < nrow;
  const signed char* qsrc = qq + (long long)(qt) * QL + (kv * G + qj) * 256;
  float m[4], l[4], o[4][16];
#pragma unroll
  for(int a = 0; a < 4; ++a){
    m[a] = KNEGINF; l[a] = 0.f;
#pragma unroll
    for(int f = 0; f < 16; ++f) o[a][f] = 0.f;
  }
  const int ntl = (nblk + 63) / 64, z = blockIdx.z, Z = gridDim.z;
  const int tlo = (int)((long long)z * ntl / Z), thi = (int)((long long)(z + 1) * ntl / Z);
  for(int p0 = tlo * 64; p0 < thi * 64; p0 += 64){
    float acc[4][4];
#pragma unroll
    for(int a = 0; a < 4; ++a)
#pragma unroll
      for(int c = 0; c < 4; ++c) acc[a][c] = 0.f;
    const int kp = p0 + lr;
    for(int b = 0; b < 8; ++b){
      __syncthreads();
      uint2 qw = make_uint2(0u, 0u), kw = make_uint2(0u, 0u);
      if(qok) qw = *(const uint2*)(qsrc + b * 32 + lq * 8);
      if(kp < nblk) kw = KLDG(Kq + (((long long)kp * n_kv + kv) * 256 + b * 32 + lq * 8) / 8);
#pragma unroll
      for(int e = 0; e < 4; ++e){
        Qs[lq * 8 + e][lr] = sbf(qw.x >> (8 * e)); Qs[lq * 8 + 4 + e][lr] = sbf(qw.y >> (8 * e));
        Ks[lq * 8 + e][lr] = sbf(kw.x >> (8 * e)); Ks[lq * 8 + 4 + e][lr] = sbf(kw.y >> (8 * e));
      }
      if(tid < 64){
        const int r = r0 + tid, t = r / G, j = r - t * G;
        dqs[tid] = r < nrow ? qd[(long long)t * (QL / 32) + (kv * G + j) * 8 + b] : 0.f;
      } else if(tid < 128){
        const int p = p0 + tid - 64;
        dks[tid - 64] = p < nblk ? h2f(KLDG(Kd + ((long long)p * n_kv + kv) * 8 + b)) : 0.f;
      }
      __syncthreads();
      float part[4][4];
#pragma unroll
      for(int a = 0; a < 4; ++a)
#pragma unroll
        for(int c = 0; c < 4; ++c) part[a][c] = 0.f;
#pragma unroll 8
      for(int i = 0; i < 32; ++i){
        const float4 qv = *(const float4*)&Qs[i][ty * 4];
        const float4 k4 = *(const float4*)&Ks[i][tx * 4];
        const float qa[4] = {qv.x, qv.y, qv.z, qv.w}, ka[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
        for(int a = 0; a < 4; ++a)
#pragma unroll
          for(int c = 0; c < 4; ++c) part[a][c] += ka[c] * qa[a];
      }
#pragma unroll
      for(int a = 0; a < 4; ++a)
#pragma unroll
        for(int c = 0; c < 4; ++c) acc[a][c] += part[a][c] * (dqs[ty * 4 + a] * dks[tx * 4 + c]);
    }
    /* online softmax over this tile's 64 positions; the 16 threads of a row group are lanes of one warp half */
#pragma unroll
    for(int a = 0; a < 4; ++a){
      float sv[4], mt = KNEGINF;
#pragma unroll
      for(int c = 0; c < 4; ++c){
        const int p = p0 + tx * 4 + c;
        sv[c] = p < nr[a] ? acc[a][c] * 0.0625f : KNEGINF;
        mt = fmaxf(mt, sv[c]);
      }
#pragma unroll
      for(int off = 8; off > 0; off >>= 1) mt = fmaxf(mt, KSHFLX(mt, off));
      const float mn = fmaxf(m[a], mt);
      float rs = 0.f, pv[4];
#pragma unroll
      for(int c = 0; c < 4; ++c){ pv[c] = mn == KNEGINF ? 0.f : expf(sv[c] - mn); rs += pv[c]; }
#pragma unroll
      for(int off = 8; off > 0; off >>= 1) rs += KSHFLX(rs, off);
      const float sc = (m[a] == KNEGINF) ? 0.f : expf(m[a] - mn);
      m[a] = mn;
      l[a] = l[a] * sc + rs;
#pragma unroll
      for(int f = 0; f < 16; ++f) o[a][f] *= sc;
      *(float4*)&Ps[ty * 4 + a][tx * 4] = make_float4(pv[0], pv[1], pv[2], pv[3]);
    }
    /* values: 4 feature quarters of 64, the tile's 64 positions each */
#pragma unroll 1
    for(int q = 0; q < 4; ++q){
      __syncthreads();
      {
        const int key = tid >> 2, f16 = (tid & 3) * 16, p = p0 + key;
        float v[16];
#pragma unroll
        for(int e = 0; e < 16; ++e) v[e] = 0.f;
        if(p < nblk){
          const uint4 w = KLDG(Vq + (((long long)p * n_kv + kv) * 256 + q * 64 + f16) / 16);
          const float d = h2f(KLDG(Vd + ((long long)p * n_kv + kv) * 8 + (q * 64 + f16) / 32));
          const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
          for(int u = 0; u < 4; ++u)
#pragma unroll
            for(int e = 0; e < 4; ++e) v[u * 4 + e] = sbf(ww[u] >> (8 * e)) * d;
        }
#pragma unroll
        for(int e = 0; e < 16; e += 4) *(float4*)&Vs[key][f16 + e] = make_float4(v[e], v[e + 1], v[e + 2], v[e + 3]);
      }
      __syncthreads();
#pragma unroll 8
      for(int k = 0; k < 64; ++k){
        const float4 v4 = *(const float4*)&Vs[k][tx * 4];
        const float va[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for(int a = 0; a < 4; ++a){
          const float pa = Ps[ty * 4 + a][k];
#pragma unroll
          for(int e = 0; e < 4; ++e) o[a][q * 4 + e] += pa * va[e];
        }
      }
    }
  }
  if(Z > 1){
#pragma unroll
    for(int a = 0; a < 4; ++a){
      const int r = r0 + ty * 4 + a;
      if(r >= nrow) continue;
      float* po = PO + (((long long)z * n_kv + kv) * nrow + r) * 260;
#pragma unroll
      for(int q = 0; q < 4; ++q)
        *(float4*)(po + q * 64 + tx * 4) = make_float4(o[a][q * 4], o[a][q * 4 + 1], o[a][q * 4 + 2], o[a][q * 4 + 3]);
      if(tx == 0){ po[256] = m[a]; po[257] = l[a]; }
    }
    return;
  }
#pragma unroll
  for(int a = 0; a < 4; ++a){
    const int r = r0 + ty * 4 + a;
    if(r >= nrow) continue;
    const int t = r / G, j = r - t * G, hh = kv * G + j;
    const float il = 1.0f / l[a];
#pragma unroll
    for(int q = 0; q < 4; ++q)
#pragma unroll
      for(int e = 0; e < 4; ++e){
        const int f = q * 64 + tx * 4 + e;
        const float g = gatev[(long long)t * ldg + hh * 512 + 256 + f];
        att[(long long)t * lda + hh * 256 + f] = (o[a][q * 4 + e] * il) * (1.0f / (1.0f + expf(-g)));
      }
  }
}

extern "C" __global__ void k_att_merge_T(const float* __restrict__ PO, int Z, const float* __restrict__ gatev, int ldg,
                                         float* __restrict__ att, int lda, int T, int n_kv, int G){
  /* grid (T G, n_kv), block 256: the Z partial results of row (kv, r) merged, normalised and gated */
  const int r = blockIdx.x, kv = blockIdx.y, f = threadIdx.x, nrow = T * G;
  float mx = KNEGINF;
  for(int z = 0; z < Z; ++z) mx = fmaxf(mx, PO[(((long long)z * n_kv + kv) * nrow + r) * 260 + 256]);
  float o = 0.f, l = 0.f;
  for(int z = 0; z < Z; ++z){
    const float* po = PO + (((long long)z * n_kv + kv) * nrow + r) * 260;
    const float mz = po[256];
    if(mz == KNEGINF) continue;
    const float w = expf(mz - mx);
    o += po[f] * w;
    l += po[257] * w;
  }
  const int t = r / G, j = r - t * G, hh = kv * G + j;
  const float g = gatev[(long long)t * ldg + hh * 512 + 256 + f];
  att[(long long)t * lda + hh * 256 + f] = (o / l) * (1.0f / (1.0f + expf(-g)));
}

/* ---------------------------------------------------------------- prompt GEMM, int8 activations (sm_61+, dp4a) */
/* KINDI_CHUNK_Q8=1 (GTX 1080 and later): the chunk's activations are quantised to int8 per 128-column group (scale
   amax / 127, round to nearest even, as llama.cpp's q8 rows), the weights unpacked to int8 trits, and each group's
   128-term dot product is an exact integer (dp4a); the group sum is scaled by x-scale * w-scale and added in fp32.
   Not exact fp32 arithmetic: the activations carry int8 rounding (llama.cpp's own matrix products do the same with
   q8_1 rows). Below sm_61 the kernel is not used. */

extern "C" __global__ void k_q8x_T(const float* __restrict__ X, int ldx, unsigned int* __restrict__ XQ,
                                   float* __restrict__ XS, int ldt, int T){
  /* grid (K / 128, T), block 128: group blockIdx.x of token blockIdx.y -> XQ[(g * 32 + k4) * ldt + t] (4 int8 of
     columns 4 k4 ..) and XS[g * ldt + t] */
  const int g = blockIdx.x, t = blockIdx.y, e = threadIdx.x, lane = e & 31, w = e >> 5;
  if(t >= T) return;
  __shared__ float wm[4];
  const float v = X[(long long)t * ldx + g * 128 + e];
  float a = fabsf(v);
  for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, KSHFLX(a, o));
  if(lane == 0) wm[w] = a;
  __syncthreads();
  const float amax = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
  const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
  const unsigned int q = (unsigned int)(__float2int_rn(v * id) & 255);
  /* pack 4 consecutive columns: lanes 4 m .. 4 m + 3 */
  unsigned int word = q << (8 * (lane & 3));
  word |= KSHFLX(word, 1);
  word |= KSHFLX(word, 2);
  if((lane & 3) == 0) XQ[((long long)g * 32 + (e >> 2)) * ldt + t] = word;
  if(e == 0) XS[(long long)g * ldt + t] = amax / 127.0f;
}

extern "C" __global__ void __launch_bounds__(256) k_q8x_T2(const float* __restrict__ X, int ldx,
    unsigned int* __restrict__ XQ, float* __restrict__ XS, int ldt, int T){
  /* grid (K / 128, ceil(T / 64)), block 256 (8 warps x 8 tokens): bit-identical XQ / XS to k_q8x_T */
  __shared__ unsigned int Q[32][65];
  const int g = blockIdx.x, t0 = blockIdx.y * 64, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for(int i = 0; i < 8; ++i){
    const int tt = w * 8 + i, t = t0 + tt;
    if(t < T){
      const float4 v = *(const float4*)(X + (long long)t * ldx + g * 128 + lane * 4);
      float a = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
      for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o));
      const float id = a != 0.0f ? 127.0f / a : 0.0f;
      Q[lane][tt] = ((unsigned int)(__float2int_rn(v.x * id) & 255)) | ((unsigned int)(__float2int_rn(v.y * id) & 255) << 8) |
                    ((unsigned int)(__float2int_rn(v.z * id) & 255) << 16) | ((unsigned int)(__float2int_rn(v.w * id) & 255) << 24);
      if(lane == 0) XS[(long long)g * ldt + t] = a / 127.0f;
    }
  }
  __syncthreads();
  for(int i = threadIdx.x; i < 32 * 64; i += 256){
    const int k4 = i >> 6, tt = i & 63, t = t0 + tt;
    if(t < T) XQ[((long long)g * 32 + k4) * ldt + t] = Q[k4][tt];
  }
}

struct MQShared {
  unsigned int A[32][MM_LD];       /* A[k4][t]: 4 int8 activations of columns 4 k4 .. of token t */
  unsigned int B[32][MM_LD];       /* B[k4][r]: 4 int8 trits (-1, 0, 1) of row r */
  uint4 raw[8 * 28];
  float d[MM_BR];
  float xs[MM_BT];
};

__device__ __forceinline__ unsigned int tq(unsigned int dg){ return (dg + 255u) & 255u; }   /* trit digit -> int8 */

template <int NBYTE, int P0>
__device__ __forceinline__ void mq_unpack_row(MQShared& S, int r, int k40){
  /* bytes P0 .. P0 + NBYTE - 1 of row r (5 columns each below byte 24, 4 from byte 24) -> words k40 .. */
  const unsigned char* rb = (const unsigned char*)S.raw;
  unsigned int word = 0u;
  int nd = 0, k4 = k40;
#pragma unroll
  for(int p = P0; p < P0 + NBYTE; ++p){
    unsigned int c = rb[((r >> 4) * 28 + p) * 16 + (r & 15)];
#pragma unroll
    for(int q = 0; q < (p < 24 ? 5 : 4); ++q){
      const unsigned int c3 = (c * 171u) >> 9;
      word |= tq(c - 3u * c3) << (8 * (nd & 3));
      c = c3;
      if((nd & 3) == 3){ S.B[k4][r] = word; word = 0u; ++k4; }
      ++nd;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 1) k_mm_g5q(const uint4* __restrict__ W, int R, int NG,
    const unsigned int* __restrict__ XQ, const float* __restrict__ XS, int ldt, float* __restrict__ Y, int ldy, int T){
  __shared__ MQShared S;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  const int r0 = blockIdx.x * MM_BR, t0 = blockIdx.y * MM_BT, NT16 = R >> 4, m0 = r0 >> 4;
  float acc[8][8];
#pragma unroll
  for(int i = 0; i < 8; ++i)
#pragma unroll
    for(int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  for(int g = 0; g < NG; ++g){
    __syncthreads();
    if(t < 8 * 28){
      const int tile = t / 28, p = t - tile * 28;
      S.raw[t] = KLDG(W + ((long long)g * NT16 + m0 + tile) * 28 + p);
    }
#pragma unroll
    for(int j = 0; j < 4; ++j){
      const int i = t + 256 * j, k4 = i >> 5, q4 = i & 31;
      *(uint4*)&S.A[k4][q4 * 4] = KLDG((const uint4*)(XQ + ((long long)g * 32 + k4) * ldt + t0) + q4);
    }
    if(t < MM_BT) S.xs[t] = KLDG(XS + (long long)g * ldt + t0 + t);
    __syncthreads();
    if(t < MM_BR){
      const unsigned char* rb = (const unsigned char*)S.raw;
      const unsigned int lo = rb[((t >> 4) * 28 + 26) * 16 + (t & 15)], hi = rb[((t >> 4) * 28 + 27) * 16 + (t & 15)];
      S.d[t] = h2f((unsigned short)(lo | (hi << 8)));
      mq_unpack_row<12, 0>(S, t, 0);                 /* columns 0 .. 59 */
    } else {
      mq_unpack_row<14, 12>(S, t - MM_BR, 15);       /* columns 60 .. 127 */
    }
    __syncthreads();
    int is[8][8];
#pragma unroll
    for(int i = 0; i < 8; ++i)
#pragma unroll
      for(int j = 0; j < 8; ++j) is[i][j] = 0;
#pragma unroll 4
    for(int k4 = 0; k4 < 32; ++k4){
      const uint4 a0 = *(const uint4*)&S.A[k4][tx * 4], a1 = *(const uint4*)&S.A[k4][64 + tx * 4];
      const uint4 b0 = *(const uint4*)&S.B[k4][ty * 4], b1 = *(const uint4*)&S.B[k4][64 + ty * 4];
      const int a[8] = {(int)a0.x, (int)a0.y, (int)a0.z, (int)a0.w, (int)a1.x, (int)a1.y, (int)a1.z, (int)a1.w};
      const int b[8] = {(int)b0.x, (int)b0.y, (int)b0.z, (int)b0.w, (int)b1.x, (int)b1.y, (int)b1.z, (int)b1.w};
#pragma unroll
      for(int i = 0; i < 8; ++i)
#pragma unroll
        for(int j = 0; j < 8; ++j) is[i][j] = kdp4a(a[i], b[j], is[i][j]);
    }
#pragma unroll
    for(int i = 0; i < 8; ++i){
      const float xs = S.xs[i < 4 ? tx * 4 + i : 64 + tx * 4 + i - 4];
#pragma unroll
      for(int j = 0; j < 8; ++j){
        const float dj = S.d[j < 4 ? ty * 4 + j : 64 + ty * 4 + j - 4];
        const float f = __int_as_float(is[i][j] + 0x4B400000) - 12582912.0f;     /* exact: |is| < 2^22 */
        acc[i][j] += f * (xs * dj);
      }
    }
  }
  mm_store(Y, ldy, T, t0, r0, tx, ty, acc, 0);
}
