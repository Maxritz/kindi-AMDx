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


/* ---------------------------------------------------------------- prompt chunks on the single-card engine */
/* The 2B / 4B / 9B engine (engine.py): 2-bit weights (k_g4 layout), fp16 K/V, GQA, head size 256. The GEMM is mm_core
   with a 2-bit unpack stage; the GDN conv and delta steps run T tokens per launch with the decode arithmetic
   (k_gdn_conv, k_gdn_delta); attention is one pass over the fp16 cache (scores as k_attn_s4w's products, online
   softmax, gated as k_attn_comb). */

__device__ __forceinline__ void g4_unpack(MMShared& S, const unsigned char* __restrict__ Wb, int NPH, int r0, int k0,
                                          int t){
  /* the 32 columns k0 .. k0 + 31 of rows r0 .. r0 + 127 (8 bytes of 4 trits each per row) into S.B, times S.d */
  const int ph = k0 >> 9, cq0 = (k0 & 511) >> 2;
  for(int i = t; i < MM_BR * 8; i += 256){
    const int r = i & (MM_BR - 1), bq = i >> 7, row = r0 + r, rr = row & 15;
    const int cq = cq0 + bq, lane = cq >> 2;
    const long long u = (((long long)(row >> 4) * NPH + ph) * 4 + (rr >> 2)) * 32 + lane;
    unsigned int c = KLDG(Wb + u * 16 + (rr & 3) * 4 + (cq & 3));
    const float dr = S.d[r], nr = -dr;
    float* b = S.B + (bq * 4) * MM_LD + r;
#pragma unroll
    for(int q = 0; q < 4; ++q){
      const unsigned int c3 = (c * 171u) >> 9, dg = c - 3u * c3;
      b[q * MM_LD] = dg == 0u ? nr : (dg == 1u ? 0.f : dr);
      c = c3;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) k_mm_g4(const unsigned char* __restrict__ Wb,
    const unsigned short* __restrict__ Sd, int R, int C, const float* __restrict__ X, int ldx, float* __restrict__ Y,
    int ldy, int T){
  /* grid (R / 128, ceil(T / 128), splits): block z walks the 128-column blocks [z NB / splits, ...) and writes its
     part to Y + z T ldy */
  __shared__ MMShared S;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  const int r0 = blockIdx.x * MM_BR, t0 = blockIdx.y * MM_BT, NPH = C >> 9, NB = C >> 7;
  const int b_lo = (int)((long long)blockIdx.z * NB / gridDim.z), b_hi = (int)((long long)(blockIdx.z + 1) * NB / gridDim.z);
  float acc[8][8];
#pragma unroll
  for(int i = 0; i < 8; ++i)
#pragma unroll
    for(int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  for(int cb = b_lo; cb < b_hi; ++cb){
    __syncthreads();
    if(t < MM_BR){
      /* the scale of row r for this 128-column block: lane (cb & 3) * 8 + l of its phase, l from the row's pair */
      const int row = r0 + t, rr = row & 15, bb = rr & ~1;
      const int l = ((bb >> 3) & 1) | (((bb >> 2) & 1) << 1) | (((bb >> 1) & 1) << 2);
      S.d[t] = h2f(KLDG(Sd + (((long long)(row >> 4) * NPH + (cb >> 2)) * 32 + (cb & 3) * 8 + l) * 2 + (rr & 1)));
    }
    __syncthreads();
    for(int st = 0; st < 4; ++st){
      if(st) __syncthreads();
      g4_unpack(S, Wb, NPH, r0, cb * 128 + st * 32, t);
      mm_load_a(S, X, ldx, T, t0, cb * 128 + st * 32, 32, t);
      __syncthreads();
      mm_core(S, 32, tx, ty, acc);
    }
  }
  mm_store(Y + (long long)blockIdx.z * T * ldy, ldy, T, t0, r0, tx, ty, acc, 0);
}

extern "C" __global__ void k_gdn_conv_T(const float* __restrict__ qkv, int ldq, float* __restrict__ conv,
                                        const float* __restrict__ cw, float* __restrict__ cs, int ldc, int C, int T){
  /* k_gdn_conv for T tokens in order, the shift register in registers */
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= C) return;
  float c0 = conv[c], c1 = conv[C + c], c2 = conv[2 * C + c];
  const float w0 = cw[4 * c], w1 = cw[4 * c + 1], w2 = cw[4 * c + 2], w3 = cw[4 * c + 3];
  for(int tok = 0; tok < T; ++tok){
    const float nw = qkv[(long long)tok * ldq + c];
    const float o = c0 * w0 + c1 * w1 + c2 * w2 + nw * w3;
    c0 = c1; c1 = c2; c2 = nw;
    cs[(long long)tok * ldc + c] = o / (1.0f + expf(-o));
  }
  conv[c] = c0; conv[C + c] = c1; conv[2 * C + c] = c2;
}

extern "C" __global__ void __launch_bounds__(128) k_gdn_delta_T(float* __restrict__ M, const float* __restrict__ q,
    const float* __restrict__ k, int ldqk, const float* __restrict__ v, int ldv, const float* __restrict__ g,
    const float* __restrict__ beta, int ldg, float* __restrict__ out, int ldo, float scale, int n_k_heads, int T){
  /* k_gdn_delta for T tokens: thread j keeps column j of the head's state in registers for the chunk */
  const int h = blockIdx.x, j = threadIdx.x;
  __shared__ float ks[128];
  __shared__ float qs[128];
  const int kh = h % n_k_heads;
  float* Mh = M + (long long)h * 128 * 128;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350
  /* sm_30 (63 registers): the state stays in global memory, as in k_gdn_delta */
  for(int tok = 0; tok < T; ++tok){
    __syncthreads();
    ks[j] = k[(long long)tok * ldqk + kh * 128 + j];
    qs[j] = q[(long long)tok * ldqk + kh * 128 + j];
    __syncthreads();
    const float e = expf(g[(long long)tok * ldg + h]);
    float sk = 0.f;
    for(int i = 0; i < 128; ++i) sk += (Mh[i * 128 + j] * e) * ks[i];
    const float dd = (v[(long long)tok * ldv + h * 128 + j] - sk) * beta[(long long)tok * ldg + h];
    float o = 0.f;
    for(int i = 0; i < 128; ++i){
      const float mm = Mh[i * 128 + j] * e + ks[i] * dd;
      Mh[i * 128 + j] = mm;
      o += mm * qs[i];
    }
    out[(long long)tok * ldo + h * 128 + j] = o * scale;
  }
#else
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
    out[(long long)tok * ldo + h * 128 + j] = o * scale;
  }
#pragma unroll
  for(int i = 0; i < 128; ++i) Mh[i * 128 + j] = m[i];
#endif
}

extern "C" __global__ void __launch_bounds__(256) k_att_flash16_T(const unsigned short* __restrict__ Kc,
    const unsigned short* __restrict__ Vc, const float* __restrict__ qr, int ldq, const float* __restrict__ gatev,
    int ldg, float* __restrict__ att, int lda, int pos, int T, int n_kv, int G, int NH, float* __restrict__ PO){
  /* k_att_flash_T for the fp16 cache and an fp32 query (rows (t, j) of K/V head blockIdx.y, 64 per block); with
     gridDim.z > 1 the partial results go to PO for k_att_merge_T */
  __shared__ float QKV[2 * 32 * 68];
  __shared__ float Ps[64][68];
  float (*Qs)[68] = (float (*)[68])QKV;
  float (*Ks)[68] = (float (*)[68])(QKV + 32 * 68);
  float (*Vs)[68] = (float (*)[68])QKV;
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, kv = blockIdx.y;
  const int r0 = blockIdx.x * 64, nrow = T * G;
  const int rlast = min(r0 + 63, nrow - 1), nblk = pos + rlast / G + 1;
  int nr[4];
#pragma unroll
  for(int a = 0; a < 4; ++a){
    const int r = r0 + ty * 4 + a;
    nr[a] = r < nrow ? pos + r / G + 1 : 0;
  }
  const int lr = tid >> 2, lq = tid & 3;
  const int qrow = r0 + lr, qt = qrow / G, qj = qrow - qt * G;
  const bool qok = qrow < nrow;
  const float* qsrc = qr + (long long)qt * ldq + (kv * G + qj) * 256;
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
      float4 q0 = make_float4(0.f, 0.f, 0.f, 0.f), q1 = q0;
      if(qok){ q0 = *(const float4*)(qsrc + b * 32 + lq * 8); q1 = *(const float4*)(qsrc + b * 32 + lq * 8 + 4); }
      uint4 kw = make_uint4(0u, 0u, 0u, 0u);
      if(kp < nblk) kw = KLDG((const uint4*)(Kc + ((long long)kp * n_kv + kv) * 256 + b * 32 + lq * 8));
      const float qa[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
      const unsigned int kk[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
      for(int e = 0; e < 8; ++e) Qs[lq * 8 + e][lr] = qa[e];
#pragma unroll
      for(int e = 0; e < 4; ++e){
        Ks[lq * 8 + 2 * e][lr] = hf((unsigned short)(kk[e] & 0xFFFFu));
        Ks[lq * 8 + 2 * e + 1][lr] = hf((unsigned short)(kk[e] >> 16));
      }
      __syncthreads();
#pragma unroll 8
      for(int i = 0; i < 32; ++i){
        const float4 qv = *(const float4*)&Qs[i][ty * 4];
        const float4 k4 = *(const float4*)&Ks[i][tx * 4];
        const float qa4[4] = {qv.x, qv.y, qv.z, qv.w}, ka[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
        for(int a = 0; a < 4; ++a)
#pragma unroll
          for(int c = 0; c < 4; ++c) acc[a][c] += ka[c] * qa4[a];
      }
    }
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
#pragma unroll 1
    for(int q = 0; q < 4; ++q){
      __syncthreads();
      {
        const int key = tid >> 2, f16 = (tid & 3) * 16, p = p0 + key;
        float vv[16];
#pragma unroll
        for(int e = 0; e < 16; ++e) vv[e] = 0.f;
        if(p < nblk){
          const uint4* src = (const uint4*)(Vc + ((long long)p * n_kv + kv) * 256 + q * 64 + f16);
#pragma unroll
          for(int u = 0; u < 2; ++u){
            const uint4 w = KLDG(src + u);
            const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for(int e = 0; e < 4; ++e){
              vv[u * 8 + 2 * e] = hf((unsigned short)(ww[e] & 0xFFFFu));
              vv[u * 8 + 2 * e + 1] = hf((unsigned short)(ww[e] >> 16));
            }
          }
        }
#pragma unroll
        for(int e = 0; e < 16; e += 4) *(float4*)&Vs[key][f16 + e] = make_float4(vv[e], vv[e + 1], vv[e + 2], vv[e + 3]);
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

/* --- small tiles for sm_30 (63 registers per thread): 64 tokens x 64 rows per block, 4 x 4 per thread */
#define MS_B 64
#define MS_LD 68

struct MSShared {
  float A[32 * MS_LD];
  float B[32 * MS_LD];
  float d[MS_B];
};

extern "C" __global__ void __launch_bounds__(256) k_mm_g4s(const unsigned char* __restrict__ Wb,
    const unsigned short* __restrict__ Sd, int R, int C, const float* __restrict__ X, int ldx, float* __restrict__ Y,
    int ldy, int T){
  /* k_mm_g4 with 64 x 64 tiles; grid (R / 64, ceil(T / 64), splits) */
  __shared__ MSShared S;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  const int r0 = blockIdx.x * MS_B, t0 = blockIdx.y * MS_B, NPH = C >> 9, NB = C >> 7;
  const int b_lo = (int)((long long)blockIdx.z * NB / gridDim.z), b_hi = (int)((long long)(blockIdx.z + 1) * NB / gridDim.z);
  float acc[4][4];
#pragma unroll
  for(int i = 0; i < 4; ++i)
#pragma unroll
    for(int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  for(int cb = b_lo; cb < b_hi; ++cb){
    __syncthreads();
    if(t < MS_B){
      const int row = r0 + t, rr = row & 15, bb = rr & ~1;
      const int l = ((bb >> 3) & 1) | (((bb >> 2) & 1) << 1) | (((bb >> 1) & 1) << 2);
      S.d[t] = h2f(KLDG(Sd + (((long long)(row >> 4) * NPH + (cb >> 2)) * 32 + (cb & 3) * 8 + l) * 2 + (rr & 1)));
    }
    __syncthreads();
    for(int st = 0; st < 4; ++st){
      const int k0 = cb * 128 + st * 32;
      if(st) __syncthreads();
      {
        const int ph = k0 >> 9, cq0 = (k0 & 511) >> 2;
        for(int i = t; i < MS_B * 8; i += 256){
          const int r = i & (MS_B - 1), bq = i >> 6, row = r0 + r, rr = row & 15;
          const int cq = cq0 + bq, lane = cq >> 2;
          const long long u = (((long long)(row >> 4) * NPH + ph) * 4 + (rr >> 2)) * 32 + lane;
          unsigned int c = KLDG(Wb + u * 16 + (rr & 3) * 4 + (cq & 3));
          const float dr = S.d[r], nr = -dr;
          float* b = S.B + (bq * 4) * MS_LD + r;
#pragma unroll
          for(int q = 0; q < 4; ++q){
            const unsigned int c3 = (c * 171u) >> 9, dg = c - 3u * c3;
            b[q * MS_LD] = dg == 0u ? nr : (dg == 1u ? 0.f : dr);
            c = c3;
          }
        }
        for(int i = t; i < MS_B * 8; i += 256){
          const int tok = i >> 3, k4 = i & 7;
          float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
          if(t0 + tok < T) v = KLDG((const float4*)(X + (long long)(t0 + tok) * ldx + k0) + k4);
          float* a = S.A + (4 * k4) * MS_LD + tok;
          a[0] = v.x; a[MS_LD] = v.y; a[2 * MS_LD] = v.z; a[3 * MS_LD] = v.w;
        }
      }
      __syncthreads();
#pragma unroll 4
      for(int k = 0; k < 32; ++k){
        const float4 a4 = *(const float4*)(S.A + k * MS_LD + tx * 4);
        const float4 b4 = *(const float4*)(S.B + k * MS_LD + ty * 4);
        const float a[4] = {a4.x, a4.y, a4.z, a4.w}, b[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
        for(int i = 0; i < 4; ++i)
#pragma unroll
          for(int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
      }
    }
  }
  float* Yz = Y + (long long)blockIdx.z * T * ldy;
#pragma unroll
  for(int i = 0; i < 4; ++i){
    const int tok = t0 + tx * 4 + i;
    if(tok >= T) continue;
    *(float4*)(Yz + (long long)tok * ldy + r0 + ty * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
  }
}

extern "C" __global__ void k_add2(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n){
  /* y = a + b (two cards' parts in card order; y may be a or b) */
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) y[i] = a[i] + b[i];
}

#define KB_T 64
#define KB_R 128
#define KB_LDA 68
#define KB_LDB 132
/* --- Kepler tiles (K420, sm_30): 64 tokens x 128 rows per block, 4 x 8 per thread (63 registers), 16-column steps
   (13 KB of shared memory: 3 blocks per SMX), the step's weight bytes staged with 16-byte loads; launched with 8-byte
   shared banks. 106 GFLOPS on the K420 (the 64 x 64 kernel: 57). */
template <int BK>
__device__ __forceinline__ void kep3_core(const unsigned char* __restrict__ Wb, const unsigned short* __restrict__ Sd,
    int R, int C, const float* __restrict__ X, int ldx, float* __restrict__ Y, int ldy, int T){
  __shared__ float A[BK * KB_LDA];
  __shared__ float B[BK * KB_LDB];
  __shared__ float dsh[KB_R];
  __shared__ uint4 raw[32 * (BK / 16)];
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  const int r0 = blockIdx.x * KB_R, t0 = blockIdx.y * KB_T, NPH = C >> 9, NB = C >> 7;
  const int b_lo = (int)((long long)blockIdx.z * NB / gridDim.z), b_hi = (int)((long long)(blockIdx.z + 1) * NB / gridDim.z);
  const int NL = BK / 16;                      /* lanes (16-column words) per step */
  float acc[4][8];
#pragma unroll
  for(int i = 0; i < 4; ++i)
#pragma unroll
    for(int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  for(int cb = b_lo; cb < b_hi; ++cb){
    __syncthreads();
    if(t < KB_R){
      const int row = r0 + t, rr = row & 15, bb = rr & ~1;
      const int l = ((bb >> 3) & 1) | (((bb >> 2) & 1) << 1) | (((bb >> 1) & 1) << 2);
      dsh[t] = h2f(KLDG(Sd + (((long long)(row >> 4) * NPH + (cb >> 2)) * 32 + (cb & 3) * 8 + l) * 2 + (rr & 1)));
    }
    for(int st = 0; st < 128 / BK; ++st){
      const int k0 = cb * 128 + st * BK;
      __syncthreads();
      const int ph = k0 >> 9, lane0 = (k0 & 511) >> 4;
      if(t < 32 * NL){
        const int g = t / (4 * NL), ch = (t / NL) & 3, ln = t % NL;
        raw[t] = KLDG((const uint4*)Wb + ((((long long)((r0 >> 4) + g) * NPH + ph) * 4 + ch) * 32 + lane0 + ln));
      }
      for(int i = t; i < KB_T * (BK / 4); i += 256){
        const int tok = i / (BK / 4), k4 = i % (BK / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if(t0 + tok < T) v = KLDG((const float4*)(X + (long long)(t0 + tok) * ldx + k0) + k4);
        float* a = A + (4 * k4) * KB_LDA + tok;
        a[0] = v.x; a[KB_LDA] = v.y; a[2 * KB_LDA] = v.z; a[3 * KB_LDA] = v.w;
      }
      __syncthreads();
      const unsigned char* rb = (const unsigned char*)raw;
      for(int i = t; i < KB_R * (BK / 4); i += 256){
        const int r = i & (KB_R - 1), bq = i >> 7, rr = r & 15;
        unsigned int c = rb[(((r >> 4) * 4 + (rr >> 2)) * NL + (bq >> 2)) * 16 + (rr & 3) * 4 + (bq & 3)];
        const float dr = dsh[r], nr = -dr;
        float* b = B + (bq * 4) * KB_LDB + r;
#pragma unroll
        for(int q = 0; q < 4; ++q){
          const unsigned int c3 = (c * 171u) >> 9, dg = c - 3u * c3;
          b[q * KB_LDB] = dg == 0u ? nr : (dg == 1u ? 0.f : dr);
          c = c3;
        }
      }
      __syncthreads();
#pragma unroll
      for(int k = 0; k < BK; ++k){
        const float4 a4 = *(const float4*)(A + k * KB_LDA + tx * 4);
        const float4 b0 = *(const float4*)(B + k * KB_LDB + ty * 4), b1 = *(const float4*)(B + k * KB_LDB + 64 + ty * 4);
        const float a[4] = {a4.x, a4.y, a4.z, a4.w};
        const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for(int i = 0; i < 4; ++i)
#pragma unroll
          for(int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
      }
    }
  }
  float* Yz = Y + (long long)blockIdx.z * T * ldy;
#pragma unroll
  for(int i = 0; i < 4; ++i){
    const int tok = t0 + tx * 4 + i;
    if(tok >= T) continue;
    *(float4*)(Yz + (long long)tok * ldy + r0 + ty * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    *(float4*)(Yz + (long long)tok * ldy + r0 + 64 + ty * 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
  }
}
extern "C" __global__ void __launch_bounds__(256) k_mm_g4k(const unsigned char* __restrict__ Wb,
    const unsigned short* __restrict__ Sd, int R, int C, const float* __restrict__ X, int ldx, float* __restrict__ Y,
    int ldy, int T){
  /* grid (R / 128, ceil(T / 64), splits) */
  kep3_core<16>(Wb, Sd, R, C, X, ldx, Y, ldy, T);
}

/* --- k_att_flash16_T for Kepler (sm_30, 63 registers): 32 query rows per block, 2 rows x 4 positions (scores) and
   2 rows x 16 features (values) per thread, 22 KB of shared memory (2 blocks per SMX). Same arithmetic. */
extern "C" __global__ void __launch_bounds__(256) k_att_flash16k_T(const unsigned short* __restrict__ Kc,
    const unsigned short* __restrict__ Vc, const float* __restrict__ qr, int ldq, const float* __restrict__ gatev,
    int ldg, float* __restrict__ att, int lda, int pos, int T, int n_kv, int G, int NH, float* __restrict__ PO){
  __shared__ float U[32 * 36 + 32 * 68];       /* scores: Q[32][36], K[32][68]; values: V[32][68] */
  __shared__ float Ps[32][68];
  float (*Qs)[36] = (float (*)[36])U;
  float (*Ks)[68] = (float (*)[68])(U + 32 * 36);
  float (*Vs)[68] = (float (*)[68])U;
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, kv = blockIdx.y;
  const int r0 = blockIdx.x * 32, nrow = T * G;
  const int rlast = min(r0 + 31, nrow - 1), nblk = pos + rlast / G + 1;
  int nr[2];
#pragma unroll
  for(int a = 0; a < 2; ++a){
    const int r = r0 + ty * 2 + a;
    nr[a] = r < nrow ? pos + r / G + 1 : 0;
  }
  const int lr = tid >> 3, lq = tid & 7;
  const int qrow = r0 + lr, qt = qrow / G, qj = qrow - qt * G;
  const bool qok = qrow < nrow;
  const float* qsrc = qr + (long long)qt * ldq + (kv * G + qj) * 256;
  float m[2], l[2], o[2][16];
#pragma unroll
  for(int a = 0; a < 2; ++a){
    m[a] = KNEGINF; l[a] = 0.f;
#pragma unroll
    for(int f = 0; f < 16; ++f) o[a][f] = 0.f;
  }
  const int ntl = (nblk + 63) / 64, z = blockIdx.z, Z = gridDim.z;
  const int tlo = (int)((long long)z * ntl / Z), thi = (int)((long long)(z + 1) * ntl / Z);
  for(int p0 = tlo * 64; p0 < thi * 64; p0 += 64){
    float acc[2][4];
#pragma unroll
    for(int a = 0; a < 2; ++a)
#pragma unroll
      for(int c = 0; c < 4; ++c) acc[a][c] = 0.f;
    const int kp = p0 + (tid >> 2), kq = tid & 3;
    for(int b = 0; b < 8; ++b){
      __syncthreads();
      float4 q0 = make_float4(0.f, 0.f, 0.f, 0.f);
      if(qok) q0 = *(const float4*)(qsrc + b * 32 + lq * 4);
      Qs[lq * 4][lr] = q0.x; Qs[lq * 4 + 1][lr] = q0.y; Qs[lq * 4 + 2][lr] = q0.z; Qs[lq * 4 + 3][lr] = q0.w;
      uint4 kw = make_uint4(0u, 0u, 0u, 0u);
      if(kp < nblk) kw = KLDG((const uint4*)(Kc + ((long long)kp * n_kv + kv) * 256 + b * 32 + kq * 8));
      const unsigned int kk[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
      for(int e = 0; e < 4; ++e){
        Ks[kq * 8 + 2 * e][tid >> 2] = hf((unsigned short)(kk[e] & 0xFFFFu));
        Ks[kq * 8 + 2 * e + 1][tid >> 2] = hf((unsigned short)(kk[e] >> 16));
      }
      __syncthreads();
#pragma unroll 8
      for(int i = 0; i < 32; ++i){
        const float2 qv = *(const float2*)&Qs[i][ty * 2];
        const float4 k4 = *(const float4*)&Ks[i][tx * 4];
        const float qa[2] = {qv.x, qv.y}, ka[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
        for(int a = 0; a < 2; ++a)
#pragma unroll
          for(int c = 0; c < 4; ++c) acc[a][c] += ka[c] * qa[a];
      }
    }
#pragma unroll
    for(int a = 0; a < 2; ++a){
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
      *(float4*)&Ps[ty * 2 + a][tx * 4] = make_float4(pv[0], pv[1], pv[2], pv[3]);
    }
#pragma unroll 1
    for(int q = 0; q < 4; ++q){
#pragma unroll 1
      for(int kh = 0; kh < 2; ++kh){
        __syncthreads();
        {
          const int key = tid >> 3, f8 = (tid & 7) * 8, p = p0 + kh * 32 + key;
          float vv[8];
#pragma unroll
          for(int e = 0; e < 8; ++e) vv[e] = 0.f;
          if(p < nblk){
            const uint4 w = KLDG((const uint4*)(Vc + ((long long)p * n_kv + kv) * 256 + q * 64 + f8));
            const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for(int e = 0; e < 4; ++e){
              vv[2 * e] = hf((unsigned short)(ww[e] & 0xFFFFu));
              vv[2 * e + 1] = hf((unsigned short)(ww[e] >> 16));
            }
          }
          *(float4*)&Vs[key][f8] = make_float4(vv[0], vv[1], vv[2], vv[3]);
          *(float4*)&Vs[key][f8 + 4] = make_float4(vv[4], vv[5], vv[6], vv[7]);
        }
        __syncthreads();
#pragma unroll 8
        for(int k = 0; k < 32; ++k){
          const float4 v4 = *(const float4*)&Vs[k][tx * 4];
          const float va[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
          for(int a = 0; a < 2; ++a){
            const float pa = Ps[ty * 2 + a][kh * 32 + k];
#pragma unroll
            for(int e = 0; e < 4; ++e) o[a][q * 4 + e] += pa * va[e];
          }
        }
      }
    }
  }
  if(Z > 1){
#pragma unroll
    for(int a = 0; a < 2; ++a){
      const int r = r0 + ty * 2 + a;
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
  for(int a = 0; a < 2; ++a){
    const int r = r0 + ty * 2 + a;
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
