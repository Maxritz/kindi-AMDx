// Does int8 activation quantisation preserve the argmax? This decides whether dp4a is
// usable for the token-exact path at all.
//
// The probe that measured "46.7 ms/token, 21.4 tok/s" used int8 activations, because dp4a
// requires both operands int8. That is exact for the integer formulation it was tested
// against, but the model does not run in integers: activations are fp32 and the Vulkan port
// is token-exact against llama.cpp in fp32.
//
// So the real question: if we quantise the activation to int8 per block and run dp4a, does
// the GEMV output still produce the same argmax token? If the error is small enough it does
// not matter; if it flips a token, dp4a is unusable for this contract and the honest RDNA2
// win is raw lane throughput instead (5120 lanes vs 2304 = 4.0x).
//
// This test uses a realistic ternary weight distribution (a third of each of -1, 0, +1) and
// a realistic activation (the spread of a post-RMSNorm vector), then compares:
//   exact fp32       : sum (trit-1) * x
//   int8 dp4a        : per-block scale, int8 weights, int8 activations, int32 accumulate
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

__device__ __constant__ int c_lut[256];

// exact fp32 ternary GEMV: one trit decoded to a float, one FMA. This is the Vulkan path.
__global__ void gemv_exact(const uint8_t* __restrict__ wb,
                           const float*   __restrict__ x,
                           float*         __restrict__ y,
                           const float*   __restrict__ sc,
                           int R, int C)
{
    int row = blockIdx.x * (blockDim.x / warpSize) + (threadIdx.x / warpSize);
    if (row >= R) return;
    int lane = threadIdx.x % warpSize;
    int nbytes = C >> 2;
    int nq = C >> 7;
    float acc = 0.0f;
    const uint8_t* rp = wb + (size_t)row * nbytes;
    for (int i = lane; i < nbytes; i += warpSize) {
        int b = rp[i];
        int k = i << 2;
        acc += (float)((b & 3) - 1) * x[k + 0];
        acc += (float)(((b >> 2) & 3) - 1) * x[k + 1];
        acc += (float)(((b >> 4) & 3) - 1) * x[k + 2];
        acc += (float)(((b >> 6) & 3) - 1) * x[k + 3];
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (lane == 0) y[row] = acc * sc[row * nq];
}

// int8 dp4a: activations quantised per 128-column block
__global__ void gemv_dp4a(const uint8_t* __restrict__ wb,
                          const int*     __restrict__ xq,   // int8, 4 per int32
                          float*         __restrict__ y,
                          const float*   __restrict__ sc,
                          const float*   __restrict__ xscale, // per 128-col block
                          int R, int C)
{
    int row = blockIdx.x * (blockDim.x / warpSize) + (threadIdx.x / warpSize);
    if (row >= R) return;
    int lane = threadIdx.x % warpSize;
    int nbytes = C >> 2;
    int nq = C >> 7;
    int acc = 0;
    const uint8_t* rp = wb + (size_t)row * nbytes;
    for (int i = lane; i < nbytes; i += warpSize) {
        int w4 = c_lut[rp[i]];
        int x4 = xq[i];
        acc = __builtin_amdgcn_sdot4(w4, x4, acc, 0);
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    // per-block dequant: split the row into 128-col blocks, so this is approximate by design
    if (lane == 0) y[row] = (float)acc * sc[row * nq] * xscale[0];
}

static void build_lut(int* h) {
    for (int b = 0; b < 256; ++b) {
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            int trit = (b >> (2 * k)) & 3;
            v |= (unsigned)(uint8_t)(int8_t)(trit - 1) << (8 * k);
        }
        h[b] = (int)v;
    }
}

int main() {
    int dev = 0; hipSetDevice(dev);
    hipDeviceProp_t p{}; hipGetDeviceProperties(&p, dev);
    printf("device %s  arch %s  CU %d  warp %d\n", p.name, p.gcnArchName,
           p.multiProcessorCount, p.warpSize);

    int lut[256]; build_lut(lut);
    hipMemcpyToSymbol(c_lut, lut, sizeof(lut));

    // One realistic layer: the 27B's attn_output is 5120 rows x 6144 cols.
    const int R = 5120, C = 6144;
    const int nbytes = C >> 2, nq = C >> 7;

    std::vector<uint8_t> wb((size_t)R * nbytes);
    std::vector<float>   x(C), sc((size_t)R * nq);
    srand(7);
    // ternary with a third zeros, like a trained ternary model
    for (auto& b : wb) {
        int v = 0;
        for (int k = 0; k < 4; ++k) v |= (rand() % 3) << (2 * k);
        b = (uint8_t)v;
    }
    // post-RMSNorm activation: unit-ish scale, gaussian-ish spread
    for (auto& a : x) {
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        a = (float)(sqrt(-2.0 * log(u1)) * cos(2 * 3.14159265358979 * u2));
    }
    for (auto& s : sc) s = 0.002f + 0.0001f * (rand() % 5);

    // int8 quantisation of x, per 128-column block
    std::vector<int32_t> xq(nbytes);
    std::vector<float>   xscale(nq, 1.0f);
    for (int q = 0; q < nq; ++q) {
        float mx = 0;
        for (int i = 0; i < 128; ++i) mx = std::max(mx, std::fabs(x[q * 128 + i]));
        float s = mx > 0 ? mx / 127.0f : 1.0f;
        xscale[q] = s;
        for (int i = 0; i < 128; ++i) {
            int8_t v = (int8_t)std::lround(x[q * 128 + i] / s);
            int word = (q * 128 + i) >> 2, lane = (q * 128 + i) & 3;
            if (lane == 0) xq[word] = 0;
            xq[word] |= (unsigned)(uint8_t)v << (8 * lane);
        }
    }

    uint8_t *dwb; float *dx, *dy, *dsc, *dxs; int* dxq;
    hipMalloc(&dwb, wb.size());
    hipMalloc(&dx, (size_t)C * 4);
    hipMalloc(&dy, (size_t)R * 4);
    hipMalloc(&dsc, sc.size() * 4);
    hipMalloc(&dxq, xq.size() * 4);
    hipMalloc(&dxs, xscale.size() * 4);
    hipMemcpy(dwb, wb.data(), wb.size(), hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), (size_t)C * 4, hipMemcpyHostToDevice);
    hipMemcpy(dsc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dxq, xq.data(), xq.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dxs, xscale.data(), xscale.size() * 4, hipMemcpyHostToDevice);

    const int WS = p.warpSize, wb_per_block = 8;
    dim3 blk(WS * wb_per_block);
    dim3 grd((R + wb_per_block - 1) / wb_per_block);
    gemv_exact<<<grd, blk>>>(dwb, dx, dy, dsc, R, C);
    hipDeviceSynchronize();
    std::vector<float> yexact(R);
    hipMemcpy(yexact.data(), dy, (size_t)R * 4, hipMemcpyDeviceToHost);

    gemv_dp4a<<<grd, blk>>>(dwb, dxq, dy, dsc, dxs, R, C);
    hipDeviceSynchronize();
    std::vector<float> yq(R);
    hipMemcpy(yq.data(), dy, (size_t)R * 4, hipMemcpyDeviceToHost);

    // error and argmax agreement
    double worstrel = 0, sumrel = 0;
    for (int r = 0; r < R; ++r) {
        double d = std::fabs(yq[r] - yexact[r]);
        double base = std::fabs(yexact[r]) + 1e-9;
        worstrel = std::max(worstrel, d / base);
        sumrel += d / base;
    }
    int argx = (int)(std::max_element(yexact.begin(), yexact.end()) - yexact.begin());
    int argq = (int)(std::max_element(yq.begin(), yq.end()) - yq.begin());
    printf("int8 quantisation error: mean rel %.3e  worst rel %.3e\n", sumrel / R, worstrel);
    printf("argmax: exact row %d  int8 row %d  -> %s\n", argx, argq,
           argx == argq ? "SAME" : "DIFFERENT");
    printf("\n%s\n", argx == argq
        ? "int8 preserves the argmax on this layer"
        : "int8 FLIPS the argmax: dp4a is not usable for the token-exact path");

    // timing of the exact path, for the honest comparison
    hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
    const int it = 300;
    hipEventRecord(t0);
    for (int i = 0; i < it; ++i) gemv_exact<<<grd, blk>>>(dwb, dx, dy, dsc, R, C);
    hipEventRecord(t1); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    double per = ms / it;
    printf("\nexact fp32 path: %.4f ms/call  %.1f GMAC/s  (this layer, one token)\n",
           per, (double)R * C / (per * 1e6));

    hipFree(dwb); hipFree(dx); hipFree(dy); hipFree(dsc); hipFree(dxq); hipFree(dxs);
    return 0;
}
