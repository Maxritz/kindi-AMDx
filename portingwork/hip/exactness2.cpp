// Does int8 activation quantisation preserve the argmax? Corrected.
//
// The first version dequantised with a single xscale[0] for the whole row, which is wrong:
// int8 activations are quantised per 128-column block, so each block must be dequantised
// with its own scale before it joins the fp32 total. That mistake made the measured error
// meaningless.
//
// Correct shape of a dp4a ternary GEMV:
//   for each 128-weight block:
//       int acc = 0;
//       for 32 int32 words:  acc = dp4a(w4, x4, acc)
//       total += (float)acc * wblock_scale * xblock_scale;
// So the dequant happens once per 128 weights, not once per row.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

__device__ __constant__ int c_lut[256];

__global__ void gemv_exact(const uint8_t* __restrict__ wb,
                           const float*   __restrict__ x,
                           float*         __restrict__ y,
                           const float*   __restrict__ sc,
                           int R, int C)
{
    int row = blockIdx.x * (blockDim.x / warpSize) + (threadIdx.x / warpSize);
    if (row >= R) return;
    int lane = threadIdx.x % warpSize;
    int nbytes = C >> 2, nq = C >> 7;
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

// dp4a with correct per-block dequant.
// One wave owns one row. Each lane owns a strided set of the 32 words inside a block, so
// the block's 128 weights are covered by 32 lanes in one pass.
__global__ void gemv_dp4a(const uint8_t* __restrict__ wb,
                          const int*     __restrict__ xq,
                          float*         __restrict__ y,
                          const float*   __restrict__ sc,
                          const float*   __restrict__ xscale,
                          int R, int C)
{
    int row = blockIdx.x * (blockDim.x / warpSize) + (threadIdx.x / warpSize);
    if (row >= R) return;
    int lane = threadIdx.x % warpSize;
    int nbytes = C >> 2;      // bytes per row
    int nq = C >> 7;          // 128-weight blocks per row
    const uint8_t* rp = wb + (size_t)row * nbytes;

    float total = 0.0f;
    for (int q = 0; q < nq; ++q) {
        // 32 words = 128 weights for this block, one word per lane
        int i = q * 32 + lane;
        int acc = __builtin_amdgcn_sdot4(c_lut[rp[i]], xq[i], 0, 0);
        for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_down(acc, o);
        if (lane == 0) total += (float)acc * sc[row * nq + q] * xscale[q];
    }
    if (lane == 0) y[row] = total;
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

    const int R = 5120, C = 6144;
    const int nbytes = C >> 2, nq = C >> 7;

    std::vector<uint8_t> wb((size_t)R * nbytes);
    std::vector<float>   x(C), sc((size_t)R * nq);
    srand(7);
    for (auto& b : wb) {
        int v = 0;
        for (int k = 0; k < 4; ++k) v |= (rand() % 3) << (2 * k);
        b = (uint8_t)v;
    }
    for (auto& a : x) {
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        a = (float)(sqrt(-2.0 * log(u1)) * cos(2 * 3.14159265358979 * u2));
    }
    for (auto& s : sc) s = 0.002f + 0.0001f * (rand() % 5);

    // per-128-block int8 quantisation of the activation
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

    const int WS = p.warpSize, wpb = 8;
    dim3 blk(WS * wpb);
    dim3 grd((R + wpb - 1) / wpb);

    gemv_exact<<<grd, blk>>>(dwb, dx, dy, dsc, R, C);
    hipDeviceSynchronize();
    std::vector<float> ye(R);
    hipMemcpy(ye.data(), dy, (size_t)R * 4, hipMemcpyDeviceToHost);

    gemv_dp4a<<<grd, blk>>>(dwb, dxq, dy, dsc, dxs, R, C);
    hipDeviceSynchronize();
    std::vector<float> yq(R);
    hipMemcpy(yq.data(), dy, (size_t)R * 4, hipMemcpyDeviceToHost);

    double worstrel = 0, meanrel = 0, absmax = 0;
    for (int r = 0; r < R; ++r) {
        double d = std::fabs(yq[r] - ye[r]);
        double base = std::fabs(ye[r]) + 1e-9;
        worstrel = std::max(worstrel, d / base);
        meanrel += d / base;
        absmax = std::max(absmax, d);
    }
    printf("int8 vs exact: mean rel %.3e   worst rel %.3e   max abs %.3e\n",
           meanrel / R, worstrel, absmax);

    int ax = (int)(std::max_element(ye.begin(), ye.end()) - ye.begin());
    int aq = (int)(std::max_element(yq.begin(), yq.end()) - yq.begin());
    printf("argmax: exact %d  int8 %d  -> %s\n", ax, aq, ax == aq ? "SAME" : "DIFFERENT");

    // How many rows land in a different order? That is what decides whether a token flips
    // downstream, not just the top-1.
    std::vector<int> idx(R);
    for (int i = 0; i < R; ++i) idx[i] = i;
    auto byE = idx, byQ = idx;
    std::sort(byE.begin(), byE.end(), [&](int a, int b){ return ye[a] > ye[b]; });
    std::sort(byQ.begin(), byQ.end(), [&](int a, int b){ return yq[a] > yq[b]; });
    int topk = 20, same = 0;
    for (int i = 0; i < topk; ++i) if (byE[i] == byQ[i]) ++same;
    printf("top-%d rows identical: %d/%d\n", topk, same, topk);

    hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
    const int it = 300;
    hipEventRecord(t0);
    for (int i = 0; i < it; ++i) gemv_exact<<<grd, blk>>>(dwb, dx, dy, dsc, R, C);
    hipEventRecord(t1); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    double pe = ms / it;

    hipEventRecord(t0);
    for (int i = 0; i < it; ++i) gemv_dp4a<<<grd, blk>>>(dwb, dxq, dy, dsc, dxs, R, C);
    hipEventRecord(t1); hipEventSynchronize(t1);
    hipEventElapsedTime(&ms, t0, t1);
    double pq = ms / it;

    printf("\nexact fp32 : %.4f ms/call  %.1f GMAC/s\n", pe, (double)R * C / (pe * 1e6));
    printf("int8 dp4a  : %.4f ms/call  %.1f GMAC/s   (%.2fx)\n",
           pq, (double)R * C / (pq * 1e6), pe / pq);

    hipFree(dwb); hipFree(dx); hipFree(dy); hipFree(dsc); hipFree(dxq); hipFree(dxs);
    return 0;
}
