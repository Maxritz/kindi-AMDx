// dp4a ternary GEMV on gfx1031: the core kernel idea for the new backend.
//
// Why this matters: on gfx803 (RX 580) there is no dp4a (vulkaninfo reports
// integerDotProduct8BitSignedAccelerated = false), so the Vulkan port had to decode every
// trit to a float and do one FMA per weight. That is ~24 instructions per weight, which is
// what capped the 27B at 5.19 tok/s. RDNA2 has v_dot4c_i32_i8, which does four int8 MACs
// in ONE instruction.
//
// The packing problem, which is the whole design:
//   PTQ1_0 stores FOUR 2-bit trits per byte (1.75 bits/weight). One dp4a multiplies four
//   int8 lanes. So a packed byte must be expanded to four int8 (one per weight) before the
//   dot. Expanding to int8 in VRAM would be 4x memory (27B -> 27 GB, does not fit), so the
//   expansion happens in registers: one 256-entry LUT turns a trit byte into four int8,
//   then one dp4a consumes it against four int8 activations.
//
// Per four weights: 1 byte load + 1 LUT load + 1 activation load + 1 dp4a, against four
// FMAs plus four decodes on the old path.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

__device__ __constant__ int c_lut[256];

// y[r] = sum_c (trit(r,c) - 1) * x[c], trits packed four per byte, 2 bits each.
__global__ void gemv_trit_dp4a(const uint8_t* __restrict__ wb,  // packed 4-trit bytes
                               const int*     __restrict__ xq,  // int8 activations, 4 per int
                               float*         __restrict__ y,
                               int R, int C)
{
    int row = blockIdx.x * (blockDim.x / warpSize) + (threadIdx.x / warpSize);
    if (row >= R) return;
    int lane = threadIdx.x % warpSize;
    int nbytes = C >> 2;                 // one byte holds four weights

    int acc = 0;
    const uint8_t* rp = wb + (size_t)row * nbytes;
    for (int i = lane; i < nbytes; i += warpSize) {
        // one byte -> four int8 trits, then four int8 MACs in one instruction
        int w4 = c_lut[rp[i]];
        int x4 = xq[i];
        acc = __builtin_amdgcn_sdot4(w4, x4, acc, 0);
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (lane == 0) y[row] = (float)acc;
}

static void build_lut(int* host) {
    for (int b = 0; b < 256; ++b) {
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            int trit = (b >> (2 * k)) & 3;      // 0,1,2
            int8_t w = (int8_t)(trit - 1);      // -1,0,+1
            v |= (unsigned)(uint8_t)w << (8 * k);
        }
        host[b] = (int)v;
    }
}

int main() {
    int dev = 0; hipSetDevice(dev);
    hipDeviceProp_t p{}; hipGetDeviceProperties(&p, dev);
    const int WS = p.warpSize;
    printf("device %s  arch %s  CU %d  warp %d\n", p.name, p.gcnArchName,
           p.multiProcessorCount, WS);

    int lut[256]; build_lut(lut);
    hipMemcpyToSymbol(c_lut, lut, sizeof(lut));

    const int R = 4096, C = 4096;
    const int nbytes = C >> 2, nwords = C >> 2;

    std::vector<uint8_t> wb((size_t)R * nbytes);
    std::vector<int8_t>  xv(C);
    srand(1234);
    for (auto& b : wb) {
        int v = 0;
        for (int k = 0; k < 4; ++k) v |= (rand() % 3) << (2 * k);
        b = (uint8_t)v;
    }
    for (auto& a : xv) a = (int8_t)((rand() % 31) - 15);

    std::vector<int32_t> xq(nwords);
    for (int i = 0; i < nwords; ++i) memcpy(&xq[i], &xv[(size_t)i * 4], 4);

    uint8_t* dwb; int* dxq; float* dy;
    hipMalloc(&dwb, wb.size());
    hipMalloc(&dxq, xq.size() * 4);
    hipMalloc(&dy, (size_t)R * 4);
    hipMemcpy(dwb, wb.data(), wb.size(), hipMemcpyHostToDevice);
    hipMemcpy(dxq, xq.data(), xq.size() * 4, hipMemcpyHostToDevice);

    int wavesPerBlock = 8;
    dim3 blk(WS * wavesPerBlock);
    dim3 grd((R + wavesPerBlock - 1) / wavesPerBlock);
    gemv_trit_dp4a<<<grd, blk>>>(dwb, dxq, dy, R, C);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("KERNEL FAILED: %s\n", hipGetErrorString(e)); return 2; }

    std::vector<float> y(R);
    hipMemcpy(y.data(), dy, (size_t)R * 4, hipMemcpyDeviceToHost);

    // exact integer reference
    double worst = 0; double sumabs = 0;
    for (int r = 0; r < R; ++r) {
        long acc = 0;
        for (int c = 0; c < C; ++c) {
            int byte = wb[(size_t)r * nbytes + (c >> 2)];
            int trit = (byte >> (2 * (c & 3))) & 3;
            acc += (long)(trit - 1) * (long)xv[c];
        }
        double d = std::fabs((double)acc - (double)y[r]);
        if (d > worst) worst = d;
        sumabs += std::fabs((double)acc);
    }
    printf("dp4a ternary GEMV vs integer reference: max abs err %.1f\n", worst);
    printf("  %s\n", worst == 0.0 ? "PASS (exact)" : "FAIL");

    const int iters = 500;
    hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    for (int i = 0; i < iters; ++i) gemv_trit_dp4a<<<grd, blk>>>(dwb, dxq, dy, R, C);
    hipEventRecord(t1); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    double per = ms / iters;
    double macs = (double)R * C;
    double wbytes = (double)R * nbytes;
    printf("wave%-3d  %.4f ms/call  %.1f GMAC/s  %.1f GB/s packed weights\n",
           WS, per, macs / (per * 1e6), wbytes / (per * 1e6));
    printf("  equivalent dense-27B time: %.1f ms/token  (27e9 weights)\n",
           27e9 / (macs / (per * 1e-3)) * 1000.0);

    hipFree(dwb); hipFree(dxq); hipFree(dy);
    hipEventDestroy(t0); hipEventDestroy(t1);
    return 0;
}
