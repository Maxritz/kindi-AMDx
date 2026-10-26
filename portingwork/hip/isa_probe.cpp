// ISA probe for gfx1031: which dot-product and matrix instructions actually exist?
//
// KANDS has no entry for v_dot4_i32_i8 / v_dot8_i32_i4 / wmma on gfx1031 and explicitly
// said to probe rather than assume. Each test compiles in isolation so one unsupported
// builtin does not hide the others.
#include <hip/hip_runtime.h>
#include <cstdint>

// 1. dp4a: four signed int8 dot product, 32-bit accumulate.
extern "C" __global__ void k_dot4(const int* a, const int* b, int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = __builtin_amdgcn_sdot4(a[i], b[i], 0, 0);
}

// 2. sdot8: eight int4 dot product (RDNA3+).
extern "C" __global__ void k_dot8(const int* a, const int* b, int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = __builtin_amdgcn_sdot8(a[i], b[i], 0, 0);
}

// 3. packed f16 dot, two at a time.
extern "C" __global__ void k_fdot2(const float* a, const float* b, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = __builtin_amdgcn_fdot2(a[i], b[i], 0.0f, false);
}

// 4. WMMA (RDNA3 matrix cores).
extern "C" __global__ void k_wmma(const short* a, const short* b, int* c) {
    *c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, 0, false, 0, 0, 0);
}
