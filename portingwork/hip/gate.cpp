// Gate test: does hipcc compile and RUN a kernel on this card?
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void saxpy(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) { printf("hipGetDevice FAILED\n"); return 1; }
    hipDeviceProp_t p{};
    hipGetDeviceProperties(&p, dev);
    printf("device      : %s\n", p.name);
    printf("gcnArchName : %s\n", p.gcnArchName);
    printf("multiProcessorCount: %d\n", p.multiProcessorCount);
    printf("clockRate   : %.2f GHz\n", p.clockRate / 1e6);
    printf("warpSize    : %d\n", p.warpSize);
    printf("totalGlobalMem: %.2f GB\n", p.totalGlobalMem / 1073741824.0);
    printf("sharedMemPerBlock: %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("regsPerBlock: %d\n", p.regsPerBlock);
    printf("memoryClockRate: %.2f GHz  busWidth: %d\n",
           p.memoryClockRate / 1e6, p.memoryBusWidth);

    const int n = 1 << 20;
    float *x, *y;
    hipMalloc(&x, n * sizeof(float));
    hipMalloc(&y, n * sizeof(float));
    float* hx = new float[n];
    float* hy = new float[n];
    for (int i = 0; i < n; ++i) { hx[i] = 1.0f; hy[i] = 2.0f; }
    hipMemcpy(x, hx, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(y, hy, n * sizeof(float), hipMemcpyHostToDevice);
    saxpy<<<(n + 255) / 256, 256>>>(y, x, 3.0f, n);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("KERNEL FAILED: %s\n", hipGetErrorString(e)); return 2; }
    hipMemcpy(hy, y, n * sizeof(float), hipMemcpyDeviceToHost);
    printf("saxpy y[0] = %.1f (expect 5.0)\n", hy[0]);
    hipFree(x); hipFree(y);
    delete[] hx; delete[] hy;
    printf("OK\n");
    return 0;
}
