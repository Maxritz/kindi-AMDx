// kindi_hip.cpp - Kindi inference engine, HIP backend, RX 6700 XT (gfx1031).
//
// Milestone 1 of HIP-PORT.md: GGUF read directly (no conversion, no pack), device init,
// buffer management, the tracing contract, and a --selftest that proves the PTQ1_0 GEMV
// against a host reference taken from the raw GGUF bytes.
//
// Design notes that matter, all measured (see HIP-PORT.md):
//
//   - Exact fp32 arithmetic. Ternary weights are decoded to floats and accumulated in
//     fp32, exactly as the Vulkan engine does, because token exactness requires it.
//     dp4a was probed and refuted: int8 activations destroy the output ranking (1 of the
//     top 20 rows survived) and it is 0.49x slower once each 128-weight block needs its
//     own dequantisation.
//   - Weights stay packed in VRAM. A 27B at 1.75 bits is 5.9 GB and fits in 11.98 GB; at
//     8 bits it would be 27 GB and would not.
//   - wave32 with RPW rows per wave. RDNA2 wave64 is two wave32 passes, so width has to
//     come from rows in flight, not from the wave.
//   - The trit table is floats in LDS (256 x 5 x 4 = 5 KB), so the inner loop is an LDS
//     load and an FMA per weight instead of a shift, an and, a convert and a subtract.
//     This was 1.09x on the Vulkan side and the reasoning is architecture-independent.
#include <hip/hip_runtime.h>
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <rocblas/rocblas.h>
#include "trace_hip.h"

// ----------------------------------------------------------------------------- device
struct Dev {
    hipDevice_t dev;
    hipDeviceProp_t prop;
    int arch_major = 0, arch_minor = 0;
};
static Dev G;

struct Buf {
    void* p = nullptr;
    size_t size = 0;
    bool mapped = false;
};
static uint64_t g_alloc_total = 0;

static Buf mkbuf(size_t bytes) {
    Buf b;
    b.size = bytes ? bytes : 4;
    if (hipMalloc(&b.p, b.size) != hipSuccess) {
        fprintf(stderr, "hipMalloc %zu failed: %s\n", b.size, hipGetErrorString(hipGetLastError()));
        exit(1);
    }
    g_alloc_total += b.size;
    ktrace::count_alloc(b.size);
    return b;
}
static Buf mkbuf_host(size_t bytes) {
    Buf b;
    b.size = bytes ? bytes : 4;
    if (hipHostMalloc(&b.p, b.size, hipHostMallocDefault) != hipSuccess) {
        fprintf(stderr, "hipHostMalloc %zu failed\n", b.size);
        exit(1);
    }
    b.mapped = true;
    ktrace::count_alloc(b.size);
    return b;
}
static void freebuf(Buf& b) {
    if (!b.p) return;
    ktrace::count_free();
    if (g_alloc_total >= b.size) g_alloc_total -= b.size;
    if (b.mapped) hipHostFree(b.p); else hipFree(b.p);
    b = {};
}
static void upload(Buf* dst, const void* src, size_t n) {
    double t0 = ktrace::now_us();
    if (hipMemcpy(dst->p, src, n, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "upload %zu failed\n", n);
        exit(1);
    }
    ktrace::count_xfer(n, ktrace::now_us() - t0);
}
static void download(void* dst, const Buf& src, size_t n) {
    double t0 = ktrace::now_us();
    hipMemcpy(dst, src.p, n, hipMemcpyDeviceToHost);
    ktrace::count_xfer(n, ktrace::now_us() - t0);
}

static void initdev() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) {
        fprintf(stderr,
            "no HIP device. On this machine that is almost always a DLL search-path problem,\n"
            "not a driver problem: amdhip64_7.dll loads rocm_kpack.dll and amd_comgr.dll from\n"
            "the EXE DIRECTORY before PATH. Copy those three next to the exe, or run from\n"
            "C:\\Rocm10\\bin. See HIP-ENV.md.\n");
        exit(1);
    }
    G.dev = 0;
    hipSetDevice(G.dev);
    hipGetDeviceProperties(&G.prop, G.dev);
    int a = 0;
    if (hipDeviceGetAttribute(&a, hipDeviceAttributeComputeCapabilityMajor, G.dev) == hipSuccess)
        G.arch_major = a;
    if (hipDeviceGetAttribute(&a, hipDeviceAttributeComputeCapabilityMinor, G.dev) == hipSuccess)
        G.arch_minor = a;

    fprintf(stderr, "device  : %s\n", G.prop.name);
    fprintf(stderr, "arch    : %s (cc %d.%d)\n", G.prop.gcnArchName, G.arch_major, G.arch_minor);
    fprintf(stderr, "compute : %d WGP (%d CU), wave %d, %.2f GHz\n",
            G.prop.multiProcessorCount, G.prop.multiProcessorCount * 2,
            G.prop.warpSize, G.prop.clockRate / 1e6);
    // GDDR6 moves 16 bits per pin per command clock, not 2 like DDR. Using 2 understated
    // this card by 8x (48 GB/s against the real 384).
    {
        double gbps = G.prop.memoryClockRate * 1e3 * 16.0 / 1e9;
        double bw = gbps * (G.prop.memoryBusWidth / 8.0);
        fprintf(stderr, "memory  : %.2f GB, %d-bit, %.0f GB/s nominal (GDDR6 %.0f Gbps)\n",
                G.prop.totalGlobalMem / 1073741824.0, G.prop.memoryBusWidth, bw, gbps);
    }
    fprintf(stderr, "LDS     : %zu KB/workgroup, %d banks\n",
            G.prop.sharedMemPerBlock / 1024, G.prop.sharedMemPerMultiprocessor > 0 ? 64 : 64);
    if (ktrace::enabled()) ktrace::init(G.prop);
}

// ----------------------------------------------------------------------------- GGUF
enum GgufType {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19, T_IQ4_NL = 20,
    T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23, T_BF16 = 30,
    T_PQ2_0 = 142, T_PTQ1_0 = 143
};
static const char* typename_of(uint32_t t) {
    switch (t) {
        case T_F32: return "F32";     case T_F16: return "F16";
        case T_Q4_0: return "Q4_0";   case T_Q4_1: return "Q4_1";
        case T_Q5_0: return "Q5_0";   case T_Q5_1: return "Q5_1";
        case T_Q8_0: return "Q8_0";   case T_Q2_K: return "Q2_K";
        case T_Q3_K: return "Q3_K";   case T_Q4_K: return "Q4_K";
        case T_Q5_K: return "Q5_K";   case T_Q6_K: return "Q6_K";
        case T_IQ2_XXS: return "IQ2_XXS"; case T_IQ2_XS: return "IQ2_XS";
        case T_IQ3_XXS: return "IQ3_XXS"; case T_IQ1_S: return "IQ1_S";
        case T_IQ4_NL: return "IQ4_NL";   case T_IQ3_S: return "IQ3_S";
        case T_IQ2_S: return "IQ2_S";     case T_IQ4_XS: return "IQ4_XS";
        case T_BF16: return "BF16";
        case T_PQ2_0: return "PQ2_0";     case T_PTQ1_0: return "PTQ1_0";
        default: return "?";
    }
}

struct GTensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
};

struct Gguf {
    const uint8_t* base = nullptr;
    size_t size = 0;
    size_t data_start = 0;
    std::vector<GTensor> tensors;
    std::map<std::string, std::string> kv_str;
    std::map<std::string, uint64_t> kv_u64;
    std::map<std::string, double> kv_f64;
};

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <class T> T rd() {
        T v{};
        if (p + sizeof(T) > end) { ok = false; return v; }
        memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
    }
    std::string str() {
        uint64_t n = rd<uint64_t>();
        if (!ok || p + n > end) { ok = false; return {}; }
        std::string s((const char*)p, (size_t)n);
        p += n;
        return s;
    }
};

static void skip_value(Reader& r, uint32_t t);
static std::string read_value_str(Reader& r, uint32_t t) {
    // Only the scalar and string kinds are interesting; arrays are skipped.
    char buf[128];
    switch (t) {
        case 0: snprintf(buf, sizeof buf, "%u", r.rd<uint8_t>()); break;
        case 1: snprintf(buf, sizeof buf, "%d", r.rd<int8_t>()); break;
        case 2: snprintf(buf, sizeof buf, "%u", r.rd<uint16_t>()); break;
        case 3: snprintf(buf, sizeof buf, "%d", r.rd<int16_t>()); break;
        case 4: snprintf(buf, sizeof buf, "%u", r.rd<uint32_t>()); break;
        case 5: snprintf(buf, sizeof buf, "%d", r.rd<int32_t>()); break;
        case 6: snprintf(buf, sizeof buf, "%g", (double)r.rd<float>()); break;
        case 7: snprintf(buf, sizeof buf, "%u", r.rd<uint8_t>()); break;
        case 8: return r.str();
        case 10: snprintf(buf, sizeof buf, "%llu", (unsigned long long)r.rd<uint64_t>()); break;
        case 11: snprintf(buf, sizeof buf, "%lld", (long long)r.rd<int64_t>()); break;
        case 12: snprintf(buf, sizeof buf, "%g", r.rd<double>()); break;
        default: skip_value(r, t); return {};
    }
    return buf;
}
static void skip_value(Reader& r, uint32_t t) {
    if (t == 9) {                       // array
        uint32_t et = r.rd<uint32_t>();
        uint64_t n = r.rd<uint64_t>();
        for (uint64_t i = 0; i < n && r.ok; ++i) skip_value(r, et);
        return;
    }
    read_value_str(r, t);
}

static bool load_gguf(const char* path, Gguf& g) {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot open %s\n", path); return false; }
    LARGE_INTEGER fs; GetFileSizeEx(h, &fs);
    HANDLE hm = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
    g.base = (const uint8_t*)MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    g.size = (size_t)fs.QuadPart;
    if (!g.base) { fprintf(stderr, "map failed\n"); return false; }

    Reader r{g.base, g.base + g.size};
    uint32_t magic = r.rd<uint32_t>();
    if (magic != 0x46554747) { fprintf(stderr, "not a GGUF (magic %08x)\n", magic); return false; }
    uint32_t ver = r.rd<uint32_t>();
    uint64_t ntens = r.rd<uint64_t>();
    uint64_t nkv = r.rd<uint64_t>();

    for (uint64_t i = 0; i < nkv && r.ok; ++i) {
        std::string k = r.str();
        uint32_t t = r.rd<uint32_t>();
        if (t == 8) {
            g.kv_str[k] = r.str();
        } else if (t == 4) {
            g.kv_u64[k] = r.rd<uint32_t>();
        } else if (t == 10) {
            g.kv_u64[k] = r.rd<uint64_t>();
        } else if (t == 6) {
            g.kv_f64[k] = r.rd<float>();
        } else if (t == 12) {
            g.kv_f64[k] = r.rd<double>();
        } else if (t == 9) {
            uint32_t et = r.rd<uint32_t>();
            uint64_t n = r.rd<uint64_t>();
            if ((et == 4 || et == 5) && n && n < 4096) {
                uint64_t sum = 0;
                for (uint64_t j = 0; j < n; ++j) sum += r.rd<uint32_t>();
                g.kv_u64[k + ".count"] = n;
                g.kv_u64[k + ".sum"] = sum;
            } else {
                for (uint64_t j = 0; j < n && r.ok; ++j) skip_value(r, et);
            }
        } else {
            skip_value(r, t);
        }
    }

    for (uint64_t i = 0; i < ntens && r.ok; ++i) {
        GTensor t;
        t.name = r.str();
        uint32_t nd = r.rd<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(r.rd<uint64_t>());
        t.type = r.rd<uint32_t>();
        t.offset = r.rd<uint64_t>();
        g.tensors.push_back(std::move(t));
    }
    if (!r.ok) { fprintf(stderr, "GGUF parse failed\n"); return false; }

    // data section is aligned to 32
    size_t pos = (size_t)(r.p - g.base);
    g.data_start = (pos + 31) & ~(size_t)31;
    fprintf(stderr, "gguf    : v%u, %zu tensors, %zu kv, data at %zu\n",
            ver, g.tensors.size(), g.kv_str.size() + g.kv_u64.size(), g.data_start);
    return true;
}

static const GTensor* find_tensor(const Gguf& g, const std::string& n) {
    for (auto& t : g.tensors) if (t.name == n) return &t;
    return nullptr;
}
static std::pair<int, int> rc_of(const GTensor& t) {   // rows, cols
    uint64_t c = t.dims.empty() ? 0 : t.dims[0];
    uint64_t r = 1;
    for (size_t i = 1; i < t.dims.size(); ++i) r *= t.dims[i];
    return {(int)r, (int)c};
}

__device__ inline float fp16_to_f32(uint bits) {
    uint sign = (bits >> 15) & 1u, exp = (bits >> 10) & 0x1Fu, mant = bits & 0x3FFu;
    if (exp == 0u) { float v = (float)mant / 1024.0f * (1.0f / 16384.0f); return sign ? -v : v; }
    if (exp == 31u) return sign ? -3.0e38f : 3.0e38f;
    float v = (1.0f + (float)mant / 1024.0f) * exp2f((float)((int)exp - 15));
    return sign ? -v : v;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, f;
    if (exp == 0) { f = sign << 31; }
    else if (exp == 31) { f = (sign << 31) | 0x7F800000u | (mant << 13); }
    else { f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13); }
    float o; memcpy(&o, &f, 4); return o;
}

// ----------------------------------------------------------------------------- kernels
// PTQ1_0 block: 128 weights in 28 bytes. bytes 0..23 are trit bytes, 24..25 are the
// high part of the 5th digit group, 26..27 are the fp16 block scale.
//   trit i is carried by byte j = (i<80 ? i&15 : i<120 ? 16+((i-80)&7) : 24+((i-120)&1))
//   at digit n = (i<80 ? i>>4 : i<120 ? (i-80)>>3 : (i-120)>>1)
//   digit: q = (raw * 3^n) & 0xFF ; trit = (q * 3) >> 8
//
// The trit table is floats in LDS, so the inner loop is one LDS load and one FMA per weight.
#define PF_RPW 4

__global__ void gemv_ptq1(const uint* __restrict__ w,
                          const float* __restrict__ x,
                          float* __restrict__ y,
                          int R, int C)
{
    // byte -> five trit floats, built once per workgroup (256 threads, one byte each)
    __shared__ float tl[1280];
    const uint P3[5] = {1u, 3u, 9u, 27u, 81u};
    {
        uint b = threadIdx.x;
        for (int n = 0; n < 5; ++n) {
            uint tr = (((b * P3[n]) & 0xFFu) * 3u) >> 8;
            tl[b * 5u + n] = (float)((int)tr - 1);
        }
    }
    __syncthreads();

    int ws = warpSize;
    int nsub = blockDim.x / ws;
    int sub = threadIdx.x / ws;
    int lane = threadIdx.x % ws;

    int row0 = (blockIdx.x * nsub + sub) * PF_RPW;
    int ra = row0, rb = row0 + 1, rc = row0 + 2, rd = row0 + 3;
    // Clamp, never return: __syncthreads() must be reached by every thread in the block.
    bool ka = ra < R, kb = rb < R, kc = rc < R, kd = rd < R;
    if (!ka) ra = 0;
    if (!kb) rb = 0;
    if (!kc) rc = 0;
    if (!kd) rd = 0;

    int bpr = C >> 7;          // 128-weight blocks per row
    int ntb = bpr * 26;        // trit-bearing bytes per row (26 of the 28)

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int bi = lane; bi < ntb; bi += ws) {
        int blk = bi / 26;
        int off = bi - blk * 26;
        int base = blk << 7;

        // one byte per row, and the scale is the high half of word (base/4 + 6) because a
        // block starts at a multiple of 28 and 28 % 4 == 0
        int w0 = (ra * bpr + blk) * 28 + off;
        int w1 = (rb * bpr + blk) * 28 + off;
        int w2 = (rc * bpr + blk) * 28 + off;
        int w3 = (rd * bpr + blk) * 28 + off;
        uint r0 = (w[w0 >> 2] >> (8u * (uint)(w0 & 3))) & 0xFFu;
        uint r1 = (w[w1 >> 2] >> (8u * (uint)(w1 & 3))) & 0xFFu;
        uint r2 = (w[w2 >> 2] >> (8u * (uint)(w2 & 3))) & 0xFFu;
        uint r3 = (w[w3 >> 2] >> (8u * (uint)(w3 & 3))) & 0xFFu;
        int s0 = ((ra * bpr + blk) * 7) + 6;
        int s1 = ((rb * bpr + blk) * 7) + 6;
        int s2 = ((rc * bpr + blk) * 7) + 6;
        int s3 = ((rd * bpr + blk) * 7) + 6;
        float c0 = fp16_to_f32(w[s0] >> 16);
        float c1 = fp16_to_f32(w[s1] >> 16);
        float c2 = fp16_to_f32(w[s2] >> 16);
        float c3 = fp16_to_f32(w[s3] >> 16);

        int b50 = (int)r0 * 5, b51 = (int)r1 * 5, b52 = (int)r2 * 5, b53 = (int)r3 * 5;
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;

        if (off < 16) {                       // five digits, stride 16
            int xi = base + off;
            #pragma unroll
            for (int n = 0; n < 5; ++n) {
                float xv = x[xi + (n << 4)];
                p0 += tl[b50 + n] * xv;
                p1 += tl[b51 + n] * xv;
                p2 += tl[b52 + n] * xv;
                p3 += tl[b53 + n] * xv;
            }
        } else if (off < 24) {                // five digits, stride 8, from trit 80
            int xi = base + 80 + (off - 16);
            #pragma unroll
            for (int n = 0; n < 5; ++n) {
                float xv = x[xi + (n << 3)];
                p0 += tl[b50 + n] * xv;
                p1 += tl[b51 + n] * xv;
                p2 += tl[b52 + n] * xv;
                p3 += tl[b53 + n] * xv;
            }
        } else {                              // four digits, from trit 120
            int xi = base + 120 + (off - 24);
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                float xv = x[xi + (n << 1)];
                p0 += tl[b50 + n] * xv;
                p1 += tl[b51 + n] * xv;
                p2 += tl[b52 + n] * xv;
                p3 += tl[b53 + n] * xv;
            }
        }
        a0 += c0 * p0; a1 += c1 * p1; a2 += c2 * p2; a3 += c3 * p3;
    }

    // warp reduction, then one store per row
    #pragma unroll
    for (int o = ws / 2; o > 0; o >>= 1) {
        a0 += __shfl_down(a0, o);
        a1 += __shfl_down(a1, o);
        a2 += __shfl_down(a2, o);
        a3 += __shfl_down(a3, o);
    }
    if (lane == 0) {
        if (ka) y[ra] = a0;
        if (kb) y[rb] = a1;
        if (kc) y[rc] = a2;
        if (kd) y[rd] = a3;
    }
}

// host reference, straight from the raw GGUF bytes. Deliberately a separate implementation
// of the same layout: if the device kernel and this agree, the byte-to-trit mapping is right.
static void ref_ptq1(const uint8_t* rowdata, const float* x, float* y, int C) {
    int bpr = C >> 7;
    double acc = 0.0;
    for (int blk = 0; blk < bpr; ++blk) {
        const uint8_t* bp = rowdata + (size_t)blk * 28;
        uint16_t hb; memcpy(&hb, bp + 26, 2);
        float sc = half_to_float(hb);
        double blk_acc = 0.0;
        for (int i = 0; i < 128; ++i) {
            int j, n;
            if (i < 80)        { j = i & 15;          n = i >> 4; }
            else if (i < 120)  { j = 16 + ((i - 80) & 7); n = (i - 80) >> 3; }
            else               { j = 24 + ((i - 120) & 1); n = (i - 120) >> 1; }
            uint32_t p = 1; for (int t = 0; t < n; ++t) p *= 3u;
            uint32_t q = ((uint32_t)bp[j] * p) & 0xFFu;
            int trit = (int)((q * 3u) >> 8);
            blk_acc += (double)(trit - 1) * (double)x[blk * 128 + i];
        }
        acc += blk_acc * (double)sc;
    }
    y[0] = (float)acc;
}

// ----------------------------------------------------------------------------- main
int main(int argc, char** argv) {
    const char* model = nullptr;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--selftest")) selftest = true;
        else if (ktrace::parse_arg(argc, argv, i)) {}
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!model) {
        fprintf(stderr, "usage: --model <path.gguf> [--selftest] [--trace] [--trace-json FILE]\n");
        return 1;
    }

    ktrace::clock_init();
    initdev();

    Gguf g;
    if (!load_gguf(model, g)) return 1;

    // report what we can and cannot handle yet
    std::map<std::string, int> hist;
    for (auto& t : g.tensors) hist[typename_of(t.type)]++;
    fprintf(stderr, "formats :");
    for (auto& kv : hist) fprintf(stderr, " %s x%d", kv.first.c_str(), kv.second);
    fprintf(stderr, "\n");

    // ---- selftest: the PTQ1_0 GEMV against a host reference from the raw GGUF bytes
    if (selftest) {
        const GTensor* t = nullptr;
        for (auto& c : g.tensors)
            if (c.type == T_PTQ1_0 && c.name.find("ssm_out") != std::string::npos) { t = &c; break; }
        if (!t) {
            for (auto& c : g.tensors) if (c.type == T_PTQ1_0) { t = &c; break; }
        }
        if (!t) {
            fprintf(stderr, "selftest: no PTQ1_0 tensor in this model, nothing to check\n");
            return 0;
        }
        auto [R, C] = rc_of(*t);
        fprintf(stderr, "selftest PTQ1_0 on %s  R=%d C=%d\n", t->name.c_str(), R, C);
        if (C % 128) { fprintf(stderr, "selftest: C not a multiple of 128\n"); return 1; }

        const uint8_t* src = g.base + g.data_start + t->offset;
        int bpr = C >> 7;
        size_t nbytes = (size_t)R * bpr * 28;

        Buf wb = mkbuf(nbytes);
        upload(&wb, src, nbytes);

        std::vector<float> xv((size_t)C);
        for (int i = 0; i < C; ++i)
            xv[i] = std::sin(0.01f * (float)i) * 0.5f + 0.1f * std::cos(0.003f * (float)i);
        Buf xb = mkbuf((size_t)C * 4);
        upload(&xb, xv.data(), (size_t)C * 4);
        Buf yb = mkbuf((size_t)R * 4);

        int ws = G.prop.warpSize, nsub = 256 / ws;
        int per = nsub * PF_RPW;
        {
            ktrace::Launch L = ktrace::begin("gemv_ptq1", (R + per - 1) / per, 1, 1, 256, nbytes);
            gemv_ptq1<<<(R + per - 1) / per, 256>>>((const uint*)wb.p, (const float*)xb.p,
                                                    (float*)yb.p, R, C);
            ktrace::end(L);
        }
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) {
            fprintf(stderr, "selftest: kernel failed: %s\n", hipGetErrorString(e));
            return 1;
        }
        std::vector<float> yv(R);
        download(yv.data(), yb, (size_t)R * 4);

        // compare sampled rows against the host reference
        double worst = 0, nrm = 0;
        int nsamp = 0;
        for (int r = 0; r < R; r += 97) {
            float ref = 0;
            ref_ptq1(src + (size_t)r * bpr * 28, xv.data(), &ref, C);
            double d = std::fabs((double)ref - (double)yv[r]);
            if (d > worst) worst = d;
            nrm += std::fabs((double)ref);
            ++nsamp;
        }
        double mean = nrm / (nsamp ? nsamp : 1);
        fprintf(stderr, "selftest PTQ1_0: sampled %d rows, max abs err vs host = %.4e (mean |y| %.4e)\n",
                nsamp, worst, mean);
        bool ok = worst < 1e-3 * (mean > 0 ? mean : 1.0);
        fprintf(stderr, "selftest PTQ1_0 vs host: %s\n", ok ? "PASS" : "FAIL");

        // ---- rocblas oracle -------------------------------------------------------
        // The second, independent check: unpack this tensor to fp32 and run the same GEMV
        // through rocblas. A heavily tested library computing the same arithmetic is a
        // stronger check than any single reference, and it is the class of bug that has
        // already bitten this project twice (a wrong packing order that produced plausible
        // numbers and only failed against a reference).
        //
        // Layout note: GGUF stores A[row*C + col], so as column-major it is m=C, n=R with
        // lda=C, and TRANSPOSE gives y[row] = sum_col A[col + row*C] * x[col] - exactly the
        // operation under test.
        bool okb = true;
        {
            std::vector<float> af((size_t)R * C);
            for (int r = 0; r < R; ++r) {
                const uint8_t* rp = src + (size_t)r * bpr * 28;
                for (int blk = 0; blk < bpr; ++blk) {
                    const uint8_t* bp = rp + (size_t)blk * 28;
                    uint16_t hb; memcpy(&hb, bp + 26, 2);
                    float sc = half_to_float(hb);
                    for (int i = 0; i < 128; ++i) {
                        int j, nn;
                        if (i < 80)       { j = i & 15;              nn = i >> 4; }
                        else if (i < 120) { j = 16 + ((i - 80) & 7); nn = (i - 80) >> 3; }
                        else              { j = 24 + ((i - 120) & 1); nn = (i - 120) >> 1; }
                        uint32_t pw = 1; for (int t = 0; t < nn; ++t) pw *= 3u;
                        uint32_t q = ((uint32_t)bp[j] * pw) & 0xFFu;
                        int trit = (int)((q * 3u) >> 8);
                        af[(size_t)r * C + blk * 128 + i] = (float)(trit - 1) * sc;
                    }
                }
            }
            Buf ab = mkbuf((size_t)R * C * 4);
            upload(&ab, af.data(), (size_t)R * C * 4);
            Buf xb2 = mkbuf((size_t)C * 4);
            upload(&xb2, xv.data(), (size_t)C * 4);
            Buf yb2 = mkbuf((size_t)R * 4);

            rocblas_handle h = nullptr;
            rocblas_status st = rocblas_create_handle(&h);
            if (st != rocblas_status_success) {
                fprintf(stderr, "selftest rocblas: handle failed (%d), oracle skipped\n", (int)st);
            } else {
                // sgemv is unimplemented for gfx1031 in this build, so use sgemm with n=1.
                // y[R] = A[R,C] * x[C]:  op(A) is R x C, so store A as C x R column-major
                // with lda=C (which is exactly the row-major R x C we already have) and ask
                // for the transpose.
                const float alpha = 1.0f, beta = 0.0f;
                ktrace::Launch L = ktrace::begin("rocblas_sgemm_n1", 1, 1, 1, 1,
                                                 (size_t)R * C * 4 + (size_t)C * 4);
                st = rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none,
                                   R, 1, C, &alpha, (const float*)ab.p, C,
                                   (const float*)xb2.p, C, &beta, (float*)yb2.p, R);
                ktrace::end(L);
                hipDeviceSynchronize();
                if (st != rocblas_status_success) {
                    fprintf(stderr,
                        "selftest rocblas: sgemm failed (%d). The library or its Tensile files\n"
                        "  are not resolvable; set ROCBLAS_TENSILE_LIBPATH to\n"
                        "  C:\\Rocm10\\bin\\rocblas\\library. Oracle skipped, not failed.\n", (int)st);
                } else {
                    std::vector<float> yr(R);
                    download(yr.data(), yb2, (size_t)R * 4);
                    double wr = 0, nr = 0;
                    for (int r = 0; r < R; r += 97) {
                        double d = std::fabs((double)yr[r] - (double)yv[r]);
                        if (d > wr) wr = d;
                        nr += std::fabs((double)yr[r]);
                    }
                    double mr = nr / (nsamp ? nsamp : 1);
                    fprintf(stderr, "selftest PTQ1_0 vs rocblas: max abs err = %.4e (mean |y| %.4e)\n",
                            wr, mr);
                    // fp32 summation order differs between rocblas and our kernel, so this
                    // is a relative tolerance, not bit-identity.
                    okb = wr < 1e-4 * (mr > 0 ? mr : 1.0);
                    fprintf(stderr, "selftest PTQ1_0 vs rocblas: %s\n", okb ? "PASS" : "FAIL");
                }
                rocblas_destroy_handle(h);
            }
            freebuf(ab); freebuf(xb2); freebuf(yb2);
        }
        ok = ok && okb;

        freebuf(wb); freebuf(xb); freebuf(yb);
        ktrace::dump();
        ktrace::shutdown();
        return ok ? 0 : 1;
    }

    fprintf(stderr, "\nthis build is milestone 1: GGUF read, device init, tracing and the\n"
                    "PTQ1_0 GEMV. The full decode graph lands in milestone 2 and 3.\n"
                    "run with --selftest to verify the kernel.\n");
    ktrace::dump();
    ktrace::shutdown();
    return 0;
}
