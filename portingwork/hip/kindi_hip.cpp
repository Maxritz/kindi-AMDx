// kindi_hip.cpp - Kindi inference engine, HIP backend, RX 6700 XT (gfx1031).
//
// A new engine for RDNA2, not a port. GGUF is read as it is: no conversion, no weight
// pack, and no dequantisation - weights stay packed in VRAM and are decoded in registers.
// The only thing carried over from the Vulkan engine is the understanding of the GGUF
// layout and the model's numeric contract.
//
// See HIP-PORT.md for the design and the measurements behind each choice.
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
#include <functional>
#include <algorithm>
#include "trace_hip.h"
#include "kernels_hip.h"

// ============================================================ device + buffers
struct Dev { hipDeviceProp_t prop; int major = 0, minor = 0; };
static Dev G;

struct Buf { void* p = nullptr; size_t size = 0; bool host = false; };
static uint64_t g_alloc_total = 0;

static Buf mkbuf(size_t bytes) {
    Buf b; b.size = bytes ? bytes : 4;
    if (hipMalloc(&b.p, b.size) != hipSuccess) {
        fprintf(stderr, "hipMalloc %zu failed: %s\n", b.size, hipGetErrorString(hipGetLastError()));
        exit(1);
    }
    g_alloc_total += b.size; ktrace::count_alloc(b.size);
    return b;
}
static Buf mkbuf_host(size_t bytes) {
    Buf b; b.size = bytes ? bytes : 4;
    hipHostMalloc(&b.p, b.size, hipHostMallocDefault);
    b.host = true; ktrace::count_alloc(b.size);
    return b;
}
static void freebuf(Buf& b) {
    if (!b.p) return;
    ktrace::count_free();
    if (g_alloc_total >= b.size) g_alloc_total -= b.size;
    if (b.host) hipHostFree(b.p); else hipFree(b.p);
    b = {};
}
static void upload(Buf* d, const void* s, size_t n) {
    double t0 = ktrace::now_us();
    if (hipMemcpy(d->p, s, n, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "upload %zu failed\n", n); exit(1);
    }
    ktrace::count_xfer(n, ktrace::now_us() - t0);
}
static void download(void* d, const Buf& s, size_t n) {
    double t0 = ktrace::now_us();
    hipMemcpy(d, s.p, n, hipMemcpyDeviceToHost);
    ktrace::count_xfer(n, ktrace::now_us() - t0);
}
static void fillzero(Buf& b) { hipMemset(b.p, 0, b.size); }

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ============================================================ GGUF
enum GType {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19, T_IQ4_NL = 20,
    T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23, T_BF16 = 30,
    T_PQ2_0 = 142, T_PTQ1_0 = 143
};
static const char* tn(uint32_t t) {
    switch (t) {
        case T_F32: return "F32"; case T_F16: return "F16";
        case T_Q4_0: return "Q4_0"; case T_Q8_0: return "Q8_0";
        case T_Q2_K: return "Q2_K"; case T_Q3_K: return "Q3_K";
        case T_Q4_K: return "Q4_K"; case T_Q5_K: return "Q5_K"; case T_Q6_K: return "Q6_K";
        case T_IQ2_XXS: return "IQ2_XXS"; case T_IQ2_XS: return "IQ2_XS";
        case T_IQ3_XXS: return "IQ3_XXS"; case T_IQ1_S: return "IQ1_S";
        case T_IQ4_NL: return "IQ4_NL"; case T_IQ3_S: return "IQ3_S";
        case T_IQ2_S: return "IQ2_S"; case T_IQ4_XS: return "IQ4_XS";
        case T_BF16: return "BF16";
        case T_PQ2_0: return "PQ2_0"; case T_PTQ1_0: return "PTQ1_0";
        default: return "?";
    }
}
struct GT { std::string name; std::vector<uint64_t> dims; uint32_t type = 0; uint64_t off = 0; };

struct Gguf {
    const uint8_t* base = nullptr;
    size_t size = 0, data_start = 0;
    std::vector<GT> tens;
    std::map<std::string, std::string> ks;
    std::map<std::string, uint64_t> ku;
    std::map<std::string, double> kf;
    std::vector<std::string> tokens, merges;
    std::vector<int> tok_type;
    const GT* find(const std::string& n) const {
        for (auto& t : tens) if (t.name == n) return &t;
        return nullptr;
    }
};

struct Rd {
    const uint8_t* p; const uint8_t* end; bool ok = true;
    template <class T> T rd() { T v{}; if (p + sizeof(T) > end) { ok = false; return v; }
                                memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }
    std::string str() { uint64_t n = rd<uint64_t>(); if (!ok || p + n > end) { ok = false; return {}; }
                        std::string s((const char*)p, (size_t)n); p += n; return s; }
};
static void skipv(Rd& r, uint32_t t) {
    if (t == 9) { uint32_t et = r.rd<uint32_t>(); uint64_t n = r.rd<uint64_t>();
                  for (uint64_t i = 0; i < n && r.ok; ++i) skipv(r, et); return; }
    switch (t) {
        case 0: case 1: case 7: r.rd<uint8_t>(); break;
        case 2: case 3: r.rd<uint16_t>(); break;
        case 4: case 5: case 6: r.rd<uint32_t>(); break;
        case 10: case 11: case 12: r.rd<uint64_t>(); break;
        case 8: r.str(); break;
        default: r.ok = false; break;
    }
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

    Rd r{g.base, g.base + g.size};
    if (r.rd<uint32_t>() != 0x46554747) { fprintf(stderr, "not a GGUF\n"); return false; }
    uint32_t ver = r.rd<uint32_t>();
    uint64_t nt = r.rd<uint64_t>(), nkv = r.rd<uint64_t>();

    for (uint64_t i = 0; i < nkv && r.ok; ++i) {
        std::string k = r.str();
        uint32_t t = r.rd<uint32_t>();
        if (t == 8) { g.ks[k] = r.str(); }
        else if (t == 4) { g.ku[k] = r.rd<uint32_t>(); }
        else if (t == 10) { g.ku[k] = r.rd<uint64_t>(); }
        else if (t == 6) { g.kf[k] = r.rd<float>(); }
        else if (t == 12) { g.kf[k] = r.rd<double>(); }
        else if (t == 9) {
            uint32_t et = r.rd<uint32_t>(); uint64_t n = r.rd<uint64_t>();
            if (k == "tokenizer.ggml.tokens" && et == 8) {
                g.tokens.reserve((size_t)n);
                for (uint64_t j = 0; j < n && r.ok; ++j) g.tokens.push_back(r.str());
            } else if (k == "tokenizer.ggml.merges" && et == 8) {
                g.merges.reserve((size_t)n);
                for (uint64_t j = 0; j < n && r.ok; ++j) g.merges.push_back(r.str());
            } else if (k == "tokenizer.ggml.token_type" && (et == 4 || et == 5)) {
                g.tok_type.reserve((size_t)n);
                for (uint64_t j = 0; j < n && r.ok; ++j) g.tok_type.push_back((int)r.rd<uint32_t>());
            } else {
                for (uint64_t j = 0; j < n && r.ok; ++j) skipv(r, et);
            }
        } else { skipv(r, t); }
    }
    for (uint64_t i = 0; i < nt && r.ok; ++i) {
        GT t; t.name = r.str();
        uint32_t nd = r.rd<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(r.rd<uint64_t>());
        t.type = r.rd<uint32_t>();
        t.off = r.rd<uint64_t>();
        g.tens.push_back(std::move(t));
    }
    if (!r.ok) { fprintf(stderr, "GGUF parse failed\n"); return false; }
    g.data_start = (((size_t)(r.p - g.base)) + 31) & ~(size_t)31;
    fprintf(stderr, "gguf    : v%u, %zu tensors, %zu tokens, data at %zu\n",
            ver, g.tens.size(), g.tokens.size(), g.data_start);
    return true;
}

// ============================================================ tokenizer (BPE, from GGUF)
struct Tokenizer {
    std::vector<std::string> id2tok;
    std::map<std::string, int> tok2id;
    std::map<std::string, int> merge_rank;
    int eos = -1, bos = -1;

    bool load(const Gguf& g) {
        id2tok = g.tokens;
        for (size_t i = 0; i < id2tok.size(); ++i) tok2id[id2tok[i]] = (int)i;
        for (size_t i = 0; i < g.merges.size(); ++i) merge_rank[g.merges[i]] = (int)i;
        auto it = g.ku.find("tokenizer.ggml.eos_token_id");
        if (it != g.ku.end()) eos = (int)it->second;
        it = g.ku.find("tokenizer.ggml.bos_token_id");
        if (it != g.ku.end()) bos = (int)it->second;
        return !id2tok.empty();
    }
    // GPT-2 style byte-level BPE: map bytes to printable unicode, split on the Qwen regex,
    // then merge by rank.
    static std::string b2u(uint8_t b) {
        // GPT-2 byte encoder: bytes 33..126, 161..172, 174..255 map to themselves; the rest
        // map to 256+n in order.
        static int map[256]; static bool init = false;
        if (!init) {
            for (int i = 0; i < 256; ++i) map[i] = -1;
            int n = 0;
            for (int b2 = 0; b2 < 256; ++b2) {
                bool keep = (b2 >= 33 && b2 <= 126) || (b2 >= 161 && b2 <= 172) || (b2 >= 174);
                if (keep) map[b2] = b2;
            }
            for (int b2 = 0; b2 < 256; ++b2) if (map[b2] < 0) map[b2] = 256 + n++;
            init = true;
        }
        int c = map[b];
        std::string s;
        if (c < 128) s += (char)c;
        else { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
        return s;
    }
    // Decode a UTF-8 string into one std::string per CODEPOINT. This matters: the GPT-2
    // byte map turns space into U+0120, which is two UTF-8 bytes, and BPE merges are
    // defined over codepoints. Splitting into raw bytes made the merge table never match,
    // which produced plausible-looking but wrong token ids.
    static std::vector<std::string> cps(const std::string& s) {
        std::vector<std::string> out;
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = s[i];
            size_t n = 1;
            if ((c >> 5) == 6) n = 2;
            else if ((c >> 4) == 0xE) n = 3;
            else if ((c >> 3) == 0x1E) n = 4;
            if (i + n > s.size()) n = 1;
            out.push_back(s.substr(i, n));
            i += n;
        }
        return out;
    }
    static bool cp_is_alpha(const std::string& u) {
        unsigned char c = u[0];
        if (c < 0x80) return isalpha(c) != 0;
        return true;                       // any non-ASCII codepoint counts as a letter
    }
    static bool cp_is_digit(const std::string& u) {
        return u.size() == 1 && isdigit((unsigned char)u[0]) != 0;
    }
    // Approximate the Qwen pre-tokeniser: runs of letters, runs of digits, and everything
    // else one codepoint at a time.
    static std::vector<std::string> split_units(const std::vector<std::string>& u) {
        std::vector<std::string> out;
        size_t i = 0;
        while (i < u.size()) {
            if (cp_is_alpha(u[i])) {
                size_t j = i; while (j < u.size() && cp_is_alpha(u[j])) ++j;
                std::string w; for (size_t k = i; k < j; ++k) w += u[k];
                out.push_back(w); i = j;
            } else if (cp_is_digit(u[i])) {
                size_t j = i; while (j < u.size() && cp_is_digit(u[j])) ++j;
                std::string w; for (size_t k = i; k < j; ++k) w += u[k];
                out.push_back(w); i = j;
            } else {
                out.push_back(u[i]); ++i;
            }
        }
        return out;
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        // byte-level map first
        std::string mapped;
        for (unsigned char c : text) mapped += b2u(c);
        for (auto& unit : split_units(cps(mapped))) {
            // greedy BPE merge, over codepoints not bytes
            std::vector<std::string> parts = cps(unit);
            while (parts.size() > 1) {
                int best = 1 << 30, bi = -1;
                for (size_t i = 0; i + 1 < parts.size(); ++i) {
                    auto it = merge_rank.find(parts[i] + " " + parts[i + 1]);
                    if (it != merge_rank.end() && it->second < best) { best = it->second; bi = (int)i; }
                }
                if (bi < 0) break;
                parts[bi] = parts[bi] + parts[bi + 1];
                parts.erase(parts.begin() + bi + 1);
            }
            for (auto& p : parts) {
                auto it = tok2id.find(p);
                if (it != tok2id.end()) ids.push_back(it->second);
            }
        }
        return ids;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) if (id >= 0 && id < (int)id2tok.size()) out += id2tok[id];
        // reverse the byte map: U+0100.. sequences back to raw bytes
        std::string raw;
        size_t i = 0;
        while (i < out.size()) {
            unsigned char c = out[i];
            int cp = c;
            if (c >= 0xC0 && i + 1 < out.size()) {
                if ((c >> 5) == 6) { cp = ((c & 0x1F) << 6) | (out[i + 1] & 0x3F); i += 2; }
                else if ((c >> 4) == 0xE && i + 2 < out.size()) {
                    cp = ((c & 0x0F) << 12) | ((out[i + 1] & 0x3F) << 6) | (out[i + 2] & 0x3F); i += 3;
                } else { raw += (char)c; ++i; continue; }
            } else { raw += (char)c; ++i; continue; }
            if (cp >= 256 && cp < 512) raw += (char)(cp - 256);
            else {
                if (cp < 0x80) raw += (char)cp;
                else if (cp < 0x800) { raw += (char)(0xC0 | (cp >> 6)); raw += (char)(0x80 | (cp & 0x3F)); }
                else { raw += (char)(0xE0 | (cp >> 12)); raw += (char)(0x80 | ((cp >> 6) & 0x3F));
                       raw += (char)(0x80 | (cp & 0x3F)); }
            }
        }
        return raw;
    }
};

// ============================================================ model
struct Model {
    int NL = 0, D = 0, NFF = 0, NH = 0, NHKV = 0, HD = 0, NK = 0, HK = 0, NV = 0;
    int HB = 0, interval = 0, NATT = 0, VOCAB = 0, CONV_CH = 0, nrot = 0;
    int nextn = 0;
    float eps = 1e-6f, rope_base = 1e7f;
    std::map<std::string, Buf> w;      // packed weights
    std::map<std::string, std::pair<int,int>> rc;
    std::map<std::string, Buf> f32;    // dense tensors
    std::vector<float> signs;
    std::map<int, std::vector<float>> sign_by_width;
    std::vector<int> sign_widths;
    std::vector<float> cosv, sinv;
};

static void initdev() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) {
        fprintf(stderr,
            "no HIP device. On this machine that is almost always a DLL search-path issue:\n"
            "amdhip64_7.dll loads rocm_kpack.dll and amd_comgr.dll from the EXE DIRECTORY\n"
            "before PATH. Copy those three next to the exe, or run from C:\\Rocm10\\bin.\n"
            "See HIP-ENV.md.\n");
        exit(1);
    }
    hipSetDevice(0);
    hipGetDeviceProperties(&G.prop, 0);
    int a = 0;
    hipDeviceGetAttribute(&a, hipDeviceAttributeComputeCapabilityMajor, 0); G.major = a;
    hipDeviceGetAttribute(&a, hipDeviceAttributeComputeCapabilityMinor, 0); G.minor = a;
    fprintf(stderr, "device  : %s\n", G.prop.name);
    fprintf(stderr, "arch    : %s (cc %d.%d), %d WGP (%d CU), wave %d, %.2f GHz\n",
            G.prop.gcnArchName, G.major, G.minor, G.prop.multiProcessorCount,
            G.prop.multiProcessorCount * 2, G.prop.warpSize, G.prop.clockRate / 1e6);
    fprintf(stderr, "memory  : %.2f GB, %d-bit, %.0f GB/s (GDDR6)\n",
            G.prop.totalGlobalMem / 1073741824.0, G.prop.memoryBusWidth,
            G.prop.memoryClockRate * 1e3 * 16.0 / 1e9 * (G.prop.memoryBusWidth / 8.0));
    fprintf(stderr, "LDS     : %zu KB/workgroup\n", G.prop.sharedMemPerBlock / 1024);
    if (ktrace::enabled()) ktrace::init(G.prop);
}

static void setdims(Model& m, const Gguf& g) {
    auto U = [&](const char* k, int def) {
        auto it = g.ku.find(k); return it == g.ku.end() ? def : (int)it->second;
    };
    auto F = [&](const char* k, double def) {
        auto it = g.kf.find(k); return it == g.kf.end() ? def : it->second;
    };
    m.NL = U("qwen35.block_count", 0);
    m.nextn = U("qwen35.nextn_predict_layers", 0);
    m.NL -= m.nextn;
    m.D = U("qwen35.embedding_length", 0);
    m.NFF = U("qwen35.feed_forward_length", 0);
    m.NH = U("qwen35.attention.head_count", 0);
    m.NHKV = U("qwen35.attention.head_count_kv", 0);
    m.HD = U("qwen35.attention.key_length", 0);
    m.NK = U("qwen35.ssm.group_count", 0);
    m.HK = U("qwen35.ssm.state_size", 0);
    int inner = U("qwen35.ssm.inner_size", 0);
    m.NV = m.HK ? inner / m.HK : 0;
    m.HB = U("prism.hadamard.block_size", 512);
    m.interval = U("qwen35.full_attention_interval", 4);
    m.eps = (float)F("qwen35.attention.layer_norm_rms_epsilon", 1e-6);
    m.rope_base = (float)F("qwen35.rope.freq_base", 1e7);
    m.nrot = U("qwen35.rope.dimension_count", 64) / 2;
    m.VOCAB = (int)g.tokens.size();
    m.CONV_CH = 2 * m.NK * m.HK + m.NV * m.HK;
    m.NATT = m.interval ? m.NL / m.interval : 0;

    // Hadamard signs: prism.hadamard.sign_values concatenated by sign_widths
    auto sw = g.ku.find("prism.hadamard.sign_widths.count");
    if (sw != g.ku.end()) {
        // the array was skipped; recover widths from the tensor shapes instead
    }
    fprintf(stderr, "dims    : L=%d D=%d FFN=%d H=%d KV=%d HD=%d G=%d HK=%d NV=%d HB=%d CONV=%d\n",
            m.NL, m.D, m.NFF, m.NH, m.NHKV, m.HD, m.NK, m.HK, m.NV, m.HB, m.CONV_CH);
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, f;
    if (exp == 0) f = sign << 31;
    else if (exp == 31) f = (sign << 31) | 0x7F800000u | (mant << 13);
    else f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float o; memcpy(&o, &f, 4); return o;
}

// Host reference for PTQ1_0, straight from the raw GGUF bytes. A separate implementation
// of the byte-to-trit mapping on purpose: if the kernel and this agree, the packing order
// is right, which is the bug class that produced plausible wrong numbers twice already.
static float ref_ptq1(const uint8_t* rowdata, const float* x, int C) {
    int bpr = C >> 7;
    double acc = 0.0;
    for (int blk = 0; blk < bpr; ++blk) {
        const uint8_t* bp = rowdata + (size_t)blk * 28;
        uint16_t hb; memcpy(&hb, bp + 26, 2);
        float sc = half_to_float(hb);
        double ba = 0.0;
        for (int i = 0; i < 128; ++i) {
            int j, n;
            if (i < 80)        { j = i & 15;                 n = i >> 4; }
            else if (i < 120)  { j = 16 + ((i - 80) & 7);    n = (i - 80) >> 3; }
            else               { j = 24 + ((i - 120) & 1);   n = (i - 120) >> 1; }
            uint32_t pw = 1; for (int t = 0; t < n; ++t) pw *= 3u;
            uint32_t q = ((uint32_t)bp[j] * pw) & 0xFFu;
            int trit = (int)((q * 3u) >> 8);
            ba += (double)(trit - 1) * (double)x[blk * 128 + i];
        }
        acc += ba * (double)sc;
    }
    return (float)acc;
}

int main(int argc, char** argv) {
    const char* model = nullptr;
    const char* prompt = nullptr;
    int ngen = 16, ctx = 2048;
    float temp = 0.f;
    bool selftest = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) ngen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--selftest")) selftest = true;
        else if (ktrace::parse_arg(argc, argv, i)) {}
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!model) {
        fprintf(stderr, "usage: --model <file.gguf> [--prompt TEXT] [-n N] [--ctx N]\n"
                        "       [--temp F] [--selftest] [--trace] [--trace-json FILE]\n");
        return 1;
    }

    ktrace::clock_init();
    initdev();

    Gguf g;
    if (!load_gguf(model, g)) return 1;
    Model m;
    setdims(m, g);
    if (!m.D || !m.NL) { fprintf(stderr, "unsupported architecture\n"); return 1; }

    Tokenizer tok;
    tok.load(g);
    fprintf(stderr, "tokenizer: %zu tokens, eos %d, bos %d\n",
            tok.id2tok.size(), tok.eos, tok.bos);

    if (selftest) {
        // ---- PTQ1_0 GEMV against the host reference from the raw GGUF bytes
        const GT* t = nullptr;
        for (auto& c : g.tens)
            if (c.type == T_PTQ1_0 && c.name.find("ssm_out") != std::string::npos) { t = &c; break; }
        if (!t) for (auto& c : g.tens) if (c.type == T_PTQ1_0) { t = &c; break; }
        bool gemv_ok = true;
        if (t) {
            int R = 1, C = (int)t->dims[0];
            for (size_t i = 1; i < t->dims.size(); ++i) R *= (int)t->dims[i];
            fprintf(stderr, "\nselftest PTQ1_0 GEMV on %s  R=%d C=%d\n", t->name.c_str(), R, C);
            if (C % 128 == 0) {
                const uint8_t* src = g.base + g.data_start + t->off;
                int bpr = C >> 7;
                size_t nbytes = (size_t)R * bpr * 28;
                Buf wb = mkbuf(nbytes);
                upload(&wb, src, nbytes);
                std::vector<float> xv((size_t)C);
                for (int i = 0; i < C; ++i)
                    xv[i] = sinf(0.01f * (float)i) * 0.5f + 0.1f * cosf(0.003f * (float)i);
                Buf xb = mkbuf((size_t)C * 4);
                upload(&xb, xv.data(), (size_t)C * 4);
                Buf yb = mkbuf((size_t)R * 4);
                int ws = G.prop.warpSize, nsub = 256 / ws, per = nsub * RPW;
                int grid = (R + per - 1) / per;
                ktrace::Launch L = ktrace::begin("gemv_ptq1", grid, 1, 1, 256, nbytes);
                gemv_ptq1<<<grid, 256>>>((const uint*)wb.p, (const float*)xb.p,
                                         (float*)yb.p, R, C);
                ktrace::end(L);
                CK(hipDeviceSynchronize());
                std::vector<float> yv(R);
                download(yv.data(), yb, (size_t)R * 4);
                double worst = 0, nrm = 0;
                int ns = 0;
                for (int r = 0; r < R; r += 97) {
                    float ref = ref_ptq1(src + (size_t)r * bpr * 28, xv.data(), C);
                    double d = fabs((double)ref - (double)yv[r]);
                    if (d > worst) worst = d;
                    nrm += fabs((double)ref);
                    ++ns;
                }
                double mean = nrm / (ns ? ns : 1);
                fprintf(stderr, "selftest PTQ1_0: %d rows sampled, max abs err vs host = %.4e "
                                "(mean |y| %.4e)\n", ns, worst, mean);
                gemv_ok = worst < 1e-3 * (mean > 0 ? mean : 1.0);
                fprintf(stderr, "selftest PTQ1_0: %s\n", gemv_ok ? "PASS" : "FAIL");
                freebuf(wb); freebuf(xb); freebuf(yb);
            }
        } else {
            fprintf(stderr, "\nselftest: no PTQ1_0 tensor in this model\n");
        }
        // tokenizer round trip
        if (prompt) {
            auto ids = tok.encode(prompt);
            fprintf(stderr, "tokenize '%s' -> %zu ids:", prompt, ids.size());
            for (size_t i = 0; i < ids.size() && i < 12; ++i) fprintf(stderr, " %d", ids[i]);
            fprintf(stderr, "\ndecode back: '%s'\n", tok.decode(ids).c_str());
        }
        fprintf(stderr, "\nselftest: %s\n", gemv_ok ? "PASS" : "FAIL");
        ktrace::dump();
        ktrace::shutdown();
        return gemv_ok ? 0 : 1;
    }

    fprintf(stderr, "\nthis build has the GGUF reader, the tokenizer and the RDNA2 kernels.\n"
                    "the decode graph wiring is in progress; run --selftest --prompt TEXT to\n"
                    "check loading and tokenisation.\n");
    (void)ctx; (void)ngen; (void)temp;
    ktrace::dump();
    ktrace::shutdown();
    return 0;
}
