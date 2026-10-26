// trace.h - shared performance instrumentation for the kindi Vulkan/CPU engines.
//
// Header-only. Always compiled in. Enabled by the --trace CLI flag (never an env var:
// a trace that needs an environment variable to exist is a trace nobody finds).
//
// What it records, per the instrumentation contract:
//   dispatch trap      - launch count, grid/block dims, push-constant size
//   memory trap        - bytes read+written per op, aggregated per component
//   synchronisation    - submits, fence waits, and the idle gap between host and device
//   allocation trap    - allocation count/bytes, frees
//   transfer trap      - host<->device copies, bytes, time
//   dependency trap    - ops are recorded in issue order, so the critical path is visible
//   accuracy trap      - generated tokens are hashed so a before/after run can be compared
//   regression trap    - --trace-json writes a machine-readable baseline to diff
//
// Device time comes from vkCmdWriteTimestamp pairs around each dispatch (this GPU
// reports timestampComputeAndGraphics = true, timestampPeriod = 40 ns). Host time is
// QueryPerformanceCounter around the submit+fence path. idle = host - device, i.e. the
// launch latency and fence wait that the device was not executing work.
//
// The instrumentation floor is measured at exit by timing 256 empty ops through the
// same record path, so the reader can tell signal from overhead.

#pragma once
#include <vulkan/vulkan.h>
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <deque>

namespace ktrace {

struct Op {
    const char* name;
    const char* comp;
    uint32_t gx, gy, gz, lx, psz;
    uint64_t bytes;
    double dev_us, host_us;
};

struct Agg {
    const char* label;
    uint64_t ops;
    double dev, host, bytes;
    double waves;
};

struct Cores {
    uint32_t cu = 0, simdPerCu = 0, wavefrontsPerSimd = 0, waveSize = 0;
    uint32_t totalWaveSlots() const { return cu * simdPerCu * wavefrontsPerSimd; }
};

struct State {
    bool on = false;
    bool floor_only = false;
    bool json = false;
    const char* json_path = nullptr;
    const char* label = "";
    const char* tag = "";   // per-phase prefix, e.g. "mtp:"

    std::vector<Op> ops;

    // counters
    uint64_t allocs = 0, alloc_bytes = 0, frees = 0;
    uint64_t xfers = 0, xfer_bytes = 0;
    double   xfer_us = 0;
    uint64_t submits = 0, fence_waits = 0;
    double   host_total_us = 0;
    double   queue_wait_us = 0;
    uint64_t flushes = 0;

    // accuracy: FNV-1a over the emitted token ids
    uint64_t token_hash = 1469598103934665603ULL;
    uint64_t token_count = 0;

    // device timing
    VkDevice dev = VK_NULL_HANDLE;
    VkQueryPool pool = VK_NULL_HANDLE;
    uint32_t max_pairs = 0;
    uint32_t pairs = 0;
    double timestamp_period_ns = 0;
    std::vector<double> dev_ticks;   // resolved device time per recorded op, in us

    double floor_host_us = 0, floor_dev_us = 0;
    Cores cores;
    LARGE_INTEGER qf{};
};

inline State& st() { static State s; return s; }

inline bool enabled() { return st().on; }

inline void clock_init() {
    QueryPerformanceFrequency(&st().qf);
}

inline double now_us() {
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e6 / (double)st().qf.QuadPart;
}

// Component bucket from a shader/op name.
inline const char* comp_of(const char* n) {
    auto has = [&](const char* s) { return n && strstr(n, s) != nullptr; };
    // phase override: the MTP draft head is timed separately from the trunk, so its
    // cost is not silently folded into the trunk's per-component totals
    if (n && !strncmp(n, "mtp:", 4))                       return "mtp";
    if (has("deq_emb"))                                   return "embed";
    if (has("argmax") || has("topk") || has("mask_neg"))   return "head+sample";
    if (has("gdn_"))                                       return "recurrent";
    if (has("smfold") || has("fold_fwht") || has("silu"))  return "ffn-activate";
    if (has("rmsfold") || has("rmsw") || has("rmsn") || has("fwht_hi") || has("l2w"))
                                                           return "norms";
    if (has("attn_") || has("softmax") || has("rope"))     return "attention";
    if (has("g4") || has("g5") || has("g2") || has("q8f") || has("q8gemv") || has("q8deq"))
                                                           return "projections";
    if (has("store_h") || has("sumparts") || has("addv") || has("cpq") ||
        has("expand") || has("band"))                      return "residual";
    return "misc";
}

// Create the timestamp pool. Call once, after the device exists.
inline void init(VkPhysicalDevice phys, VkDevice dev, uint32_t max_ops = 200000) {
    State& s = st();
    s.dev = dev;
    clock_init();
    VkPhysicalDeviceProperties pp{};
    vkGetPhysicalDeviceProperties(phys, &pp);
    s.timestamp_period_ns = (double)pp.limits.timestampPeriod;
    if (s.timestamp_period_ns <= 0) s.timestamp_period_ns = 1.0;
    s.max_pairs = max_ops;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = max_ops * 2;
    if (vkCreateQueryPool(dev, &qi, nullptr, &s.pool) != VK_SUCCESS) {
        s.pool = VK_NULL_HANDLE;
        fprintf(stderr, "trace: timestamp pool unavailable, device time disabled\n");
    }
    s.ops.reserve(max_ops);
    s.dev_ticks.reserve(max_ops);
}

// Emit the two timestamps into an already-recording command buffer.
inline void mark_before(VkCommandBuffer cb, uint32_t& idx) {
    State& s = st();
    if (!s.on || s.pool == VK_NULL_HANDLE || s.pairs >= s.max_pairs) { idx = UINT32_MAX; return; }
    idx = s.pairs;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, idx * 2);
}
inline void mark_after(VkCommandBuffer cb, uint32_t idx) {
    State& s = st();
    if (!s.on || idx == UINT32_MAX) return;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, idx * 2 + 1);
}

// Record one op. dev_us < 0 means "resolve later from the query pool".
inline void record(const char* name, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx,
                   uint32_t psz, uint64_t bytes, double host_us, double dev_us,
                   uint32_t pair_idx) {
    State& s = st();
    if (!s.on || s.floor_only) return;
    Op o{};
    if (s.tag && s.tag[0]) {
        static std::deque<std::string> tagged;   // stable c_str() for the op table
        tagged.push_back(std::string(s.tag) + name);
        o.name = tagged.back().c_str();
    } else {
        o.name = name;
    }
    o.comp = comp_of(o.name);
    o.gx = gx; o.gy = gy; o.gz = gz; o.lx = lx; o.psz = psz;
    o.bytes = bytes; o.host_us = host_us; o.dev_us = dev_us;
    if (pair_idx != UINT32_MAX && dev_us < 0) {
        if (s.dev_ticks.size() <= pair_idx) s.dev_ticks.resize(pair_idx + 1, -1.0);
        s.dev_ticks[pair_idx] = -2.0;   // pending
    }
    s.ops.push_back(o);
    s.host_total_us += host_us;
    s.pairs++;
}

// Resolve pending device times. Call after the queue is idle.
inline void resolve() {
    State& s = st();
    if (!s.on || s.pool == VK_NULL_HANDLE || s.pairs == 0) return;
    std::vector<uint64_t> ts((size_t)s.pairs * 2);
    if (vkGetQueryPoolResults(s.dev, s.pool, 0, s.pairs * 2, ts.size() * sizeof(uint64_t),
                              ts.data(), sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
        return;
    // ops recorded without a pair index keep whatever dev_us they were given
    size_t pi = 0;
    for (auto& o : s.ops) {
        if (pi < s.pairs) {
            uint64_t a = ts[pi * 2], b = ts[pi * 2 + 1];
            o.dev_us = (b >= a) ? (double)(b - a) * s.timestamp_period_ns / 1000.0 : 0.0;
            pi++;
        }
    }
}

// Phase tag for the trace ("" clears it). The engine sets "mtp:" around draft work.
inline void set_tag(const char* t) { st().tag = t ? t : ""; }

inline void count_alloc(uint64_t bytes) { if (st().on) { st().allocs++; st().alloc_bytes += bytes; } }
inline void count_free()               { if (st().on) st().frees++; }
inline void count_xfer(uint64_t bytes, double us) {
    if (!st().on) return;
    st().xfers++; st().xfer_bytes += bytes; st().xfer_us += us;
}
inline void set_cores(uint32_t cu, uint32_t simd, uint32_t wf, uint32_t ws) {
    State& s = st(); s.cores.cu = cu; s.cores.simdPerCu = simd;
    s.cores.wavefrontsPerSimd = wf; s.cores.waveSize = ws;
}

inline void count_submit() { if (st().on) st().submits++; }
inline void count_fence()  { if (st().on) st().fence_waits++; }
inline void add_queue_wait(double us) { if (st().on) { st().queue_wait_us += us; st().flushes++; } }

inline void accuracy_token(int id) {
    State& s = st();
    if (!s.on) return;
    uint64_t h = s.token_hash;
    for (int i = 0; i < 4; ++i) { h ^= (uint64_t)((uint32_t)id >> (i * 8)) & 0xFFULL; h *= 1099511628211ULL; }
    s.token_hash = h; s.token_count++;
}

// Measure the cost of the recording path itself, so overhead is quantified not assumed.
inline void measure_floor() {
    State& s = st();
    if (!s.on) return;
    s.floor_only = true;
    double t0 = now_us();
    for (int i = 0; i < 256; ++i) {
        s.floor_only = false;
        record("__floor__", 0, 0, 0, 0, 0, 0, now_us() - t0, 0.0, UINT32_MAX);
        s.ops.pop_back();
        s.floor_only = true;
    }
    double t1 = now_us();
    s.floor_only = false;
    s.floor_host_us = (t1 - t0) / 256.0;
    s.floor_dev_us = 0.0;
}

inline void dump() {
    State& s = st();
    if (!s.on) return;
    resolve();
    measure_floor();

    // ---- per-component table
    std::map<std::string, Agg> ag;
    for (auto& o : s.ops) {
        auto& a = ag[o.comp];
        a.label = o.comp; a.ops++; a.dev += o.dev_us; a.host += o.host_us; a.bytes += o.bytes;
        a.waves += (double)o.gx * o.gy * o.gz * ((o.lx + 63) / 64);
    }
    std::vector<Agg> v; v.reserve(ag.size());
    for (auto& kv : ag) v.push_back(kv.second);
    std::sort(v.begin(), v.end(), [](const Agg& a, const Agg& b) { return a.dev > b.dev; });
    double tot = 0; for (auto& a : v) tot += a.dev;

    fprintf(stderr, "\n===== TRACE %s =====\n", s.label);
    if (s.cores.cu)
        fprintf(stderr, "     compute cores: %u CU x %u SIMD x %u wavefronts = %u wave slots "
                        "(wave size %u, %u threads/CU)\n",
                s.cores.cu, s.cores.simdPerCu, s.cores.wavefrontsPerSimd,
                s.cores.totalWaveSlots(), s.cores.waveSize,
                s.cores.simdPerCu * s.cores.waveSize);
    fprintf(stderr, "     %-16s %7s %6s %12s %12s %12s %9s\n",
            "component", "ops", "%dev", "dev us", "idle us", "host us", "bindMB");
    for (auto& a : v) {
        double idle = a.host - a.dev; if (idle < 0) idle = 0;
        fprintf(stderr, "     %-16s %7llu %5.1f%% %12.2f %12.2f %12.2f %9.2f\n",
                a.label, (unsigned long long)a.ops, tot > 0 ? 100.0 * a.dev / tot : 0.0,
                a.dev, idle, a.host, a.bytes / 1048576.0);
    }
    {
        double hid = s.host_total_us - tot;
        fprintf(stderr, "     %-16s %7llu %5.1f%% %12.2f %12.2f %12.2f %9.2f\n",
                "TOTAL", (unsigned long long)s.ops.size(), 100.0, tot,
                hid > 0 ? hid : 0.0, s.host_total_us, 0.0);
    }

    // ---- per-shader table
    std::map<std::string, Agg> sh;
    for (auto& o : s.ops) {
        auto& a = sh[o.name];
        a.label = o.name; a.ops++; a.dev += o.dev_us; a.host += o.host_us; a.bytes += o.bytes;
        a.waves += (double)o.gx * o.gy * o.gz * ((o.lx + 63) / 64);
    }
    std::vector<Agg> sv; sv.reserve(sh.size());
    for (auto& kv : sh) sv.push_back(kv.second);
    std::sort(sv.begin(), sv.end(), [](const Agg& a, const Agg& b) { return a.dev > b.dev; });

    fprintf(stderr, "\n     --- per shader (top 25 of %zu) ---  bindMB = sum of bound buffer ranges (traffic upper bound)\n", sv.size());
    fprintf(stderr, "     %-16s %7s %12s %12s %12s %9s %10s %10s %8s\n",
            "shader", "ops", "dev us", "idle us", "host us", "bindMB", "dev/op", "waves", "w/CU");
    for (size_t i = 0; i < sv.size() && i < 25; ++i) {
        double idle = sv[i].host - sv[i].dev; if (idle < 0) idle = 0;
        fprintf(stderr, "     %-16s %7llu %12.2f %12.2f %12.2f %9.2f %10.3f %10.0f %8.1f\n",
                sv[i].label, (unsigned long long)sv[i].ops, sv[i].dev, idle, sv[i].host,
                sv[i].bytes / 1048576.0, sv[i].ops ? sv[i].dev / (double)sv[i].ops : 0.0,
                sv[i].waves, s.cores.cu ? sv[i].waves / (double)s.cores.cu : 0.0);
    }

    // ---- traps
    {
        double ops_per_submit = s.submits ? (double)s.ops.size() / (double)s.submits : 0.0;
        fprintf(stderr, "\n     synchronisation: submits=%llu fence_waits=%llu  (%.1f ops per submit)\n",
                (unsigned long long)s.submits, (unsigned long long)s.fence_waits, ops_per_submit);
        if (s.submits && ops_per_submit < 2.0)
            fprintf(stderr, "                      <-- one submit per op: batching disabled or broken\n");
    }
    fprintf(stderr, "     allocation     : %llu allocs (%.2f MB), %llu frees\n",
            (unsigned long long)s.allocs, s.alloc_bytes / 1048576.0, (unsigned long long)s.frees);
    fprintf(stderr, "     transfer       : %llu copies (%.2f MB, %.2f us)\n",
            (unsigned long long)s.xfers, s.xfer_bytes / 1048576.0, s.xfer_us);
    fprintf(stderr, "     accuracy       : %llu tokens, hash=%016llx\n",
            (unsigned long long)s.token_count, (unsigned long long)s.token_hash);
    fprintf(stderr, "     queue wait     : %.2f ms in %llu flushes (%.2f ms each)\n",
            s.queue_wait_us / 1000.0, (unsigned long long)s.flushes,
            s.flushes ? s.queue_wait_us / 1000.0 / (double)s.flushes : 0.0);
    fprintf(stderr, "     device+wait    : %.2f ms   (device %.2f ms)\n",
            (tot + s.queue_wait_us) / 1000.0, tot / 1000.0);
    fprintf(stderr, "     host total     : %.2f ms across %zu ops (%.2f us/op)\n",
            s.host_total_us / 1000.0, s.ops.size(),
            s.ops.empty() ? 0.0 : s.host_total_us / (double)s.ops.size());
    fprintf(stderr, "     floor          : %.3f us host per recorded op (256 empty ops)\n",
            s.floor_host_us);

    if (s.json && s.json_path) {
        FILE* f = fopen(s.json_path, "w");
        if (f) {
            fprintf(f, "{\n  \"label\": \"%s\",\n", s.label);
            fprintf(f, "  \"ops\": %zu,\n  \"dev_us\": %.2f,\n  \"host_us\": %.2f,\n",
                    s.ops.size(), tot, s.host_total_us);
            fprintf(f, "  \"submits\": %llu,\n  \"fence_waits\": %llu,\n",
                    (unsigned long long)s.submits, (unsigned long long)s.fence_waits);
            fprintf(f, "  \"allocs\": %llu,\n  \"alloc_bytes\": %llu,\n",
                    (unsigned long long)s.allocs, (unsigned long long)s.alloc_bytes);
            fprintf(f, "  \"xfers\": %llu,\n  \"xfer_bytes\": %llu,\n",
                    (unsigned long long)s.xfers, (unsigned long long)s.xfer_bytes);
            fprintf(f, "  \"tokens\": %llu,\n  \"token_hash\": \"%016llx\",\n",
                    (unsigned long long)s.token_count, (unsigned long long)s.token_hash);
            fprintf(f, "  \"floor_host_us\": %.4f,\n", s.floor_host_us);
            fprintf(f, "  \"components\": [\n");
            for (size_t i = 0; i < v.size(); ++i) {
                double idle = v[i].host - v[i].dev; if (idle < 0) idle = 0;
                fprintf(f, "    {\"name\": \"%s\", \"ops\": %llu, \"dev_us\": %.3f, "
                           "\"idle_us\": %.3f, \"host_us\": %.3f, \"bytes\": %llu}%s\n",
                        v[i].label, (unsigned long long)v[i].ops, v[i].dev, idle, v[i].host,
                        (unsigned long long)v[i].bytes, i + 1 < v.size() ? "," : "");
            }
            fprintf(f, "  ],\n  \"shaders\": [\n");
            for (size_t i = 0; i < sv.size(); ++i) {
                fprintf(f, "    {\"name\": \"%s\", \"ops\": %llu, \"dev_us\": %.3f, "
                           "\"bytes\": %llu}%s\n",
                        sv[i].label, (unsigned long long)sv[i].ops, sv[i].dev,
                        (unsigned long long)sv[i].bytes, i + 1 < sv.size() ? "," : "");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
            fprintf(stderr, "     baseline written: %s\n", s.json_path);
        }
    }
}

// Parse the shared tracing flags. Returns true if the arg was consumed.
inline bool parse_arg(int argc, char** argv, int& i) {
    State& s = st();
    if (!strcmp(argv[i], "--trace")) { s.on = true; return true; }
    if (!strcmp(argv[i], "--trace-label") && i + 1 < argc) { s.label = argv[++i]; return true; }
    if (!strcmp(argv[i], "--trace-json") && i + 1 < argc) {
        s.on = true; s.json = true; s.json_path = argv[++i]; return true;
    }
    return false;
}

} // namespace ktrace
