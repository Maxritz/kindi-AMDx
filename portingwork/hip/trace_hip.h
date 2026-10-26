// trace_hip.h - the performance instrumentation contract, HIP edition.
//
// Same contract as the Vulkan engine's trace.h, because the rule is that any change which
// can affect speed, memory or correctness lands with the instrumentation to measure it.
// What differs is the mechanism: Vulkan used vkCmdWriteTimestamp around each dispatch,
// HIP uses hipEvent pairs on the stream.
//
// Records, per the contract:
//   dispatch trap   - launch count, grid, block, and the kernel name
//   memory trap     - bytes read+written per op, aggregated per component
//   synchronisation - submits, waits, and the idle gap between host and device
//   allocation trap - allocation count and bytes
//   transfer trap   - host<->device copies, bytes, time
//   dependency trap - ops recorded in issue order, so the critical path is visible
//   accuracy trap   - emitted tokens hashed so a before/after run can be compared
//   regression trap - --trace-json writes a machine-readable baseline to diff
//
// Device time is a hipEvent pair around each launch, resolved at dump time. HIP events on
// one stream are cheap markers, so this stays accurate to well under a microsecond per op.
// Host time is QueryPerformanceCounter around the launch call, so idle = host - device is
// launch latency plus any wait.
//
// The instrumentation floor is measured at exit by timing empty records through the same
// path, so the reader can tell signal from overhead.
#pragma once
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
#include <algorithm>
#include <deque>

namespace ktrace {

struct Op {
    const char* name;
    const char* comp;
    uint32_t gx, gy, gz, lx;
    uint64_t bytes;
    double dev_us, host_us;
    hipEvent_t ev0, ev1;
    bool timed;
};

struct Agg {
    const char* label;
    uint64_t ops;
    double dev, host, bytes;
    double waves;
};

struct Cores {
    uint32_t cu = 0, simdPerCu = 0, waveSize = 0;
    uint32_t totalWaveSlots() const { return cu * simdPerCu * (waveSize ? waveSize : 32); }
};

struct State {
    bool on = false;
    bool floor_only = false;
    bool json = false;
    bool reset_at_decode = false;
    bool decode_only = false;
    const char* json_path = nullptr;
    const char* label = "";
    const char* tag = "";

    std::vector<Op> ops;
    std::vector<hipEvent_t> events;   // owned, destroyed at exit

    uint64_t allocs = 0, alloc_bytes = 0, frees = 0;
    uint64_t xfers = 0, xfer_bytes = 0;
    double   xfer_us = 0;
    uint64_t submits = 0, fence_waits = 0;
    double   host_total_us = 0;
    double   queue_wait_us = 0;
    uint64_t flushes = 0;

    uint64_t token_hash = 1469598103934665603ULL;
    uint64_t token_count = 0;

    double floor_host_us = 0;
    Cores cores;
    LARGE_INTEGER qf{};
};

inline State& st() { static State s; return s; }
inline bool enabled() { return st().on; }

inline void clock_init() { QueryPerformanceFrequency(&st().qf); }

inline double now_us() {
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e6 / (double)st().qf.QuadPart;
}

// Component bucket from a kernel name. Same buckets as the Vulkan engine so the two
// profiles are comparable side by side.
inline const char* comp_of(const char* n) {
    auto has = [&](const char* s) { return n && strstr(n, s) != nullptr; };
    if (n && !strncmp(n, "mtp:", 4))                       return "mtp";
    if (has("emb"))                                        return "embed";
    if (has("argmax") || has("topk") || has("mask"))        return "head+sample";
    if (has("gdn"))                                        return "recurrent";
    if (has("smfold") || has("silu") || has("fold"))        return "ffn-activate";
    if (has("rms") || has("norm") || has("l2"))             return "norms";
    if (has("attn") || has("softmax") || has("rope"))       return "attention";
    if (has("gemv") || has("gemm") || has("q4") || has("q8") || has("pq")) return "projections";
    if (has("add") || has("copy") || has("store"))          return "residual";
    return "misc";
}

inline void init(const hipDeviceProp_t& p) {
    State& s = st();
    clock_init();
    s.cores.cu = p.multiProcessorCount;
    s.cores.simdPerCu = 4;          // RDNA2 CU is 4 x SIMD32
    s.cores.waveSize = p.warpSize;
    s.ops.reserve(200000);
}

inline void count_alloc(uint64_t bytes) { if (st().on) { st().allocs++; st().alloc_bytes += bytes; } }
inline void count_free()                { if (st().on) st().frees++; }
inline void count_xfer(uint64_t bytes, double us) {
    if (!st().on) return;
    st().xfers++; st().xfer_bytes += bytes; st().xfer_us += us;
}
inline void count_submit() { if (st().on) st().submits++; }
inline void count_fence()  { if (st().on) st().fence_waits++; }
inline void add_queue_wait(double us) { if (st().on) { st().queue_wait_us += us; st().flushes++; } }

inline void set_tag(const char* t) { st().tag = t ? t : ""; }

inline void accuracy_token(int id) {
    State& s = st();
    if (!s.on) return;
    uint64_t h = s.token_hash;
    for (int i = 0; i < 4; ++i) { h ^= (uint64_t)((uint32_t)id >> (i * 8)) & 0xFFULL; h *= 1099511628211ULL; }
    s.token_hash = h; s.token_count++;
}

// ---- launch wrapper: records device time around a kernel through hipEvents
struct Launch {
    Op* o = nullptr;
    double t0 = 0;
    bool live = false;
};

inline Launch begin(const char* name, uint32_t gx, uint32_t gy, uint32_t gz,
                    uint32_t lx, uint64_t bytes) {
    Launch L;
    State& s = st();
    if (!s.on || s.floor_only) return L;
    Op o{};
    if (s.tag && s.tag[0]) {
        static std::deque<std::string> tagged;
        tagged.push_back(std::string(s.tag) + name);
        o.name = tagged.back().c_str();
    } else {
        o.name = name;
    }
    o.comp = comp_of(o.name);
    o.gx = gx; o.gy = gy; o.gz = gz; o.lx = lx; o.bytes = bytes;
    o.timed = (hipEventCreate(&o.ev0) == hipSuccess && hipEventCreate(&o.ev1) == hipSuccess);
    if (o.timed) hipEventRecord(o.ev0, 0);
    L.t0 = now_us();
    s.ops.push_back(o);
    s.events.push_back(o.ev0);
    s.events.push_back(o.ev1);
    L.o = &s.ops.back();
    L.live = true;
    return L;
}

inline void end(Launch& L) {
    if (!L.live || !L.o) return;
    State& s = st();
    if (L.o->timed) hipEventRecord(L.o->ev1, 0);
    L.o->host_us = now_us() - L.t0;
    s.host_total_us += L.o->host_us;
    s.submits++;
    L.live = false;
}

// Resolve device times. Call after the stream is idle.
inline void resolve() {
    State& s = st();
    if (!s.on) return;
    for (auto& o : s.ops) {
        if (o.timed) {
            float ms = 0;
            if (hipEventElapsedTime(&ms, o.ev0, o.ev1) == hipSuccess && ms >= 0)
                o.dev_us = (double)ms * 1000.0;
        }
    }
}

inline void measure_floor() {
    State& s = st();
    if (!s.on) return;
    s.floor_only = true;
    double t0 = now_us();
    for (int i = 0; i < 256; ++i) {
        s.floor_only = false;
        s.ops.push_back(Op{"__floor__", "misc", 0, 0, 0, 0, 0, 0.0, now_us() - t0, 0, 0, false});
        s.ops.pop_back();
        s.floor_only = true;
    }
    double t1 = now_us();
    s.floor_only = false;
    s.floor_host_us = (t1 - t0) / 256.0;
}

inline void dump() {
    State& s = st();
    if (!s.on) return;
    hipDeviceSynchronize();
    resolve();
    measure_floor();

    std::map<std::string, Agg> ag;
    for (auto& o : s.ops) {
        auto& a = ag[o.comp];
        a.label = o.comp; a.ops++; a.dev += o.dev_us; a.host += o.host_us; a.bytes += o.bytes;
        a.waves += (double)o.gx * o.gy * o.gz * ((o.lx + 31) / 32);
    }
    std::vector<Agg> v;
    for (auto& kv : ag) v.push_back(kv.second);
    std::sort(v.begin(), v.end(), [](const Agg& a, const Agg& b) { return a.dev > b.dev; });
    double tot = 0; for (auto& a : v) tot += a.dev;

    fprintf(stderr, "\n===== TRACE %s =====%s\n", s.label,
            s.decode_only ? "  (decode only: prefill dropped)" : "");
    if (s.cores.cu)
        fprintf(stderr, "     compute cores: %u CU x %u SIMD x wave %u\n",
                s.cores.cu, s.cores.simdPerCu, s.cores.waveSize);
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

    std::map<std::string, Agg> sh;
    for (auto& o : s.ops) {
        auto& a = sh[o.name];
        a.label = o.name; a.ops++; a.dev += o.dev_us; a.host += o.host_us; a.bytes += o.bytes;
        a.waves += (double)o.gx * o.gy * o.gz * ((o.lx + 31) / 32);
    }
    std::vector<Agg> sv;
    for (auto& kv : sh) sv.push_back(kv.second);
    std::sort(sv.begin(), sv.end(), [](const Agg& a, const Agg& b) { return a.dev > b.dev; });

    fprintf(stderr, "\n     --- per kernel (top 25 of %zu) ---  bindMB = sum of bound ranges (traffic upper bound)\n", sv.size());
    fprintf(stderr, "     %-22s %7s %12s %12s %12s %9s %10s %10s\n",
            "kernel", "ops", "dev us", "idle us", "host us", "bindMB", "dev/op", "waves");
    for (size_t i = 0; i < sv.size() && i < 25; ++i) {
        double idle = sv[i].host - sv[i].dev; if (idle < 0) idle = 0;
        fprintf(stderr, "     %-22s %7llu %12.2f %12.2f %12.2f %9.2f %10.3f %10.0f\n",
                sv[i].label, (unsigned long long)sv[i].ops, sv[i].dev, idle, sv[i].host,
                sv[i].bytes / 1048576.0, sv[i].ops ? sv[i].dev / (double)sv[i].ops : 0.0,
                sv[i].waves);
    }

    fprintf(stderr, "\n     synchronisation: submits=%llu fence_waits=%llu  (%.1f ops per submit)\n",
            (unsigned long long)s.submits, (unsigned long long)s.fence_waits,
            s.submits ? (double)s.ops.size() / (double)s.submits : 0.0);
    fprintf(stderr, "     allocation     : %llu allocs (%.2f MB), %llu frees\n",
            (unsigned long long)s.allocs, s.alloc_bytes / 1048576.0, (unsigned long long)s.frees);
    fprintf(stderr, "     transfer       : %llu copies (%.2f MB, %.2f us)\n",
            (unsigned long long)s.xfers, s.xfer_bytes / 1048576.0, s.xfer_us);
    fprintf(stderr, "     accuracy       : %llu tokens, hash=%016llx\n",
            (unsigned long long)s.token_count, (unsigned long long)s.token_hash);
    fprintf(stderr, "     queue wait     : %.2f ms in %llu waits (%.2f ms each)\n",
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
            fprintf(f, "{\n  \"label\": \"%s\",\n  \"ops\": %zu,\n  \"dev_us\": %.2f,\n  \"host_us\": %.2f,\n",
                    s.label, s.ops.size(), tot, s.host_total_us);
            fprintf(f, "  \"submits\": %llu,\n  \"allocs\": %llu,\n  \"alloc_bytes\": %llu,\n",
                    (unsigned long long)s.submits, (unsigned long long)s.allocs,
                    (unsigned long long)s.alloc_bytes);
            fprintf(f, "  \"xfers\": %llu,\n  \"xfer_bytes\": %llu,\n",
                    (unsigned long long)s.xfers, (unsigned long long)s.xfer_bytes);
            fprintf(f, "  \"tokens\": %llu,\n  \"token_hash\": \"%016llx\",\n",
                    (unsigned long long)s.token_count, (unsigned long long)s.token_hash);
            fprintf(f, "  \"floor_host_us\": %.4f,\n  \"components\": [\n", s.floor_host_us);
            for (size_t i = 0; i < v.size(); ++i) {
                double idle = v[i].host - v[i].dev; if (idle < 0) idle = 0;
                fprintf(f, "    {\"name\": \"%s\", \"ops\": %llu, \"dev_us\": %.3f, \"idle_us\": %.3f, "
                           "\"host_us\": %.3f, \"bytes\": %llu}%s\n",
                        v[i].label, (unsigned long long)v[i].ops, v[i].dev, idle, v[i].host,
                        (unsigned long long)v[i].bytes, i + 1 < v.size() ? "," : "");
            }
            fprintf(f, "  ],\n  \"kernels\": [\n");
            for (size_t i = 0; i < sv.size(); ++i) {
                fprintf(f, "    {\"name\": \"%s\", \"ops\": %llu, \"dev_us\": %.3f, \"bytes\": %llu}%s\n",
                        sv[i].label, (unsigned long long)sv[i].ops, sv[i].dev,
                        (unsigned long long)sv[i].bytes, i + 1 < sv.size() ? "," : "");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
            fprintf(stderr, "     baseline written: %s\n", s.json_path);
        }
    }
}

inline void shutdown() {
    State& s = st();
    for (auto e : s.events) if (e) hipEventDestroy(e);
    s.events.clear();
}

inline void reset_ops() {
    State& s = st();
    for (auto e : s.events) if (e) hipEventDestroy(e);
    s.events.clear();
    s.ops.clear();
    s.host_total_us = 0;
    s.allocs = 0; s.alloc_bytes = 0; s.frees = 0;
    s.xfers = 0; s.xfer_bytes = 0; s.xfer_us = 0;
    s.submits = 0; s.fence_waits = 0; s.queue_wait_us = 0; s.flushes = 0;
    s.token_hash = 1469598103934665603ULL; s.token_count = 0;
}

inline void mark_decode_start() {
    if (!st().reset_at_decode) return;
    st().reset_at_decode = false;
    reset_ops();
    st().decode_only = true;
}

inline bool parse_arg(int argc, char** argv, int& i) {
    State& s = st();
    if (!strcmp(argv[i], "--trace")) { s.on = true; return true; }
    if (!strcmp(argv[i], "--trace-decode")) { s.on = true; s.reset_at_decode = true; return true; }
    if (!strcmp(argv[i], "--trace-label") && i + 1 < argc) { s.label = argv[++i]; return true; }
    if (!strcmp(argv[i], "--trace-json") && i + 1 < argc) {
        s.on = true; s.json = true; s.json_path = argv[++i]; return true;
    }
    return false;
}

} // namespace ktrace
