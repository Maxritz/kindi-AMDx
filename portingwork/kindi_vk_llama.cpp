// kindi_vk_llama.cpp - Llama-3.2-1B-Instruct Q8_0 decode engine on Vulkan
//
// Adapted from kindi_vk_decode.cpp (Ternary-Bonsai-2, token-exact) with the
// Llama-specific differences:
//   * Q8_0 weights (type 8) instead of ternary: block = fp16 d then 32 int8
//     (34 bytes). codes and fp16 scales are split into two buffers, exactly
//     like the ternary path splits trits and scales.
//   * activations are quantized to Q8_0 before every mat-vec, matching
//     ggml_mul_mat's vec_dot_q8_0_q8_0 path (int32 block dots, then
//     acc += (d_w*d_x)*idot).
//   * no Hadamard / FWHT / sign vectors, no GDN, no attention gate.
//   * RoPE is GGML_ROPE_TYPE_NORMAL (adjacent pairs) because the GGUF q/k rows
//     are already permuted by the converter; HD == rope_dim == 64 so the whole
//     head rotates. --neox switches to rotate-half for comparison.
//   * rms norm follows ggml_compute_forward_rms_norm_f32 (fp64 sum, eps from
//     the GGUF, 1e-5 here).
//
// Instrumentation: every dispatch is timed and bucketed; --trace prints the
// per-op table, --dump prints per-layer residuals and top-5 logits.
#include <vulkan/vulkan.h>
#define NOMINMAX
#include <windows.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include <functional>
#include "tokenizer_llama.h"
#include <deque>
#include "trace.h"

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){fprintf(stderr,"Vulkan error %d at %s:%d\n",(int)_r,__FILE__,__LINE__);exit(1);} } while(0)

// ---------------- device + buffer + pipeline helpers ----------------
struct DeviceCtx {
    VkInstance inst; VkPhysicalDevice phys; VkDevice dev;
    VkQueue q0, q1, qt; uint32_t cf, tf;
    VkCommandPool cp, tp;
    VkPhysicalDeviceMemoryProperties mp; VkPhysicalDeviceProperties pp;
};
static DeviceCtx G;

static uint32_t findmt(uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < G.mp.memoryTypeCount; ++i)
        if (((bits >> i) & 1) && ((G.mp.memoryTypes[i].propertyFlags & f) == f)) return i;
    fprintf(stderr, "nomemtype\n"); exit(1);
}
struct Buf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceSize size = 0; void* mapped = nullptr; };

static Buf mkbuf(VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags f) {
    Buf b; b.size = sz;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = sz; ci.usage = u; ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
    uint32_t qf[2] = {G.cf, G.tf};
    ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = qf;
    VK_CHECK(vkCreateBuffer(G.dev, &ci, nullptr, &b.buf));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(G.dev, b.buf, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size; ai.memoryTypeIndex = findmt((uint32_t)r.memoryTypeBits, f);
    VK_CHECK(vkAllocateMemory(G.dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(G.dev, b.buf, b.mem, 0));
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VK_CHECK(vkMapMemory(G.dev, b.mem, 0, sz, 0, &b.mapped));
    return b;
}
static void freebuf(Buf& b) {
    if (b.mapped) vkUnmapMemory(G.dev, b.mem);
    if (b.buf) vkDestroyBuffer(G.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(G.dev, b.mem, nullptr);
    b = Buf{};
}

static void initdev() {
    VkApplicationInfo a{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    a.pApplicationName = "kindi-vk-llama"; a.applicationVersion = 1;
    a.pEngineName = "kindi-vk-llama"; a.engineVersion = 1; a.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &a;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &G.inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(G.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> ds(n); vkEnumeratePhysicalDevices(G.inst, &n, ds.data());
    G.phys = ds[0];
    vkGetPhysicalDeviceProperties(G.phys, &G.pp);
    vkGetPhysicalDeviceMemoryProperties(G.phys, &G.mp);
    fprintf(stderr, "GPU: %s\n", G.pp.deviceName);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &qn, qf.data());
    G.cf = UINT32_MAX; G.tf = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i) {
        if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            qf[i].queueCount >= 2 && G.cf == UINT32_MAX) G.cf = i;
        if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            G.tf == UINT32_MAX && i != G.cf) G.tf = i;
    }
    if (G.cf == UINT32_MAX || G.tf == UINT32_MAX) {
        for (uint32_t i = 0; i < qn; ++i) {
            if (G.cf == UINT32_MAX && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) G.cf = i;
            if (G.tf == UINT32_MAX && (qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && i != G.cf) G.tf = i;
        }
        if (G.tf == UINT32_MAX) G.tf = G.cf;
    }
    fprintf(stderr, "queues: compute=%u transfer=%u\n", G.cf, G.tf);
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for (int qi = 0; qi < 2; ++qi) {
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = G.cf; q.queueCount = 1;
        static float p = 1.0f; q.pQueuePriorities = &p;
        qcis.push_back(q);
    }
    {
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = G.tf; q.queueCount = 1;
        static float p = 1.0f; q.pQueuePriorities = &p;
        qcis.push_back(q);
    }
    VkPhysicalDeviceFeatures f{}; f.shaderFloat64 = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = (uint32_t)qcis.size(); dci.pQueueCreateInfos = qcis.data();
    dci.pEnabledFeatures = &f;
    VK_CHECK(vkCreateDevice(G.phys, &dci, nullptr, &G.dev));
    vkGetDeviceQueue(G.dev, G.cf, 0, &G.q0);
    vkGetDeviceQueue(G.dev, G.cf, 1, &G.q1);
    vkGetDeviceQueue(G.dev, G.tf, 0, &G.qt);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = G.cf; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(G.dev, &cpci, nullptr, &G.cp));
    cpci.queueFamilyIndex = G.tf;
    VK_CHECK(vkCreateCommandPool(G.dev, &cpci, nullptr, &G.tp));
    if (ktrace::enabled()) ktrace::init(G.phys, G.dev);
}

// per-op trace comes from the shared trace.h (--trace)

static std::vector<uint32_t> readspv(const char* p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "nospv %s\n", p); exit(1); }
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> b(sz / 4); f.read((char*)b.data(), sz); return b;
}

struct Pipe {
    VkShaderModule mod; VkDescriptorSetLayout dsl; VkPipelineLayout pl;
    VkPipeline pipe; VkDescriptorPool pool; VkDescriptorSet dset;
    const char* name = nullptr; uint32_t nbind = 0; uint64_t brange[16] = {0}; uint32_t lx = 0;
};
static Pipe mkpipe(const char* spv, int nbind, int pushsz) {
    Pipe P{}; auto code = readspv(spv);
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = code.size() * 4; sm.pCode = code.data();
    VK_CHECK(vkCreateShaderModule(G.dev, &sm, nullptr, &P.mod));
    std::vector<VkDescriptorSetLayoutBinding> binds(nbind);
    for (int i = 0; i < nbind; ++i) {
        binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount = nbind; dsl.pBindings = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(G.dev, &dsl, nullptr, &P.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pushsz};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &P.dsl;
    pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(G.dev, &pl, nullptr, &P.pl));
    VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    st.stage = VK_SHADER_STAGE_COMPUTE_BIT; st.module = P.mod; st.pName = "main";
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = st; cpi.layout = P.pl;
    VK_CHECK(vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &P.pipe));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nbind};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(G.dev, &dpi, nullptr, &P.pool));
    VkDescriptorSetAllocateInfo ds{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ds.descriptorPool = P.pool; ds.descriptorSetCount = 1; ds.pSetLayouts = &P.dsl;
    VK_CHECK(vkAllocateDescriptorSets(G.dev, &ds, &P.dset));
    P.nbind = (uint32_t)nbind;
    {   // shader name = basename without ".comp.spv"
        const char* b = spv; for (const char* c = spv; *c; ++c) if (*c == '/' || *c == '\\') b = c + 1;
        static std::deque<std::string> keep;
        keep.push_back(std::string(b));
        std::string& k = keep.back(); size_t dot = k.find(".comp"); if (dot != std::string::npos) k = k.substr(0, dot);
        P.name = k.c_str();
    }
    return P;
}
static void bindbufs(Pipe& P, const std::vector<Buf>& bufs, const std::vector<VkDeviceSize>& offs = {}) {
    std::vector<VkDescriptorBufferInfo> infos(bufs.size());
    std::vector<VkWriteDescriptorSet> w(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        infos[i].buffer = bufs[i].buf;
        infos[i].offset = (i < offs.size()) ? offs[i] : 0;
        infos[i].range = bufs[i].size - infos[i].offset;
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].pNext = nullptr;
        w[i].dstSet = P.dset; w[i].dstBinding = (uint32_t)i; w[i].dstArrayElement = 0;
        w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pImageInfo = nullptr; w[i].pBufferInfo = &infos[i]; w[i].pTexelBufferView = nullptr;
    }
    vkUpdateDescriptorSets(G.dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    for (size_t i = 0; i < bufs.size() && i < 16; ++i) P.brange[i] = infos[i].range;
}
static uint64_t pipe_bytes(const Pipe& P) {
    uint64_t b = 0; for (uint32_t i = 0; i < P.nbind && i < 16; ++i) b += P.brange[i]; return b;
}
static void dispatch(const Pipe& P, VkQueue q, VkCommandPool pool,
                     uint32_t gx, uint32_t gy, uint32_t gz, const void* push, uint32_t psz) {
    const bool tr = ktrace::enabled();
    double th0 = tr ? ktrace::now_us() : 0.0;
    uint32_t pair = UINT32_MAX;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(G.dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.pl, 0, 1, &P.dset, 0, nullptr);
    if (push && psz) vkCmdPushConstants(cb, P.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, psz, push);
    if (tr) ktrace::mark_before(cb, pair);
    vkCmdDispatch(cb, gx, gy, gz);
    if (tr) ktrace::mark_after(cb, pair);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence;
    VK_CHECK(vkCreateFence(G.dev, &fi, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(q, 1, &si, fence));
    ktrace::count_submit();
    VK_CHECK(vkWaitForFences(G.dev, 1, &fence, VK_TRUE, 60000000000ULL));
    ktrace::count_fence();
    vkDestroyFence(G.dev, fence, nullptr);
    vkFreeCommandBuffers(G.dev, pool, 1, &cb);
    if (tr) {
        double th1 = ktrace::now_us();
        ktrace::record(P.name ? P.name : "?", gx, gy, gz, P.lx, psz, pipe_bytes(P),
                       th1 - th0, -1.0, pair);
    }
}
static void upload(Buf* dst, const void* data, size_t nb) {
    Buf st = mkbuf(nb, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(st.mapped, data, nb);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = G.tp; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer c; VK_CHECK(vkAllocateCommandBuffers(G.dev, &ai, &c));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c, &bi));
    VkBufferCopy cp{0, 0, (VkDeviceSize)nb};
    vkCmdCopyBuffer(c, st.buf, dst->buf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(c));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe;
    VK_CHECK(vkCreateFence(G.dev, &fi, nullptr, &fe));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &c;
    VK_CHECK(vkQueueSubmit(G.qt, 1, &si, fe));
    VK_CHECK(vkWaitForFences(G.dev, 1, &fe, VK_TRUE, 60000000000ULL));
    vkDestroyFence(G.dev, fe, nullptr);
    vkFreeCommandBuffers(G.dev, G.tp, 1, &c);
    freebuf(st);
}
static void fillbuf(Buf& b) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = G.cp; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer c; VK_CHECK(vkAllocateCommandBuffers(G.dev, &ai, &c));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c, &bi));
    vkCmdFillBuffer(c, b.buf, 0, b.size, 0);
    VK_CHECK(vkEndCommandBuffer(c));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe;
    VK_CHECK(vkCreateFence(G.dev, &fi, nullptr, &fe));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &c;
    VK_CHECK(vkQueueSubmit(G.q0, 1, &si, fe));
    VK_CHECK(vkWaitForFences(G.dev, 1, &fe, VK_TRUE, 60000000000ULL));
    vkDestroyFence(G.dev, fe, nullptr);
    vkFreeCommandBuffers(G.dev, G.cp, 1, &c);
}
static void copybuf(Buf& src, Buf& dst, size_t bytes) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = G.cp; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer c; VK_CHECK(vkAllocateCommandBuffers(G.dev, &ai, &c));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c, &bi));
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(c, src.buf, dst.buf, 1, &region);
    VK_CHECK(vkEndCommandBuffer(c));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe;
    VK_CHECK(vkCreateFence(G.dev, &fi, nullptr, &fe));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &c;
    VK_CHECK(vkQueueSubmit(G.q0, 1, &si, fe));
    VK_CHECK(vkWaitForFences(G.dev, 1, &fe, VK_TRUE, 60000000000ULL));
    vkDestroyFence(G.dev, fe, nullptr);
    vkFreeCommandBuffers(G.dev, G.cp, 1, &c);
}

// ---------------- main ----------------
int main(int argc, char** argv) {
    const char* model = nullptr;
    const char* prompt = nullptr;
    const char* outpath = nullptr;
    float temperature = 0.0f;
    int topk = 0, dump = 0, neox = 0, npredict = 5, trace = 0, no_rope_ff = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) topk = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "--n-predict")) && i + 1 < argc) npredict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outpath = argv[++i];
        else if (!strcmp(argv[i], "--dump")) dump = 1;
        else if (ktrace::parse_arg(argc, argv, i)) { /* --trace / --trace-json / --trace-label */ }
        else if (!strcmp(argv[i], "--neox")) neox = 1;
        else if (!strcmp(argv[i], "--no-ropefreqs")) no_rope_ff = 1;
    }
    if (!model) { fprintf(stderr, "usage: --model <path.gguf> [--prompt s] [-n N] [--temp t] [--top-k k] [--dump] [--trace] [--out f]\n"); return 1; }

    
    initdev();

    // ---- memory-map GGUF ----
    HANDLE hFile = CreateFileA(model, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot open %s\n", model); return 1; }
    LARGE_INTEGER fsz; GetFileSizeEx(hFile, &fsz);
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    const unsigned char* filebase = (const unsigned char*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!filebase) { fprintf(stderr, "cannot map %s\n", model); return 1; }

    // ---- GGUF parse ----
    const unsigned char* p = filebase;
    auto rd_u32 = [&]() { uint32_t v; memcpy(&v, p, 4); p += 4; return v; };
    auto rd_u64 = [&]() { uint64_t v; memcpy(&v, p, 8); p += 8; return v; };
    auto rd_f32 = [&]() { float v; memcpy(&v, p, 4); p += 4; return v; };
    auto rd_f64 = [&]() { double v; memcpy(&v, p, 8); p += 8; return v; };
    auto rd_str = [&]() { uint64_t n = rd_u64(); std::string s((const char*)p, (size_t)n); p += n; return s; };
    uint32_t magic = rd_u32();
    if (magic != 0x46554747) { fprintf(stderr, "not GGUF (magic %x)\n", magic); return 1; }
    uint32_t version = rd_u32();
    uint64_t n_tensors = rd_u64();
    uint64_t n_kv = rd_u64();
    fprintf(stderr, "GGUF v%u: %llu tensors, %llu kv\n", version,
            (unsigned long long)n_tensors, (unsigned long long)n_kv);

    int D = 0, NL = 0, NFF = 0, NH = 0, NHKV = 0, HD = 0, rope_dim = 0, VOCAB = 0, bos = -1, eos = -1;
    float rope_base = 10000.0f, rms_eps = 1e-6f;
    uint32_t align = 32;
    std::vector<std::string> gguf_tokens, gguf_merges;
    std::vector<int> token_types;

    auto read_scalar_int = [&](uint32_t t) -> long long {
        if (t == 0) { uint8_t v = *p; p += 1; return v; }
        if (t == 1) { int8_t v; memcpy(&v, p, 1); p += 1; return v; }
        if (t == 2) { uint16_t v; memcpy(&v, p, 2); p += 2; return v; }
        if (t == 3) { int16_t v; memcpy(&v, p, 2); p += 2; return v; }
        if (t == 4) { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
        if (t == 5) { int32_t v; memcpy(&v, p, 4); p += 4; return v; }
        if (t == 6) { float v; memcpy(&v, p, 4); p += 4; return (long long)v; }
        if (t == 7) { uint8_t v = *p; p += 1; return v; }
        if (t == 10) { uint64_t v; memcpy(&v, p, 8); p += 8; return (long long)v; }
        if (t == 11) { int64_t v; memcpy(&v, p, 8); p += 8; return (long long)v; }
        if (t == 12) { double v; memcpy(&v, p, 8); p += 8; return (long long)v; }
        fprintf(stderr, "not int type %u\n", t); exit(1);
    };
    auto skip_value = [&](uint32_t type) {
        std::function<void(uint32_t)> skip = [&](uint32_t t) {
            if (t == 9) { uint32_t et = rd_u32(); uint64_t n = rd_u64(); for (uint64_t i = 0; i < n; ++i) skip(et); }
            else if (t == 8) { rd_str(); }
            else if (t == 0 || t == 1 || t == 7) { p += 1; }
            else if (t == 2 || t == 3) { p += 2; }
            else if (t == 4 || t == 5 || t == 6) { p += 4; }
            else if (t == 10 || t == 11 || t == 12) { p += 8; }
            else { fprintf(stderr, "unknown type %u\n", t); exit(1); }
        };
        skip(type);
    };
    auto read_f32_or_f64 = [&](uint32_t t) -> float {
        if (t == 6) return rd_f32();
        if (t == 12) return (float)rd_f64();
        if (t == 4) return (float)rd_u32();
        if (t == 5) { int32_t v; memcpy(&v, p, 4); p += 4; return (float)v; }
        skip_value(t); return 0.0f;
    };

    for (uint64_t ki = 0; ki < n_kv; ++ki) {
        std::string key = rd_str();
        uint32_t type = rd_u32();
        if (key == "llama.block_count") NL = (int)read_scalar_int(type);
        else if (key == "llama.embedding_length") D = (int)read_scalar_int(type);
        else if (key == "llama.feed_forward_length") NFF = (int)read_scalar_int(type);
        else if (key == "llama.attention.head_count") NH = (int)read_scalar_int(type);
        else if (key == "llama.attention.head_count_kv") NHKV = (int)read_scalar_int(type);
        else if (key == "llama.attention.key_length") HD = (int)read_scalar_int(type);
        else if (key == "llama.attention.value_length") { int v = (int)read_scalar_int(type); (void)v; }
        else if (key == "llama.rope.dimension_count") rope_dim = (int)read_scalar_int(type);
        else if (key == "llama.rope.freq_base") rope_base = read_f32_or_f64(type);
        else if (key == "llama.attention.layer_norm_rms_epsilon") rms_eps = read_f32_or_f64(type);
        else if (key == "llama.vocab_size") VOCAB = (int)read_scalar_int(type);
        else if (key == "general.alignment") align = (uint32_t)read_scalar_int(type);
        else if (key == "tokenizer.ggml.bos_token_id") bos = (int)read_scalar_int(type);
        else if (key == "tokenizer.ggml.eos_token_id") eos = (int)read_scalar_int(type);
        else if (key == "tokenizer.ggml.tokens") {
            uint32_t et = rd_u32(); uint64_t n = rd_u64();
            gguf_tokens.resize(n);
            for (uint64_t i = 0; i < n; ++i) {
                if (et == 8) gguf_tokens[i] = rd_str();
                else { skip_value(et); gguf_tokens[i].clear(); }
            }
        }
        else if (key == "tokenizer.ggml.merges") {
            uint32_t et = rd_u32(); uint64_t n = rd_u64();
            gguf_merges.resize(n);
            for (uint64_t i = 0; i < n; ++i) {
                if (et == 8) gguf_merges[i] = rd_str();
                else { skip_value(et); gguf_merges[i].clear(); }
            }
        }
        else if (key == "tokenizer.ggml.token_type") {
            uint32_t et = rd_u32(); uint64_t n = rd_u64();
            token_types.resize(n);
            for (uint64_t i = 0; i < n; ++i) token_types[i] = (int)read_scalar_int(et);
        }
        else skip_value(type);
    }
    if (!VOCAB) VOCAB = (int)gguf_tokens.size();
    if (!rope_dim) rope_dim = HD;
    if (NHKV <= 0) NHKV = NH;
    fprintf(stderr, "dims: L=%d D=%d FFN=%d H=%d KV=%d HD=%d rope=%d base=%g eps=%g vocab=%d bos=%d eos=%d align=%u\n",
            NL, D, NFF, NH, NHKV, HD, rope_dim, rope_base, rms_eps, VOCAB, bos, eos, align);
    if (NL <= 0 || D <= 0 || NFF <= 0 || NH <= 0 || HD <= 0) { fprintf(stderr, "bad dims\n"); return 1; }
    if (rope_dim != HD) fprintf(stderr, "WARNING: rope_dim %d != HD %d (partial rotation)\n", rope_dim, HD);

    // tensor infos
    struct GT { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t offset; };
    std::vector<GT> gtensors;
    for (uint64_t ti = 0; ti < n_tensors; ++ti) {
        GT t;
        t.name = rd_str();
        uint32_t nd = rd_u32();
        t.dims.resize(nd);
        for (uint32_t i = 0; i < nd; ++i) t.dims[i] = rd_u64();
        t.type = rd_u32();
        t.offset = rd_u64();
        gtensors.push_back(t);
    }
    size_t pos0 = (size_t)(p - filebase);
    size_t data_start = (pos0 + align - 1) / align * align;
    fprintf(stderr, "header %zu bytes, data at %zu\n", pos0, data_start);

    std::map<std::string, Buf> codebufs, scalebufs, f32bufs;
    std::map<std::string, std::pair<int, int>> tinfo;
    std::vector<float> rope_ff;   // rope_freqs.weight: llama3 rope-scaling factors

    for (auto& gt : gtensors) {
        uint64_t ne0 = gt.dims.empty() ? 0 : gt.dims[0];
        uint64_t rows = 1;
        for (size_t i = 1; i < gt.dims.size(); ++i) rows *= gt.dims[i];
        int r = (int)rows, c = (int)ne0;
        const unsigned char* data = filebase + data_start + gt.offset;
        if (gt.type == 8) {   // Q8_0: 34-byte blocks, fp16 d first then 32 int8
            int bpr = c / 32;
            size_t nb = (size_t)r * bpr;
            std::vector<unsigned char> codes((size_t)r * (size_t)c);
            std::vector<uint32_t> scales(nb);
            for (size_t bi = 0; bi < nb; ++bi) {
                const unsigned char* blk = data + bi * 34;
                uint16_t d; memcpy(&d, blk, 2);
                scales[bi] = (uint32_t)d;
                memcpy(codes.data() + bi * 32, blk + 2, 32);
            }
            Buf cb = mkbuf((size_t)r * (size_t)c, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&cb, codes.data(), codes.size());
            Buf sb = mkbuf(nb * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&sb, scales.data(), nb * 4);
            codebufs[gt.name] = cb;
            scalebufs[gt.name] = sb;
            tinfo[gt.name] = std::make_pair(r, c);
        } else if (gt.type == 0) {   // F32
            size_t n = (size_t)r * (size_t)c;
            Buf f = mkbuf(n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&f, data, n * 4);
            f32bufs[gt.name] = f;
            tinfo[gt.name] = std::make_pair(r, c);
            if (gt.name == "rope_freqs.weight") rope_ff.assign((const float*)data, (const float*)data + n);
        } else if (gt.type == 1) {   // F16 -> F32
            size_t n = (size_t)r * (size_t)c;
            std::vector<float> f32(n);
            const uint16_t* src = (const uint16_t*)data;
            for (size_t i = 0; i < n; ++i) {
                uint32_t h = src[i];
                uint32_t sign = (h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, fb;
                if (exp == 0) { if (mant == 0) fb = sign; else { int e = 1; while (!(mant & 0x400)) { mant <<= 1; --e; } mant &= 0x3FF; fb = sign | ((uint32_t)(e + 127 - 15) << 23) | (mant << 13); } }
                else if (exp == 31) fb = sign | 0x7F800000u | (mant << 13);
                else fb = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                memcpy(&f32[i], &fb, 4);
            }
            Buf f = mkbuf(n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&f, f32.data(), n * 4);
            f32bufs[gt.name] = f;
            tinfo[gt.name] = std::make_pair(r, c);
        } else {
            fprintf(stderr, "unsupported tensor type %u for %s\n", gt.type, gt.name.c_str());
            return 1;
        }
    }
    fprintf(stderr, "weights uploaded: %zu Q8_0 + %zu f32\n", codebufs.size(), f32bufs.size());

    std::string shdir = "shaders";
    Pipe pq8q   = mkpipe((shdir + "/q8q.comp.spv").c_str(), 3, 4);
    Pipe pq8g   = mkpipe((shdir + "/q8gemv.comp.spv").c_str(), 5, 8);
    Pipe pq8d   = mkpipe((shdir + "/q8deq.comp.spv").c_str(), 3, 8);
    Pipe prmsn  = mkpipe((shdir + "/rmsn.comp.spv").c_str(), 3, 8);
    Pipe prope  = mkpipe((shdir + "/rope_llama.comp.spv").c_str(), 4, 8);
    Pipe psilu  = mkpipe((shdir + "/silu_mul.comp.spv").c_str(), 3, 4);
    Pipe padd   = mkpipe((shdir + "/addv.comp.spv").c_str(), 3, 4);
    Pipe psq    = mkpipe((shdir + "/attn_sq.comp.spv").c_str(), 3, 16);
    Pipe pvq    = mkpipe((shdir + "/attn_vq.comp.spv").c_str(), 3, 16);
    Pipe psoft  = mkpipe((shdir + "/softmax_s.comp.spv").c_str(), 1, 8);
    Pipe pstore = mkpipe((shdir + "/store_h.comp.spv").c_str(), 2, 4);
    Pipe pargmax= mkpipe((shdir + "/argmax.comp.spv").c_str(), 2, 4);
    fprintf(stderr, "pipelines created\n");

    const int ctx = 2048;
    const int GQA = NH / NHKV;
    const int QDIM = NH * HD, KVDIM = NHKV * HD;

    auto Z = [&](int n) {
        return mkbuf((size_t)n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };
    Buf h = Z(D), h2 = Z(D), xp = Z(D), res = Z(D);
    Buf qn = Z(QDIM), qr = Z(QDIM), kn = Z(KVDIM), kr = Z(KVDIM), vv = Z(KVDIM), att = Z(QDIM);
    Buf ffg = Z(NFF), ffu = Z(NFF), ffx = Z(NFF);
    Buf s = Z(NH * ctx);
    Buf logits = Z(VOCAB);
    int maxC = std::max(std::max(D, NFF), std::max(QDIM, KVDIM));
    Buf xqc = mkbuf((size_t)maxC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf xqs = mkbuf((size_t)(maxC / 32) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf aout = mkbuf(4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buf loghost = mkbuf((size_t)VOCAB * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    std::vector<Buf> Kc, Vc;
    for (int i = 0; i < NL; ++i) {
        Kc.push_back(mkbuf((size_t)ctx * KVDIM * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        Vc.push_back(mkbuf((size_t)ctx * KVDIM * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    }
    for (auto& b : Kc) fillbuf(b);
    for (auto& b : Vc) fillbuf(b);

    std::vector<float> cosv((size_t)ctx * (rope_dim / 2)), sinv((size_t)ctx * (rope_dim / 2));

    // ---- rope tables, ggml accumulation order (float32 theta) ----
    // ggml_rope_cache_init: theta starts at pos and multiplies by theta_scale
    // each step; the applied angle is theta/ff where ff is the llama3
    // rope-scaling factor from rope_freqs.weight (1.0 when absent).
    bool use_rope_ff = !rope_ff.empty() && !no_rope_ff;
    if (!rope_ff.empty() && dump) {
        fprintf(stderr, "rope_freqs:");
        for (size_t i = 0; i < rope_ff.size(); ++i) fprintf(stderr, " %g", rope_ff[i]);
        fprintf(stderr, "\n");
    }
    {
        float theta_scale = powf(rope_base, -2.0f / (float)rope_dim);
        for (int pos = 0; pos < ctx; ++pos) {
            float theta = (float)pos;
            for (int i = 0; i < rope_dim / 2; ++i) {
                float ff = (use_rope_ff && i < (int)rope_ff.size()) ? rope_ff[i] : 1.0f;
                float th = theta / ff;
                cosv[(size_t)pos * (rope_dim / 2) + i] = cosf(th);
                sinv[(size_t)pos * (rope_dim / 2) + i] = sinf(th);
                theta *= theta_scale;
            }
        }
    }
    fprintf(stderr, "rope: base=%g freq_factors=%s (%zu entries)\n",
            rope_base, use_rope_ff ? "applied" : "none", rope_ff.size());
    Buf cost = mkbuf(cosv.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    upload(&cost, cosv.data(), cosv.size() * 4);
    Buf sint = mkbuf(sinv.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    upload(&sint, sinv.data(), sinv.size() * 4);

    auto readbuf = [&](Buf& b, int n) -> std::vector<float> {
        Buf t = mkbuf((VkDeviceSize)n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        copybuf(b, t, (size_t)n * 4);
        std::vector<float> v((float*)t.mapped, (float*)t.mapped + n);
        freebuf(t); return v;
    };
    auto dumpv = [&](const char* tag, Buf& b, int n) {
        std::vector<float> v = readbuf(b, n);
        fprintf(stderr, "G %s: %.6f %.6f %.6f\n", tag, v[0], v[1], v[2]);
    };

    // ---- Q8_0 GEMV: quantize the activation row, then int8 dot ----
    auto gemv = [&](const std::string& wname, Buf& xbuf, Buf& outbuf) {
        auto it = tinfo.find(wname);
        if (it == tinfo.end()) { fprintf(stderr, "missing tensor %s\n", wname.c_str()); exit(1); }
        int rows = it->second.first, cols = it->second.second;
        int bpr = cols / 32;
        {
            std::vector<Buf> bufs = {xbuf, xqc, xqs};
            bindbufs(pq8q, bufs);
            struct { int C; } pc{cols};
            dispatch(pq8q, G.q0, G.cp, (uint32_t)bpr, 1, 1, &pc, 4);
        }
        {
            std::vector<Buf> bufs = {codebufs[wname], scalebufs[wname], xqc, xqs, outbuf};
            bindbufs(pq8g, bufs);
            struct { int R; int C; } pc{rows, cols};
            dispatch(pq8g, G.q0, G.cp, (uint32_t)rows, 1, 1, &pc, 8);
        }
    };
    auto rmsn = [&](Buf& x, const std::string& wname, Buf& outbuf) {
        std::vector<Buf> bufs = {x, f32bufs[wname], outbuf};
        bindbufs(prmsn, bufs);
        struct { int n; float eps; } pc{D, rms_eps};
        dispatch(prmsn, G.q0, G.cp, 1, 1, 1, &pc, 8);
    };
    auto embed = [&](int tok) {
        std::vector<Buf> bufs = {codebufs["token_embd.weight"], scalebufs["token_embd.weight"], h};
        bindbufs(pq8d, bufs);
        struct { int row; int C; } pc{tok, D};
        dispatch(pq8d, G.q0, G.cp, 1, 1, 1, &pc, 8);
        if (dump) dumpv("embed_h", h, D);
    };
    auto rope = [&](Buf& in, Buf& outb, int rows, int pos) {
        std::vector<Buf> bufs = {in, cost, sint, outb};
        std::vector<VkDeviceSize> offs = {0, (VkDeviceSize)pos * (rope_dim / 2) * 4,
                                          (VkDeviceSize)pos * (rope_dim / 2) * 4, 0};
        bindbufs(prope, bufs, offs);
        struct { int HD; int neox; } pc{HD, neox};
        dispatch(prope, G.q0, G.cp, (uint32_t)rows, 1, 1, &pc, 8);
    };
    auto addv = [&](Buf& a, Buf& b, Buf& outb, int n) {
        std::vector<Buf> bufs = {a, b, outb};
        bindbufs(padd, bufs);
        struct { int n; } pc{n};
        dispatch(padd, G.q0, G.cp, (uint32_t)((n + 255) / 256), 1, 1, &pc, 4);
    };

    const float kq_scale = 1.0f / sqrtf((float)HD);

    // ---- attention block for layer il at position pos ----
    auto attn = [&](int il, int pos) {
        std::string pfx = "blk." + std::to_string(il) + ".";
        rmsn(h, pfx + "attn_norm.weight", xp);
        if (dump && pos == 0 && il == 0) dumpv("xp_attn", xp, D);
        gemv(pfx + "attn_q.weight", xp, qn);
        gemv(pfx + "attn_k.weight", xp, kn);
        gemv(pfx + "attn_v.weight", xp, vv);
        if (dump && pos == 0 && il == 0) { dumpv("q_raw", qn, QDIM); dumpv("k_raw", kn, KVDIM); dumpv("v_raw", vv, KVDIM); }
        rope(qn, qr, NH, pos);
        rope(kn, kr, NHKV, pos);
        if (dump && pos == 0 && il == 0) { dumpv("q_rope", qr, QDIM); dumpv("k_rope", kr, KVDIM); }
        {
            std::vector<Buf> bufs = {kr, Kc[il]};
            bindbufs(pstore, bufs, {0, (VkDeviceSize)pos * KVDIM * 2});
            struct { int n; } pc{KVDIM};
            dispatch(pstore, G.q0, G.cp, (uint32_t)((KVDIM / 2 + 255) / 256), 1, 1, &pc, 4);
        }
        {
            std::vector<Buf> bufs = {vv, Vc[il]};
            bindbufs(pstore, bufs, {0, (VkDeviceSize)pos * KVDIM * 2});
            struct { int n; } pc{KVDIM};
            dispatch(pstore, G.q0, G.cp, (uint32_t)((KVDIM / 2 + 255) / 256), 1, 1, &pc, 4);
        }
        int n = pos + 1;
        {
            std::vector<Buf> bufs = {Kc[il], qr, s};
            bindbufs(psq, bufs);
            struct { int n; int nh; int nhkv; int hd; } pc{n, NH, NHKV, HD};
            dispatch(psq, G.q0, G.cp, (uint32_t)NHKV, 1, 1, &pc, 16);
        }
        if (dump && pos == 0 && il == 0) dumpv("scores", s, NH);
        {
            std::vector<Buf> bufs = {s};
            bindbufs(psoft, bufs);
            struct { int n; float scale; } pc{n, kq_scale};
            dispatch(psoft, G.q0, G.cp, (uint32_t)NH, 1, 1, &pc, 8);
        }
        if (dump && pos == 0 && il == 0) dumpv("probs", s, NH);
        {
            std::vector<Buf> bufs = {s, Vc[il], att};
            bindbufs(pvq, bufs);
            struct { int n; int nh; int nhkv; int hd; } pc{n, NH, NHKV, HD};
            dispatch(pvq, G.q0, G.cp, (uint32_t)NHKV, 1, 1, &pc, 16);
        }
        if (dump && pos == 0 && il == 0) dumpv("att", att, QDIM);
        gemv(pfx + "attn_output.weight", att, res);
        addv(h, res, h2, D);
        std::swap(h, h2);
    };

    auto ffn = [&](int il) {
        std::string pfx = "blk." + std::to_string(il) + ".";
        rmsn(h, pfx + "ffn_norm.weight", xp);
        gemv(pfx + "ffn_gate.weight", xp, ffg);
        gemv(pfx + "ffn_up.weight", xp, ffu);
        {
            std::vector<Buf> bufs = {ffg, ffu, ffx};
            bindbufs(psilu, bufs);
            struct { int n; } pc{NFF};
            dispatch(psilu, G.q0, G.cp, (uint32_t)((NFF + 255) / 256), 1, 1, &pc, 4);
        }
        gemv(pfx + "ffn_down.weight", ffx, res);
        addv(h, res, h2, D);
        std::swap(h, h2);
    };

    auto body = [&](int tok, int pos) {
        embed(tok);
        for (int il = 0; il < NL; ++il) {
            attn(il, pos);
            ffn(il);
            if (dump && pos == 0) {
                std::vector<float> hv = readbuf(h, 3);
                fprintf(stderr, "G L%02d res %.6f %.6f %.6f\n", il, hv[0], hv[1], hv[2]);
            }
        }
    };

    auto step = [&](int tok, int pos) -> int {
        body(tok, pos);
        rmsn(h, "output_norm.weight", xp);
        if (dump) dumpv("xp_out", xp, D);
        gemv("token_embd.weight", xp, logits);
        if (temperature <= 0.0f) {
            std::vector<Buf> bufs = {logits, aout};
            bindbufs(pargmax, bufs);
            struct { int n; } pc{VOCAB};
            dispatch(pargmax, G.q0, G.cp, 1, 1, 1, &pc, 4);
            int* r = (int*)aout.mapped;
            if (dump) {
                std::vector<float> lg = readbuf(logits, VOCAB);
                std::vector<int> idx(VOCAB);
                for (int i = 0; i < VOCAB; ++i) idx[i] = i;
                std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                                  [&](int a, int b) { return lg[a] > lg[b]; });
                fprintf(stderr, "G top5:");
                for (int i = 0; i < 5; ++i) fprintf(stderr, " %d:%.5f", idx[i], lg[idx[i]]);
                fprintf(stderr, "\n");
            }
            return r[0];
        }
        copybuf(logits, loghost, (size_t)VOCAB * 4);
        float* L = (float*)loghost.mapped;
        for (int i = 0; i < VOCAB; ++i) L[i] /= temperature;
        if (topk > 0 && topk < VOCAB) {
            std::vector<float> tmp(L, L + VOCAB);
            std::nth_element(tmp.begin(), tmp.begin() + (VOCAB - topk), tmp.end());
            float thr = tmp[VOCAB - topk];
            for (int i = 0; i < VOCAB; ++i) if (L[i] < thr) L[i] = -1e30f;
        }
        float mx = -1e30f;
        for (int i = 0; i < VOCAB; ++i) if (L[i] > mx) mx = L[i];
        double sum = 0;
        for (int i = 0; i < VOCAB; ++i) { L[i] = expf(L[i] - mx); sum += L[i]; }
        for (int i = 0; i < VOCAB; ++i) L[i] /= (float)sum;
        double r = (double)rand() / RAND_MAX, acc = 0;
        for (int i = 0; i < VOCAB; ++i) { acc += L[i]; if (r < acc) return i; }
        return VOCAB - 1;
    };

    // ---- tokenizer ----
    kindi_llama_tok::LlamaTokenizer tok;
    std::string tokerr;
    size_t specials_first = gguf_tokens.size();
    for (size_t i = 0; i < token_types.size(); ++i)
        if (token_types[i] != 1) { specials_first = i; break; }
    fprintf(stderr, "specials start at %zu\n", specials_first);
    if (!tok.load_from_strings(gguf_tokens, gguf_merges, specials_first, &tokerr)) {
        fprintf(stderr, "tokenizer load failed: %s\n", tokerr.c_str()); return 1;
    }
    std::string promptstr = prompt ? prompt : "The sky is blue because";
    std::vector<int> ids = tok.encode(promptstr);
    if (ids.empty()) ids.push_back(0);
    if (bos >= 0) ids.insert(ids.begin(), bos);
    fprintf(stderr, "prompt ids (%zu):", ids.size());
    for (int t : ids) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    for (size_t i = 0; i + 1 < ids.size(); ++i) body(ids[i], (int)i);
    int pos = (int)ids.size();
    std::vector<int> out;
    // the last prompt token sits at position pos-1, so the first generated token
    // occupies position pos, the next pos+1, and so on
    int next = step(ids.back(), pos - 1);
    out.push_back(next);
    while ((int)out.size() < npredict) {
        next = step(out.back(), pos);
        out.push_back(next);
        pos++;
        if (pos >= ctx) break;
    }
    fprintf(stderr, "\nGenerated %zu tokens:", out.size());
    for (int t : out) fprintf(stderr, " %d", t);
    std::string decoded = tok.decode(out);
    fprintf(stderr, "\nDecoded: %s\n", decoded.c_str());
    if (outpath) {
        std::ofstream of(outpath, std::ios::binary);
        of.write(decoded.data(), (std::streamsize)decoded.size());
    }

    ktrace::dump();
    return 0;
}
