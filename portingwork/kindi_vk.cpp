// kindi-vk: native Vulkan compute host
#include <vulkan/vulkan.h>
#include <windows.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); exit(1); } } while(0)

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint32_t> buf(sz / 4);
    f.read((char*)buf.data(), sz);
    return buf;
}

struct DeviceCtx {
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue q_compute0, q_compute1, q_transfer;
    uint32_t cfam, tfam;
    VkCommandPool cpool, tpool;
    VkPhysicalDeviceMemoryProperties mprops;
    VkPhysicalDeviceProperties props;
};
static DeviceCtx g_dev;

static uint32_t find_mem_type(uint32_t bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < g_dev.mprops.memoryTypeCount; ++i)
        if ((bits >> i) & 1u)
            if ((g_dev.mprops.memoryTypes[i].propertyFlags & flags) == flags)
                return i;
    fprintf(stderr, "no mem type flags=%u\n", (unsigned)flags); exit(1);
}

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

static Buf create_buf(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mflags) {
    Buf b; b.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
    uint32_t qfis[2] = {g_dev.cfam, g_dev.tfam};
    ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = qfis;
    VK_CHECK(vkCreateBuffer(g_dev.dev, &ci, nullptr, &b.buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(g_dev.dev, b.buf, &req);
    uint32_t mt = find_mem_type((uint32_t)req.memoryTypeBits, mflags);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = mt;
    VK_CHECK(vkAllocateMemory(g_dev.dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(g_dev.dev, b.buf, b.mem, 0));
    if (mflags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        VK_CHECK(vkMapMemory(g_dev.dev, b.mem, 0, size, 0, &b.mapped));
    return b;
}
static void destroy_buf(Buf& b) {
    if (b.mapped) vkUnmapMemory(g_dev.dev, b.mem);
    if (b.buf) vkDestroyBuffer(g_dev.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(g_dev.dev, b.mem, nullptr);
    b = {};
}

static void init_device() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "kindi-vk"; app.applicationVersion = 1;
    app.pEngineName = "kindi-vk"; app.engineVersion = 1;
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &g_dev.inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(g_dev.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(g_dev.inst, &n, devs.data());
    g_dev.phys = devs[0];
    vkGetPhysicalDeviceProperties(g_dev.phys, &g_dev.props);
    vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &g_dev.mprops);
    fprintf(stderr, "GPU: %s\n", g_dev.props.deviceName);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(g_dev.phys, &qn, qf.data());
    g_dev.cfam = UINT32_MAX; g_dev.tfam = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i) {
        if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].queueCount >= 2 && g_dev.cfam == UINT32_MAX)
            g_dev.cfam = i;
        if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && g_dev.tfam == UINT32_MAX && i != g_dev.cfam)
            g_dev.tfam = i;
    }
    assert(g_dev.cfam != UINT32_MAX && g_dev.tfam != UINT32_MAX);
    fprintf(stderr, "compute family=%u transfer family=%u\n", g_dev.cfam, g_dev.tfam);
    std::vector<VkDeviceQueueGlobalPriorityCreateInfoKHR> prio_chains;
    prio_chains.reserve(4);
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for (int qi = 0; qi < 2; ++qi) {
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = g_dev.cfam; qci.queueCount = 1;
        static float prio = 1.0f;
        qci.pQueuePriorities = &prio;
        uint32_t en = 0; vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &en, nullptr);
        std::vector<VkExtensionProperties> exts(en);
        vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &en, exts.data());
        bool has_prio = false;
        for (auto& e : exts) if (strcmp(e.extensionName, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) == 0) has_prio = true;
        if (has_prio) {
            VkDeviceQueueGlobalPriorityCreateInfoKHR ch{VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR};
            ch.globalPriority = (qi == 0) ? VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR : VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
            prio_chains.push_back(ch);
            qci.pNext = &prio_chains.back();
        }
        qcis.push_back(qci);
    }
    {
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = g_dev.tfam; qci.queueCount = 1;
        static float prio = 1.0f;
        qci.pQueuePriorities = &prio;
        qcis.push_back(qci);
    }
    VkPhysicalDeviceFeatures feats{};
    feats.shaderFloat64 = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = (uint32_t)qcis.size();
    dci.pQueueCreateInfos = qcis.data();
    dci.pEnabledFeatures = &feats;
    uint32_t en = 0; vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(g_dev.phys, nullptr, &en, exts.data());
    bool has_prio = false;
    for (auto& e : exts) if (strcmp(e.extensionName, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) == 0) has_prio = true;
    const char* extname = VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME;
    if (has_prio) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &extname; }
    VK_CHECK(vkCreateDevice(g_dev.phys, &dci, nullptr, &g_dev.dev));
    fprintf(stderr, "device created OK\n");
    vkGetDeviceQueue(g_dev.dev, g_dev.cfam, 0, &g_dev.q_compute0);
    vkGetDeviceQueue(g_dev.dev, g_dev.cfam, 1, &g_dev.q_compute1);
    vkGetDeviceQueue(g_dev.dev, g_dev.tfam, 0, &g_dev.q_transfer);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = g_dev.cfam; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(g_dev.dev, &cpci, nullptr, &g_dev.cpool));
    cpci.queueFamilyIndex = g_dev.tfam;
    VK_CHECK(vkCreateCommandPool(g_dev.dev, &cpci, nullptr, &g_dev.tpool));
}

static void upload_one(Buf* dst, const void* data, size_t nbytes) {
    Buf st = create_buf(nbytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(st.mapped, data, nbytes);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_dev.cpool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(g_dev.dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy cp{0, 0, (VkDeviceSize)nbytes};
    vkCmdCopyBuffer(cb, st.buf, dst->buf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; VK_CHECK(vkCreateFence(g_dev.dev, &fi, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(g_dev.q_compute0, 1, &si, fence));
    VK_CHECK(vkWaitForFences(g_dev.dev, 1, &fence, VK_TRUE, 60000000000ULL));
    vkDestroyFence(g_dev.dev, fence, nullptr);
    vkFreeCommandBuffers(g_dev.dev, g_dev.tpool, 1, &cb);
    destroy_buf(st);
}

static void test_g4() {
    int R = 64, C = 512;
    std::vector<int8_t> W(R * C);
    uint32_t state = 12345;
    auto rnd = [&](){ state = state * 1664525u + 1013904223u; return (int)(state >> 16) & 0x7fff; };
    for (int i = 0; i < R * C; ++i) W[i] = (int8_t)((rnd() % 3) - 1);
    std::vector<uint32_t> codes(R * (C / 4));
    for (int r = 0; r < R; ++r)
        for (int k = 0; k < C; k += 4) {
            uint32_t b = (uint32_t)(W[r*C+k]+1) + 3u*(uint32_t)(W[r*C+k+1]+1)
                       + 9u*(uint32_t)(W[r*C+k+2]+1) + 27u*(uint32_t)(W[r*C+k+3]+1);
            codes[r*(C/4) + k/4] = b;
        }
    std::vector<uint32_t> scales(R * (C / 128));
    for (int i = 0; i < (int)scales.size(); ++i) scales[i] = 0x3C00u;
    std::vector<float> x(C);
    for (int i = 0; i < C; ++i) x[i] = (float)(rnd() % 100) / 100.0f - 0.5f;
    std::vector<float> ref(R, 0.0f);
    for (int r = 0; r < R; ++r)
        for (int k = 0; k < C; ++k)
            ref[r] += (float)W[r*C+k] * x[k];

    VkBufferUsageFlags su = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    Buf codes_b = create_buf(codes.size()*4, su, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buf scales_b = create_buf(scales.size()*4, su, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buf x_b = create_buf(x.size()*4, su, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buf y_b = create_buf(R*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(codes_b.mapped, codes.data(), codes.size()*4);
    memcpy(scales_b.mapped, scales.data(), scales.size()*4);
    memcpy(x_b.mapped, x.data(), x.size()*4);
    fprintf(stderr, "CPU x[0]=%.4f  GPU x_b.mapped[0]=%.4f\n", x[0], ((float*)x_b.mapped)[0]); fflush(stderr);

    auto code = read_spv("shaders/g4simple.comp.spv");
    fprintf(stderr, "loaded spv: %zu words\n", code.size()); fflush(stderr);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size()*4; smci.pCode = code.data();
    VkShaderModule mod; VK_CHECK(vkCreateShaderModule(g_dev.dev, &smci, nullptr, &mod));
    VkDescriptorSetLayoutBinding binds[4];
    for (int i = 0; i < 4; ++i) { binds[i].binding=i; binds[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binds[i].descriptorCount=1; binds[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount=4; dslci.pBindings=binds;
    VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(g_dev.dev, &dslci, nullptr, &dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,8};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(g_dev.dev, &plci, nullptr, &pl));
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; stage.module=mod; stage.pName="main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage=stage; cpci.layout=pl;
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(g_dev.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&ps;
    VkDescriptorPool pool; VK_CHECK(vkCreateDescriptorPool(g_dev.dev, &dpci, nullptr, &pool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=pool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet dset; VK_CHECK(vkAllocateDescriptorSets(g_dev.dev, &dsai, &dset));
    Buf bufs[4] = {codes_b, scales_b, x_b, y_b};
    fprintf(stderr, "buf handles: codes=%p scales=%p x=%p y=%p\n",
        (void*)codes_b.buf, (void*)scales_b.buf, (void*)x_b.buf, (void*)y_b.buf); fflush(stderr);
    VkDescriptorBufferInfo infos[4];
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; ++i) {
        infos[i].buffer=bufs[i].buf; infos[i].offset=0; infos[i].range=bufs[i].size;
        writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext=nullptr;
        writes[i].dstSet=dset; writes[i].dstBinding=i; writes[i].dstArrayElement=0; writes[i].descriptorCount=1;
        writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pImageInfo=nullptr;
        writes[i].pBufferInfo=&infos[i];
        writes[i].pTexelBufferView=nullptr;
    }
    vkUpdateDescriptorSets(g_dev.dev, 4, writes, 0, nullptr);

    struct { int R, C; } pc{R, C};
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool=g_dev.cpool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(g_dev.dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, R, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; VK_CHECK(vkCreateFence(g_dev.dev, &fi, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount=1; si.pCommandBuffers=&cb;
    VK_CHECK(vkQueueSubmit(g_dev.q_compute0, 1, &si, fence));
    VK_CHECK(vkWaitForFences(g_dev.dev, 1, &fence, VK_TRUE, 60000000000ULL));
    fprintf(stderr, "fence waited\n"); fflush(stderr);
    vkDestroyFence(g_dev.dev, fence, nullptr);
    vkFreeCommandBuffers(g_dev.dev, g_dev.cpool, 1, &cb);

    float* got = (float*)y_b.mapped;
    fprintf(stderr, "y_b[0]=%.4f\n", got[0]); fflush(stderr);
    float maxerr = 0;
    for (int i = 0; i < R; ++i) {
        float e = fabsf(got[i] - ref[i]);
        if (e > maxerr) maxerr = e;
    }
    fprintf(stderr, "max|got-ref| = %.6f\n", maxerr);
    fprintf(stderr, "ref[0..3] = %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
    fprintf(stderr, "got[0..3] = %.4f %.4f %.4f %.4f\n", got[0], got[1], got[2], got[3]);
    fprintf(stderr, "%s\n", maxerr < 1e-3f ? "PASS" : "FAIL");

    destroy_buf(codes_b); destroy_buf(scales_b); destroy_buf(x_b); destroy_buf(y_b);
}

int main(int argc, char** argv) {
    init_device();
    if (argc > 1 && strcmp(argv[1], "--test-g4") == 0) {
        test_g4();
        return 0;
    }
    fprintf(stderr, "kindi-vk host ready (use --test-g4)\n");
    return 0;
}
