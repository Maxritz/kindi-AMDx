// kindi-vk: complete 2B decode engine - ALL shaders wired, full decode loop
#include <vulkan/vulkan.h>
// winsock2.h must come before windows.h: windows.h otherwise pulls in the 1.x winsock.h
// and the two sets of declarations collide (SourceCount / PIP_MSFILTER / ...).
#include <winsock2.h>
// NOMINMAX: windows.h otherwise defines min/max as macros, which breaks std::min.
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
#include <ctime>
#include <algorithm>
#include <set>
#include <cmath>
#include <map>
#include <unordered_map>
#include <functional>
#include <deque>
#include <nlohmann/json.hpp>
#include "tokenizer.h"
#include "trace.h"
#include "http.h"
using json = nlohmann::json;

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){fprintf(stderr,"Vulkan error %d at %s:%d\n",(int)_r,__FILE__,__LINE__);exit(1);} } while(0)

// ---- device + buffer + pipeline helpers ----
struct DeviceCtx { VkInstance inst; VkPhysicalDevice phys; VkDevice dev; VkQueue q0,q1,qt; uint32_t cf,tf; VkCommandPool cp,tp; VkPhysicalDeviceMemoryProperties mp; VkPhysicalDeviceProperties pp; };
static DeviceCtx G;
static uint32_t findmt(uint32_t bits, VkMemoryPropertyFlags f){ for(uint32_t i=0;i<G.mp.memoryTypeCount;++i) if(((bits>>i)&1)&&((G.mp.memoryTypes[i].propertyFlags&f)==f)) return i; fprintf(stderr,"nomemtype\n");exit(1);}
struct Buf { VkBuffer buf=VK_NULL_HANDLE; VkDeviceMemory mem=VK_NULL_HANDLE; VkDeviceSize size=0; void* mapped=nullptr; };
// live device allocation total: the context fit needs to know what the weights actually
// took, which is only known after they are uploaded
static size_t g_alloc_total = 0;
static Buf mkbuf(VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags f){
    Buf b; b.size=sz; VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=sz; ci.usage=u; ci.sharingMode=VK_SHARING_MODE_CONCURRENT; uint32_t qf[2]={G.cf,G.tf}; ci.queueFamilyIndexCount=2; ci.pQueueFamilyIndices=qf;
    VK_CHECK(vkCreateBuffer(G.dev,&ci,nullptr,&b.buf));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(G.dev,b.buf,&r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=r.size; ai.memoryTypeIndex=findmt((uint32_t)r.memoryTypeBits,f);
    g_alloc_total += r.size;
    if(getenv("KINDI_MEMLOG")) fprintf(stderr,"alloc %zu (total %.2f GB) type %u\n", (size_t)r.size, g_alloc_total/1073741824.0, ai.memoryTypeIndex);
    VK_CHECK(vkAllocateMemory(G.dev,&ai,nullptr,&b.mem)); VK_CHECK(vkBindBufferMemory(G.dev,b.buf,b.mem,0));
    if(f&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VK_CHECK(vkMapMemory(G.dev,b.mem,0,sz,0,&b.mapped));
    return b;
}
// Allocation that reports failure instead of exiting. Optional buffers (the prefix-cache
// checkpoint) must degrade to "feature off" on an 8 GB card, not kill the server.
static bool mkbuf_try(VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags f, Buf& out){
    out = Buf{};
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=sz; ci.usage=u;
    ci.sharingMode=VK_SHARING_MODE_CONCURRENT;
    uint32_t qf[2]={G.cf,G.tf}; ci.queueFamilyIndexCount=2; ci.pQueueFamilyIndices=qf;
    if(vkCreateBuffer(G.dev,&ci,nullptr,&out.buf)!=VK_SUCCESS){ out=Buf{}; return false; }
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(G.dev,out.buf,&r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize=r.size;
    uint32_t mt=UINT32_MAX;
    for(uint32_t i=0;i<G.mp.memoryTypeCount;++i)
        if(((r.memoryTypeBits>>i)&1) && ((G.mp.memoryTypes[i].propertyFlags&f)==f)){ mt=i; break; }
    if(mt==UINT32_MAX){ vkDestroyBuffer(G.dev,out.buf,nullptr); out=Buf{}; return false; }
    ai.memoryTypeIndex=mt;
    if(vkAllocateMemory(G.dev,&ai,nullptr,&out.mem)!=VK_SUCCESS){
        vkDestroyBuffer(G.dev,out.buf,nullptr); out=Buf{}; return false;
    }
    vkBindBufferMemory(G.dev,out.buf,out.mem,0);
    out.size=sz;
    g_alloc_total += r.size;
    return true;
}

static void freebuf(Buf&b){ ktrace::count_free(); if(b.size && g_alloc_total>=b.size) g_alloc_total-=b.size; if(b.mapped)vkUnmapMemory(G.dev,b.mem); if(b.buf)vkDestroyBuffer(G.dev,b.buf,nullptr); if(b.mem)vkFreeMemory(G.dev,b.mem,nullptr); b={}; }
static void initdev(){
    VkApplicationInfo a{VK_STRUCTURE_TYPE_APPLICATION_INFO}; a.pApplicationName="kindi-vk"; a.applicationVersion=1; a.pEngineName="kindi-vk"; a.engineVersion=1; a.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&a; VK_CHECK(vkCreateInstance(&ici,nullptr,&G.inst));
    uint32_t n=0; vkEnumeratePhysicalDevices(G.inst,&n,nullptr); std::vector<VkPhysicalDevice> ds(n); vkEnumeratePhysicalDevices(G.inst,&n,ds.data());
    if(n==0){ fprintf(stderr,"no Vulkan device\n"); exit(1); }
    // ---- device selection ----
    // Never take ds[0] blindly: on a machine with an iGPU, a remote-display adapter or a
    // second card, index 0 is not necessarily the card we want. Prefer the RX 580 by name
    // (KINDI_DEVICE env or --device N overrides), then any discrete AMD, then the first.
    const char* want = getenv("KINDI_DEVICE");
    int chosen = -1;
    {
        std::vector<VkPhysicalDeviceProperties> all(n);
        for(uint32_t i=0;i<n;++i) vkGetPhysicalDeviceProperties(ds[i],&all[i]);
        for(uint32_t i=0;i<n;++i)
            fprintf(stderr,"device %u: %s (%s%s)\n", i, all[i].deviceName,
                    all[i].deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?"discrete":"other",
                    all[i].vendorID==0x1002?", AMD":"");
        if(want && *want){
            if(isdigit((unsigned char)want[0])) chosen=atoi(want);
            else for(uint32_t i=0;i<n;++i) if(strstr(all[i].deviceName,want)){ chosen=(int)i; break; }
        }
        if(chosen<0) for(uint32_t i=0;i<n;++i) if(strstr(all[i].deviceName,"RX 580")){ chosen=(int)i; break; }
        if(chosen<0) for(uint32_t i=0;i<n;++i) if(all[i].vendorID==0x1002 &&
                        all[i].deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){ chosen=(int)i; break; }
        if(chosen<0) for(uint32_t i=0;i<n;++i) if(all[i].deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){ chosen=(int)i; break; }
        if(chosen<0) chosen=0;
        if(chosen>=(int)n) chosen=0;
        fprintf(stderr,"using device %d: %s\n", chosen, all[chosen].deviceName);
    }
    G.phys=ds[chosen];
    vkGetPhysicalDeviceProperties(G.phys,&G.pp); vkGetPhysicalDeviceMemoryProperties(G.phys,&G.mp);
    for(uint32_t hi=0; hi<G.mp.memoryHeapCount; ++hi){
        fprintf(stderr,"heap %u: %.2f GB %s\n", hi, G.mp.memoryHeaps[hi].size/1073741824.0,
                (G.mp.memoryHeaps[hi].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "device-local" : "host");
    } fprintf(stderr,"GPU: %s\n",G.pp.deviceName);
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(G.phys,&qn,nullptr); std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(G.phys,&qn,qf.data());
    G.cf=UINT32_MAX; G.tf=UINT32_MAX;
    for(uint32_t i=0;i<qn;++i){ if(!(qf[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)&&(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT)&&qf[i].queueCount>=2&&G.cf==UINT32_MAX)G.cf=i; if(!(qf[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)&&(qf[i].queueFlags&VK_QUEUE_TRANSFER_BIT)&&G.tf==UINT32_MAX&&i!=G.cf)G.tf=i; }
    assert(G.cf!=UINT32_MAX&&G.tf!=UINT32_MAX);
    // ---- queue priority ----
    // pQueuePriorities is the WITHIN-family priority (1.0 = already maximum).
    // The hardware scheduling class is separate: VK_KHR_global_priority. Vulkan offers no
    // core/CU-affinity control, so this is the strongest lever for "give compute the GPU".
    static float qprio = 1.0f;
    // KINDI_QUEUE_PRIORITY lets the priority request be A/B measured rather than assumed.
    VkQueueGlobalPriorityKHR want_gp = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
    bool gp_pinned=false;
    { const char* e=getenv("KINDI_QUEUE_PRIORITY");
      if(e){ gp_pinned=true;
             if(!strcmp(e,"low")) want_gp=VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;
             else if(!strcmp(e,"medium")) want_gp=VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
             else if(!strcmp(e,"high")) want_gp=VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
             else if(!strcmp(e,"realtime")) want_gp=VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR; } }
    bool have_gp = false;
    {
        uint32_t ec=0; vkEnumerateDeviceExtensionProperties(G.phys,nullptr,&ec,nullptr);
        std::vector<VkExtensionProperties> ex(ec);
        vkEnumerateDeviceExtensionProperties(G.phys,nullptr,&ec,ex.data());
        for(auto&e:ex) if(!strcmp(e.extensionName,"VK_KHR_global_priority")||
                          !strcmp(e.extensionName,"VK_EXT_global_priority")){ have_gp=true; break; }
        // Global priority is a QUEUE FAMILY property, so it must go through the
        // queue-family-properties chain (putting it in DeviceProperties2 silently
        // returns priorityCount = 0 and you stay on MEDIUM).
        PFN_vkGetPhysicalDeviceQueueFamilyProperties2 gqf2=
            (PFN_vkGetPhysicalDeviceQueueFamilyProperties2)vkGetInstanceProcAddr(G.inst,"vkGetPhysicalDeviceQueueFamilyProperties2");
        if(gqf2){
            // vkGetPhysicalDeviceQueueFamilyProperties2(pd, &count, pProps): no family index,
            // it fills an array of all families. Query the count, then read the family we use.
            uint32_t fcount=0;
            gqf2(G.phys,&fcount,nullptr);
            if(fcount>0){
                std::vector<VkQueueFamilyGlobalPriorityPropertiesKHR> gpps(
                    fcount, VkQueueFamilyGlobalPriorityPropertiesKHR{
                        VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR});
                std::vector<VkQueueFamilyProperties2> qfps(fcount, VkQueueFamilyProperties2{
                        VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2});
                for(uint32_t i=0;i<fcount;++i) qfps[i].pNext=&gpps[i];
                gqf2(G.phys,&fcount,qfps.data());
                uint32_t fi = (G.cf < fcount) ? G.cf : 0;
                fprintf(stderr,"queue family %u priority classes:", fi);
                for(uint32_t i=0;i<gpps[fi].priorityCount;++i){
                    VkQueueGlobalPriorityKHR v = gpps[fi].priorities[i];
                    fprintf(stderr," %s", v==VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR?"REALTIME":
                                         v==VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR?"HIGH":
                                         v==VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR?"MEDIUM":"LOW");
                    if(!gp_pinned && v>want_gp) want_gp=v;
                }
                fprintf(stderr,"\n");
            }
        }
        fprintf(stderr,"queue global priority: %s, requesting %s\n",
            have_gp?"supported":"not supported",
            want_gp==VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR?"REALTIME":
            want_gp==VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR?"HIGH":
            want_gp==VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR?"MEDIUM":"LOW");
    }
    VkDeviceQueueGlobalPriorityCreateInfoKHR gpqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR};
    gpqci.globalPriority = want_gp;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for(int qi=0;qi<2;++qi){ VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueFamilyIndex=G.cf; q.queueCount=1; q.pQueuePriorities=&qprio;
        if(have_gp) q.pNext=&gpqci;
        qcis.push_back(q); }
    { VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueFamilyIndex=G.tf; q.queueCount=1; q.pQueuePriorities=&qprio; qcis.push_back(q); }
    VkPhysicalDeviceFeatures f{}; f.shaderFloat64=VK_TRUE;
    std::vector<const char*> devext;
    if(have_gp) devext.push_back("VK_KHR_global_priority");
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=(uint32_t)qcis.size(); dci.pQueueCreateInfos=qcis.data(); dci.pEnabledFeatures=&f;
    dci.enabledExtensionCount=(uint32_t)devext.size(); dci.ppEnabledExtensionNames=devext.empty()?nullptr:devext.data();
    VkResult dcr = vkCreateDevice(G.phys,&dci,nullptr,&G.dev);
    if(dcr != VK_SUCCESS && have_gp){
        fprintf(stderr,"global priority rejected (%d), retrying MEDIUM\n",(int)dcr);
        gpqci.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
        dcr = vkCreateDevice(G.phys,&dci,nullptr,&G.dev);
    }
    if(dcr != VK_SUCCESS){
        fprintf(stderr,"retrying device creation without global priority\n");
        for(auto&q:qcis) q.pNext=nullptr;
        dci.enabledExtensionCount=0; dci.ppEnabledExtensionNames=nullptr;
        VK_CHECK(vkCreateDevice(G.phys,&dci,nullptr,&G.dev));
    }
    vkGetDeviceQueue(G.dev,G.cf,0,&G.q0); vkGetDeviceQueue(G.dev,G.cf,1,&G.q1); vkGetDeviceQueue(G.dev,G.tf,0,&G.qt);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex=G.cf; cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(G.dev,&cpci,nullptr,&G.cp)); cpci.queueFamilyIndex=G.tf; VK_CHECK(vkCreateCommandPool(G.dev,&cpci,nullptr,&G.tp));
    if(ktrace::enabled()) ktrace::init(G.phys,G.dev);
    {   // real compute-core topology (VK_AMD_shader_core_properties). Properties can be
        // queried without enabling the extension.
        VkPhysicalDeviceShaderCorePropertiesAMD sc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD};
        VkPhysicalDeviceVulkan11Properties v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext=&sc; sc.pNext=&v11;   // subgroupSize lives in Vulkan11Properties, not limits
        PFN_vkGetPhysicalDeviceProperties2 gp2=(PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(G.inst,"vkGetPhysicalDeviceProperties2");
        if(gp2){
            gp2(G.phys,&p2);
            uint32_t cu = sc.shaderEngineCount * sc.shaderArraysPerEngineCount * sc.computeUnitsPerShaderArray;
            uint32_t ws = v11.subgroupSize;
            ktrace::set_cores(cu, sc.simdPerComputeUnit, sc.wavefrontsPerSimd, ws);
            fprintf(stderr,"shader cores: %u SE x %u array x %u CU = %u CU, %u SIMD/CU, "
                           "%u wavefronts/SIMD (wave %u)\n",
                    sc.shaderEngineCount, sc.shaderArraysPerEngineCount,
                    sc.computeUnitsPerShaderArray, cu,
                    sc.simdPerComputeUnit, sc.wavefrontsPerSimd, ws);
        }
    }
}
static std::vector<uint32_t> readspv(const char*p){ std::ifstream f(p,std::ios::binary|std::ios::ate); if(!f){fprintf(stderr,"nospv %s\n",p);exit(1);} size_t sz=(size_t)f.tellg(); f.seekg(0); std::vector<uint32_t> b(sz/4); f.read((char*)b.data(),sz); return b; }

struct Pipe { VkShaderModule mod; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe;
             VkDescriptorPool pool; VkDescriptorSet dset;
             const char* name=nullptr; uint32_t nbind=0; uint64_t brange[16]={0}; uint32_t lx=0; };
static Pipe mkpipe(const char* spv, int nbind, int pushsz){
    Pipe P{}; auto code=readspv(spv);
    {   // Derive the push-constant size from the SPIR-V: the largest
        // OpMemberDecorate <PC> <member> Offset <n> plus one word. Never trust the
        // literal passed in - a short range silently feeds the shader garbage.
        uint32_t maxoff = 0;
        for(size_t i=0;i+3<code.size();++i){
            if(code[i]==71){                       // OpDecorate
                // OpDecorate target MemberDecorate? handled below via OpMemberDecorate (72)
            }
            if(code[i]==72 && i+4<code.size()){    // OpMemberDecorate: target, member, decoration, value
                uint32_t decoration = code[i+2];
                if(decoration==35){                // Offset
                    uint32_t off = code[i+3];
                    if(off > maxoff) maxoff = off;
                }
            }
        }
        if(maxoff > 0){
            uint32_t derived = maxoff + 4;
            if(derived > (uint32_t)pushsz){
                fprintf(stderr,"mkpipe %s: push range %d too small, derived %u from SPIR-V\n",
                        spv, pushsz, derived);
                pushsz = (int)derived;
            }
        }
    }
    {   // OpExecutionMode(16) LocalSize(17) x y z  -> the actual workgroup width
        for(size_t i=0;i+5<code.size();++i){
            if(code[i]==16 && code[i+2]==17){ P.lx=code[i+3]; break; }
        }
        if(P.lx==0) P.lx=1;
    }
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize=code.size()*4; sm.pCode=code.data(); VK_CHECK(vkCreateShaderModule(G.dev,&sm,nullptr,&P.mod));
    std::vector<VkDescriptorSetLayoutBinding> binds(nbind);
    for(int i=0;i<nbind;++i){ binds[i].binding=i; binds[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binds[i].descriptorCount=1; binds[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dsl.bindingCount=nbind; dsl.pBindings=binds.data(); VK_CHECK(vkCreateDescriptorSetLayout(G.dev,&dsl,nullptr,&P.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,(uint32_t)pushsz};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount=1; pl.pSetLayouts=&P.dsl; pl.pushConstantRangeCount=1; pl.pPushConstantRanges=&pcr; VK_CHECK(vkCreatePipelineLayout(G.dev,&pl,nullptr,&P.pl));
    VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st.stage=VK_SHADER_STAGE_COMPUTE_BIT; st.module=P.mod; st.pName="main";
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage=st; cpi.layout=P.pl; VK_CHECK(vkCreateComputePipelines(G.dev,VK_NULL_HANDLE,1,&cpi,nullptr,&P.pipe));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,(uint32_t)nbind};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=1; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; VK_CHECK(vkCreateDescriptorPool(G.dev,&dpi,nullptr,&P.pool));
    VkDescriptorSetAllocateInfo ds{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ds.descriptorPool=P.pool; ds.descriptorSetCount=1; ds.pSetLayouts=&P.dsl; VK_CHECK(vkAllocateDescriptorSets(G.dev,&ds,&P.dset));
    P.nbind=(uint32_t)nbind;
    {   // shader name = basename without ".comp.spv", for the per-shader trace table
        const char* b=spv; for(const char* c=spv;*c;++c) if(*c=='/'||*c=='\\') b=c+1;
        static std::deque<std::string> keep;   // deque: push_back does not invalidate c_str()
        keep.push_back(std::string(b));
        std::string& k=keep.back(); size_t dot=k.find(".comp"); if(dot!=std::string::npos) k=k.substr(0,dot);
        P.name=k.c_str();
    }
    return P;
}
// ---------------------------------------------------------------- dispatch batching
// One command buffer accumulates every shader launch; flush() ends+submits+waits once.
// This removes the per-dispatch fence that dominated the trace (3642 submits for 5 tokens).
static VkCommandBuffer g_bcb = VK_NULL_HANDLE;
static VkDescriptorPool g_bpool = VK_NULL_HANDLE;
static bool g_batching = false;
static VkDescriptorSet g_batch_dset = VK_NULL_HANDLE;
static uint64_t g_batch_ops = 0;
static VkBuffer g_barr_buf[16]; static uint32_t g_barr_n = 0;

static void batch_begin(){
    if(g_batching) return;
    if(g_bcb==VK_NULL_HANDLE){
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=G.cp; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        VK_CHECK(vkAllocateCommandBuffers(G.dev,&ai,&g_bcb));
    }
    if(g_bpool==VK_NULL_HANDLE){
        // g4lut3 binds 10 storage buffers, so a set can need 10 descriptors.
        const uint32_t MAXSETS = 4096u;
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAXSETS*10u};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets=MAXSETS; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps;
        VK_CHECK(vkCreateDescriptorPool(G.dev,&dpi,nullptr,&g_bpool));
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(g_bcb,&bi));
    g_batching=true; g_batch_ops=0;
}

static void batch_barrier(VkCommandBuffer cb){
    static const char* bmode = getenv("KINDI_BARRIER");
    static const bool no_barrier = getenv("KINDI_NO_BARRIER") != nullptr;
    if(no_barrier) return;
    if(bmode && !strcmp(bmode,"readonly")){
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,&mb, 0,nullptr, 0,nullptr);
        return;
    }
    if(bmode && !strcmp(bmode,"buffer") && g_barr_n>0){
        // scope the dependency to the buffers this dispatch touched
        static std::vector<VkBufferMemoryBarrier> bb;
        bb.clear();
        for(uint32_t i=0;i<g_barr_n;++i){
            VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
            b.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.buffer=g_barr_buf[i];
            b.offset=0; b.size=VK_WHOLE_SIZE;
            bb.push_back(b);
        }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0,nullptr, (uint32_t)bb.size(), bb.data(), 0,nullptr);
        return;
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,&mb, 0,nullptr, 0,nullptr);
}

static void batch_flush(){
    if(!g_batching) return;
    batch_barrier(g_bcb);
    VK_CHECK(vkEndCommandBuffer(g_bcb));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence;
    VK_CHECK(vkCreateFence(G.dev,&fi,nullptr,&fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount=1; si.pCommandBuffers=&g_bcb;
    VK_CHECK(vkQueueSubmit(G.q0,1,&si,fence));
    ktrace::count_submit();
    double _qw0 = ktrace::enabled() ? ktrace::now_us() : 0.0;
    VK_CHECK(vkWaitForFences(G.dev,1,&fence,VK_TRUE,60000000000ULL));
    if(ktrace::enabled()) ktrace::add_queue_wait(ktrace::now_us()-_qw0);
    ktrace::count_fence();
    vkDestroyFence(G.dev,fence,nullptr);
    // safe now: nothing references these sets any more
    if(g_bpool) VK_CHECK(vkResetDescriptorPool(G.dev,g_bpool,0));
    g_batch_dset=VK_NULL_HANDLE;
    g_batching=false;
}

static void bindbufs(Pipe&P, const std::vector<Buf>& bufs, const std::vector<VkDeviceSize>& offs={}){
    std::vector<VkDescriptorBufferInfo> infos(bufs.size()); std::vector<VkWriteDescriptorSet> w(bufs.size());
    for(size_t i=0;i<bufs.size();++i){ infos[i].buffer=bufs[i].buf; infos[i].offset=(i<offs.size())?offs[i]:0; infos[i].range=bufs[i].size-infos[i].offset;
        w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].pNext=nullptr; w[i].dstSet=P.dset; w[i].dstBinding=(uint32_t)i; w[i].dstArrayElement=0; w[i].descriptorCount=1; w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pImageInfo=nullptr; w[i].pBufferInfo=&infos[i]; w[i].pTexelBufferView=nullptr; }
    if(g_batching){
        // allocate a fresh set: the previous one may still be referenced by the batch
        VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        a.descriptorPool=g_bpool; a.descriptorSetCount=1; a.pSetLayouts=&P.dsl;
        VkDescriptorSet nds=VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(G.dev,&a,&nds));
        for(auto&ww:w) ww.dstSet=nds;
        g_batch_dset=nds;
    }
    vkUpdateDescriptorSets(G.dev,(uint32_t)w.size(),w.data(),0,nullptr);
    for(size_t i=0;i<bufs.size() && i<16;++i) P.brange[i]=infos[i].range;
    g_barr_n=(uint32_t)(bufs.size()<16?bufs.size():16);
    for(uint32_t i=0;i<g_barr_n;++i) g_barr_buf[i]=bufs[i].buf;
}
static uint64_t pipe_bytes(const Pipe&P){
    uint64_t b=0; for(uint32_t i=0;i<P.nbind && i<16;++i) b+=P.brange[i]; return b;
}
// Set by dispatch_nb() for the next dispatch only.
//
// The barrier after every dispatch is a COMPUTE -> COMPUTE pipeline barrier. It has
// to drain the producer before the consumer starts, and on this driver the cost is
// the barrier itself, not its scope: measured 6156 ms wall with a full VkMemoryBarrier,
// 6163 with SHADER_WRITE->SHADER_READ only, and 6139 with a buffer-scoped
// VkBufferMemoryBarrier. All the same. Removing barriers altogether gives 4919 ms, so
// 1180 ms (44 us per dispatch) is the drain, and the only way to recover it is to not
// emit a barrier where there is no dependency.
//
// dispatch_nb() is therefore only safe where the following dispatch provably does not
// read anything this one writes. The sibling projections after a norm are that case:
// they all read the norm output and write disjoint outputs.
static bool g_nobar_next = false;

static void dispatch(const Pipe&P, VkQueue q, VkCommandPool pool, uint32_t gx,uint32_t gy,uint32_t gz, const void*push,uint32_t psz){
    const bool tr = ktrace::enabled();
    double th0 = tr ? ktrace::now_us() : 0.0;
    uint32_t pair = UINT32_MAX;
    batch_begin();
    VkCommandBuffer cb=g_bcb;
    VkDescriptorSet ds = (g_batch_dset!=VK_NULL_HANDLE) ? g_batch_dset : P.dset;
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P.pipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P.pl,0,1,&ds,0,nullptr);
    if(push&&psz) vkCmdPushConstants(cb,P.pl,VK_SHADER_STAGE_COMPUTE_BIT,0,psz,push);
    if(tr) ktrace::mark_before(cb,pair);
    vkCmdDispatch(cb,gx,gy,gz);
    if(tr) ktrace::mark_after(cb,pair);
    if(!g_nobar_next) batch_barrier(cb);
    g_nobar_next = false;
    g_batch_ops++;
    if(tr){
        double th1 = ktrace::now_us();
        ktrace::record(P.name?P.name:"?", gx,gy,gz, P.lx, psz, pipe_bytes(P), th1-th0, -1.0, pair);
    }
    // regression trap: KINDI_NO_BATCH=1 restores one submit+fence per dispatch
    static const bool no_batch = getenv("KINDI_NO_BATCH") != nullptr;
    // bound the batch: a 144-token prefill is ~64k dispatches, which would overflow the
    // descriptor pool and the command buffer if left to accumulate.
    static const uint64_t BATCH_MAX = [](){
        const char* e=getenv("KINDI_BATCH_MAX"); return e?(uint64_t)atoll(e):(uint64_t)2048; }();
    if(no_batch || g_batch_ops >= BATCH_MAX) batch_flush();
}
// Dispatch with no trailing barrier. Only for a dispatch whose output is not read
// until after the next explicit barrier_now().
static void dispatch_nb(const Pipe&P, VkQueue q, VkCommandPool pool, uint32_t gx,uint32_t gy,uint32_t gz, const void*push,uint32_t psz){
    g_nobar_next = true;
    dispatch(P,q,pool,gx,gy,gz,push,psz);
}

// Emit one barrier at the point where a real dependency exists.
static void barrier_now(){ batch_barrier(g_bcb); }

static void upload(Buf*dst, const void*data, size_t nb){
    double _tx0 = ktrace::enabled() ? ktrace::now_us() : 0.0;
    // One persistent staging buffer, grown on demand. A fresh alloc/free per tensor
    // (400 of them on the 27B) fragments the 7.75 GB device-local heap and OOMs.
    static Buf st{};
    if(st.buf==VK_NULL_HANDLE || st.size < (VkDeviceSize)nb){
        if(st.buf) freebuf(st);
        st=mkbuf((VkDeviceSize)nb,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    memcpy(st.mapped,data,nb);
    if(ktrace::enabled()) ktrace::count_xfer((uint64_t)nb, ktrace::now_us()-_tx0);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=G.tp; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer c; VK_CHECK(vkAllocateCommandBuffers(G.dev,&ai,&c));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VK_CHECK(vkBeginCommandBuffer(c,&bi));
    VkBufferCopy cp{0,0,(VkDeviceSize)nb}; vkCmdCopyBuffer(c,st.buf,dst->buf,1,&cp); VK_CHECK(vkEndCommandBuffer(c));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; VK_CHECK(vkCreateFence(G.dev,&fi,nullptr,&fe));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c; VK_CHECK(vkQueueSubmit(G.qt,1,&si,fe));
    VK_CHECK(vkWaitForFences(G.dev,1,&fe,VK_TRUE,60000000000ULL)); vkDestroyFence(G.dev,fe,nullptr); vkFreeCommandBuffers(G.dev,G.tp,1,&c);
}

static void fillbuf(Buf&b){
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=G.cp; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer c; VK_CHECK(vkAllocateCommandBuffers(G.dev,&ai,&c));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VK_CHECK(vkBeginCommandBuffer(c,&bi));
    vkCmdFillBuffer(c,b.buf,0,b.size,0); VK_CHECK(vkEndCommandBuffer(c));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; VK_CHECK(vkCreateFence(G.dev,&fi,nullptr,&fe));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c; VK_CHECK(vkQueueSubmit(G.q0,1,&si,fe));
    VK_CHECK(vkWaitForFences(G.dev,1,&fe,VK_TRUE,60000000000ULL)); vkDestroyFence(G.dev,fe,nullptr); vkFreeCommandBuffers(G.dev,G.cp,1,&c);
}

// Record a device-to-device copy into the current batch (no submit, no flush).
// Used by the chunked path to move one token's residual in/out of the attention buffers.
static void copyB(Buf&src, Buf&dst, size_t bytes, size_t srcOff=0, size_t dstOff=0){
    batch_begin();
    {   // the source is normally a compute dispatch's output: make the shader writes
        // visible to the transfer stage before the copy reads them
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(g_bcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,&mb, 0,nullptr, 0,nullptr);
    }
    VkBufferCopy r{srcOff, dstOff, bytes};
    vkCmdCopyBuffer(g_bcb, src.buf, dst.buf, 1, &r);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(g_bcb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,&mb, 0,nullptr, 0,nullptr);
    g_batch_ops++;
}
// Copy recorded into the current batch with NO barrier of its own. Use with
// bulk_bar_compute_to_transfer / bulk_bar_transfer_to_compute around a run of copies:
// copyB emits a full memory barrier per call, and a state snapshot is ~96 copies, so
// 96 barriers dominated the pass it was added to.
static void copyB_nb(Buf&src, Buf&dst, size_t bytes, size_t srcOff=0, size_t dstOff=0){
    batch_begin();
    VkBufferCopy r{srcOff, dstOff, bytes};
    vkCmdCopyBuffer(g_bcb, src.buf, dst.buf, 1, &r);
    g_batch_ops++;
}
static void bulk_bar_compute_to_transfer(){
    batch_begin();
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(g_bcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,&mb, 0,nullptr, 0,nullptr);
}
static void bulk_bar_transfer_to_compute(){
    batch_begin();
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(g_bcb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,&mb, 0,nullptr, 0,nullptr);
}

static void copybuf(Buf&src, Buf&dst, size_t bytes){
    batch_flush();   // host readback: the batch must be complete
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=G.cp; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer c; VK_CHECK(vkAllocateCommandBuffers(G.dev,&ai,&c));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VK_CHECK(vkBeginCommandBuffer(c,&bi));
    VkBufferCopy region{0,0,bytes}; vkCmdCopyBuffer(c,src.buf,dst.buf,1,&region); VK_CHECK(vkEndCommandBuffer(c));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; VK_CHECK(vkCreateFence(G.dev,&fi,nullptr,&fe));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c; VK_CHECK(vkQueueSubmit(G.q0,1,&si,fe));
    VK_CHECK(vkWaitForFences(G.dev,1,&fe,VK_TRUE,60000000000ULL)); vkDestroyFence(G.dev,fe,nullptr); vkFreeCommandBuffers(G.dev,G.cp,1,&c);
}

int main(int argc, char**argv){
    const char* model=nullptr;
    const char* prompt=nullptr;
    float temperature=0.0f;
    int topk=0;
    int dump=0;
    int native5=0;   // PTQ1_0 kept in its 28-byte/128-weight form (27B does not fit expanded)
    int force_gen=0;
    int raw_embed=0;
    int ngt=5;       // -n / --n-predict: number of tokens to generate
    int dumpvec=0;   // full-vector dumps for offline fold verification
    int selftest=0;  // prove gemvM correctness at T>1
    int mtp_accept=0; // --mtp-accept N: run the MTP draft head one token ahead, count hits
    int spec_n=0;     // --spec N: speculative decode with the MTP draft head, k=1
    int serve=0;      // --serve: run the OpenAI-compatible HTTP API
    const char* srv_host="127.0.0.1";
    int srv_port=8080;
    int prefix_cache=1;   // --no-prefix-cache disables prompt reuse
    int ctx_arg=0;   // --ctx N: override the context length (VRAM budget)   // deq_emb2: skip the fold (raw PQ2_0 row) for cross-format comparison  // force the generic GQA attention path (G != 4); used to validate it on 2B
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--model")&&i+1<argc) model=argv[++i];
        else if(!strcmp(argv[i],"--prompt")&&i+1<argc) prompt=argv[++i];
        else if(!strcmp(argv[i],"--temp")&&i+1<argc) temperature=atof(argv[++i]);
        else if(!strcmp(argv[i],"--top-k")&&i+1<argc) topk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dump")) dump=1;
        else if(!strcmp(argv[i],"--native5")) native5=1;
        else if(!strcmp(argv[i],"--force-gen")) force_gen=1;
        else if(!strcmp(argv[i],"--raw-embed")) raw_embed=1;
        else if(!strcmp(argv[i],"--dumpvec")) dumpvec=1;
        else if(!strcmp(argv[i],"--selftest")) selftest=1;
        else if(!strcmp(argv[i],"--mtp-accept")&&i+1<argc) mtp_accept=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--spec")&&i+1<argc) spec_n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--serve")) serve=1;
        else if(!strcmp(argv[i],"--host")&&i+1<argc) srv_host=argv[++i];
        else if(!strcmp(argv[i],"--port")&&i+1<argc) srv_port=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--no-prefix-cache")) prefix_cache=0;
        else if(!strcmp(argv[i],"--ctx")&&i+1<argc) ctx_arg=atoi(argv[++i]);
        else if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--n-predict"))&&i+1<argc) ngt=atoi(argv[++i]);
        else if(ktrace::parse_arg(argc,argv,i)) { /* --trace / --trace-json / --trace-label */ }
    }
    if(!model){ fprintf(stderr,"usage: --model <path.gguf>\n"); return 1; }
    initdev();
    ktrace::clock_init();
    // ---- memory-map GGUF ----
    HANDLE hFile = CreateFileA(model, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot open %s\n", model); return 1; }
    LARGE_INTEGER fsz; GetFileSizeEx(hFile, &fsz);
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    const unsigned char* filebase = (const unsigned char*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    size_t filesize = (size_t)fsz.QuadPart;

    // ---- GGUF parse ----
    const unsigned char* p = filebase;
    auto rd_u32 = [&]() { uint32_t v; memcpy(&v, p, 4); p += 4; return v; };
    auto rd_u64 = [&]() { uint64_t v; memcpy(&v, p, 8); p += 8; return v; };
    auto rd_f32 = [&]() { float v; memcpy(&v, p, 4); p += 4; return v; };
    auto rd_f64 = [&]() { double v; memcpy(&v, p, 8); p += 8; return v; };
    auto rd_str = [&]() { uint64_t n = rd_u64(); std::string s((const char*)p, n); p += n; return s; };
    uint32_t magic = rd_u32();
    if (magic != 0x46554747) { fprintf(stderr, "not GGUF (magic %x)\\n", magic); return 1; }
    uint32_t version = rd_u32();
    (void)version;
    uint64_t n_tensors = rd_u64();
    uint64_t n_kv = rd_u64();

    int D=0, NL=0, NFF=0, NH=0, NHKV=0, HD=0, NK=0, HK=0, NV=0, rope_dim=0, interval=0, HB=512, ctx=4096;
    int gdn_inner = 0;
    int nextn_layers = 0;   // blk.<NL>.. are MTP/nextn heads, not transformer blocks
    int eos_id = -1;
    std::vector<std::string> gguf_tokens, gguf_merges;
    std::vector<float> sign_values;
    std::vector<int> sign_widths;
    uint32_t align = 32;

    auto read_scalar_int = [&](uint32_t t)->int {
        if (t==0){ uint8_t v=*p; p+=1; return v; }
        if (t==1){ int8_t v; memcpy(&v,p,1); p+=1; return v; }
        if (t==2){ uint16_t v; memcpy(&v,p,2); p+=2; return v; }
        if (t==3){ int16_t v; memcpy(&v,p,2); p+=2; return v; }
        if (t==4){ uint32_t v; memcpy(&v,p,4); p+=4; return (int)v; }
        if (t==5){ int32_t v; memcpy(&v,p,4); p+=4; return v; }
        if (t==6){ float v; memcpy(&v,p,4); p+=4; return (int)v; }
        if (t==7){ uint8_t v=*p; p+=1; return v; }
        if (t==10){ uint64_t v; memcpy(&v,p,8); p+=8; return (int)v; }
        if (t==11){ int64_t v; memcpy(&v,p,8); p+=8; return (int)v; }
        if (t==12){ double v; memcpy(&v,p,8); p+=8; return (int)v; }
        fprintf(stderr,"not int type %u\n",t); exit(1);
        return 0;
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

    for (uint64_t ki = 0; ki < n_kv; ++ki) {
        std::string key = rd_str();
        uint32_t type = rd_u32();
        if (key == "qwen35.block_count") NL = read_scalar_int(type);
        else if (key == "qwen35.embedding_length") D = read_scalar_int(type);
        else if (key == "qwen35.feed_forward_length") NFF = read_scalar_int(type);
        else if (key == "qwen35.attention.head_count") NH = read_scalar_int(type);
        else if (key == "qwen35.attention.head_count_kv") NHKV = read_scalar_int(type);
        else if (key == "qwen35.attention.key_length") HD = read_scalar_int(type);
        else if (key == "qwen35.rope.dimension_count") rope_dim = read_scalar_int(type);
        else if (key == "qwen35.ssm.state_size") HK = read_scalar_int(type);
        else if (key == "qwen35.ssm.group_count") NK = read_scalar_int(type);
        else if (key == "qwen35.ssm.inner_size") gdn_inner = read_scalar_int(type);
        else if (key == "qwen35.full_attention_interval") interval = read_scalar_int(type);
        else if (key == "qwen35.nextn_predict_layers") nextn_layers = read_scalar_int(type);
        else if (key == "tokenizer.ggml.eos_token_id") eos_id = read_scalar_int(type);
        else if (key == "qwen35.context_length") { int cl = read_scalar_int(type); (void)cl; }
        else if (key == "qwen35.rope.freq_base") {
            if (type == 6) { float v = rd_f32(); (void)v; }
            else if (type == 12) { double v = rd_f64(); (void)v; }
            else skip_value(type);
        }
        else if (key == "general.alignment") align = (uint32_t)read_scalar_int(type);
        else if (key == "prism.hadamard.block_size") HB = read_scalar_int(type);
        else if (key == "prism.hadamard.sign_values") {
            uint32_t et = rd_u32(); uint64_t n = rd_u64();
            sign_values.resize(n);
            if (et == 6) for (uint64_t i = 0; i < n; ++i) sign_values[i] = rd_f32();
            else if (et == 12) for (uint64_t i = 0; i < n; ++i) sign_values[i] = (float)rd_f64();
            else if (et == 5) for (uint64_t i = 0; i < n; ++i) { int32_t v; memcpy(&v,p,4); p+=4; sign_values[i]=(float)v; }
            else if (et == 1) for (uint64_t i = 0; i < n; ++i) { int8_t v; memcpy(&v,p,1); p+=1; sign_values[i]=(float)v; }
            else if (et == 4) for (uint64_t i = 0; i < n; ++i) { uint32_t v; memcpy(&v,p,4); p+=4; sign_values[i]=(float)v; }
            else { for (uint64_t i = 0; i < n; ++i) skip_value(et); }
        }
        else if (key == "prism.hadamard.sign_widths") {
            uint32_t et = rd_u32(); uint64_t n = rd_u64();
            sign_widths.resize(n);
            for (uint64_t i = 0; i < n; ++i) sign_widths[i] = read_scalar_int(et);
        }
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
        else {
            skip_value(type);
        }
    }
    NV = gdn_inner / HK;
    if (nextn_layers > 0) {
        // blk.<NL-nextn>..blk.<NL-1> are MTP heads (nextn.eh_proj/enorm/hnorm), not transformer blocks:
        // running a normal GDN block on them reads missing tensors and yields NaN.
        fprintf(stderr,"nextn layers (MTP head, not run in the base loop): %d -> transformer blocks %d\n",
                nextn_layers, NL - nextn_layers);
        NL -= nextn_layers;
    }
    int CONV_CH = 2*NK*HK + NV*HK;   // 2B: 6144, 27B: 10240
    int n_gdn_est = 0;
    for(int il=0;il<NL;++il) if((il+1)%interval!=0) n_gdn_est++;
    int VOCAB = 248320; int NATT = NL / interval;
    if(ctx_arg > 0){ ctx = ctx_arg; fprintf(stderr,"ctx forced to %d by --ctx\n", ctx); }
    // chunked prefill width (0 = serial, the default and the reference path)
    int CHUNK = 0;
    { const char* e=getenv("KINDI_CHUNK"); if(e) CHUNK=atoi(e); if(CHUNK<0) CHUNK=0; if(CHUNK>4) CHUNK=4;
      // The API gets long prompts, where serial prefill dominates; default it on there.
      // KINDI_CHUNK still overrides, including KINDI_CHUNK=0 for the serial reference.
      if(serve && !getenv("KINDI_CHUNK")) CHUNK=4;
      if(CHUNK) fprintf(stderr,"chunked prefill: KINDI_CHUNK=%d\n", CHUNK); }
    {
        // Fit the context-sized buffers (sbuf, apart, Kc, Vc) beside the weights.
        // per position: K/V fp16 (NATT*2*NHKV*HD*2) + scores (NH*4) + partials (NH*HD*4/256)
        double per_pos = (double)NATT*2*NHKV*HD*2 + NH*4.0 + (NH*(double)HD*4.0)/256.0;
        double budget  = (double)3.4e9;                 // generous for 4/9B; clamped below for 27B
        if(ctx_arg > 0) budget = (double)per_pos * ctx;  // --ctx given: keep exactly that
        double nfit    = budget / per_pos;
        int fit = (int)(nfit/256)*256;
        if(fit < ctx){ fprintf(stderr,"ctx reduced %d -> %d to fit VRAM\n", ctx, fit); ctx = fit; }
        if(ctx < 512) ctx = 512;
    }
    fprintf(stderr,"dims: L=%d D=%d FFN=%d H=%d KV=%d HD=%d G=%d HK=%d NV=%d HB=%d\n",NL,D,NFF,NH,NHKV,HD,NK,HK,NV,HB);

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

    // data_start: round the end of the tensor-info block up to `align`, but VERIFY it.
    // Some files (27B PQ2_0) carry extra padding between the tensor infos and the data
    // section, so the naive round-up lands 50340 bytes early and every tensor is read
    // from the wrong offset. Derive a second candidate from the file size + the last
    // tensor, then pick whichever makes output_norm.weight look like an RMS norm
    // weight (finite, small, positive-ish). Falls back to the round-up.
    auto align_up = [&](size_t v) { return (v + align - 1) / align * align; };
    size_t ds_a = align_up(pos0);
    size_t ds_b = ds_a;
    {
        const GT* last = nullptr;
        for (auto& g : gtensors) if (!last || g.offset > last->offset) last = &g;
        if (last) {
            uint64_t n = 1;
            for (uint64_t d : last->dims) n *= d;
            uint64_t bytes = 0;
            if (last->type == 142)      bytes = (n / 128) * 34;
            else if (last->type == 143) bytes = (n / 128) * 28;
            else if (last->type == 0)   bytes = n * 4;
            else if (last->type == 30)  bytes = n * 2;
            else if (last->type == 8)   bytes = n + (n / 32) * 4;
            else if (last->type == 1)   bytes = n * 2;
            if (bytes && filesize > last->offset + bytes)
                ds_b = ((filesize - bytes - last->offset) / align) * align;
        }
    }
    auto norm_sane = [&](size_t ds) -> bool {
        const GT* on = nullptr;
        for (auto& g : gtensors) if (g.name == "output_norm.weight") on = &g;
        if (!on || on->type != 0) return false;
        uint64_t n = 1;
        for (uint64_t d : on->dims) n *= d;
        if (n < 8) return false;
        if (ds + on->offset + n * 4 > filesize) return false;
        const float* f = (const float*)(filebase + ds + on->offset);
        int good = 0;
        for (int i = 0; i < 8; ++i) {
            float v = f[i];
            if (v == v && v > 0.01f && v < 100.0f) ++good;
        }
        return good >= 6;
    };
    size_t data_start = ds_a;
    if (!norm_sane(ds_a) && norm_sane(ds_b)) {
        fprintf(stderr, "data_start: %zu (round-up) is wrong, using %zu (from file size); delta %lld\n",
                ds_a, ds_b, (long long)((long long)ds_b - (long long)ds_a));
        data_start = ds_b;
    } else {
        fprintf(stderr, "data_start: %zu\n", data_start);
    }

    std::map<std::string,Buf> codebufs, scalebufs, f32bufs;
    // The PTQ1_0 fp16 block scale is read straight out of the 28-byte block in the
    // shader (unpackHalf2x16). Widening it into a float buffer first was 8% faster
    // but cost 0.79 GB of VRAM, which pushed the 27B past the 8 GB limit once the
    // MTP state snapshot was allocated, so it is not used.
    std::map<std::string,std::pair<int,int>> tinfo;

    uint32_t pow3[6] = {1,3,9,27,81,243};
    auto half_to_float = [](uint16_t h) -> float {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) f = sign;
            else {
                int e = 1;
                while (!(mant & 0x400)) { mant <<= 1; --e; }
                mant &= 0x3FF;
                f = sign | ((e + 127 - 15) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f = sign | 0x7F800000 | (mant << 13);
        } else {
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float out; memcpy(&out, &f, 4); return out;
    };

    size_t n_tern=0, n_q8=0, n_q4=0;
    std::set<std::string> q8names;
    std::set<std::string> q4names;
    std::set<std::string> native_names;
    std::set<std::string> pq2_names;
    size_t n_pq2=0;
    for (auto& gt : gtensors) {
        uint64_t ne0 = gt.dims[0];
        uint64_t rows = 1;
        for (size_t i = 1; i < gt.dims.size(); ++i) rows *= gt.dims[i];
        int r = (int)rows, c = (int)ne0;
        const unsigned char* data = filebase + data_start + gt.offset;
        if (gt.type == 142) {
            // PQ2_0: 34-byte blocks (fp16 d + uint8 qs[32]), 128 weights, 2 bits each.
            // Kept packed; g2pq / deq_emb2 decode on the device.
            int bpr = c / 128;
            size_t nbytes = (size_t)r * bpr * 34;
            std::vector<uint32_t> words((nbytes + 3) / 4, 0u);
            memcpy(words.data(), data, nbytes);
            Buf cb = mkbuf(words.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&cb, words.data(), words.size() * 4);
            codebufs[gt.name] = cb;
            tinfo[gt.name] = std::make_pair(r, c);
            pq2_names.insert(gt.name);
            n_pq2++;
        } else if (gt.type == 143 && native5) {
            // Native PTQ1_0: keep the 28-byte/128-weight blocks verbatim (1.75 bpw).
            int bpr = c / 128;
            size_t nbytes = (size_t)r * bpr * 28;
            std::vector<uint32_t> words((nbytes + 3) / 4, 0u);
            memcpy(words.data(), data, nbytes);
            Buf cb = mkbuf(words.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&cb, words.data(), words.size() * 4);
            codebufs[gt.name] = cb;
            tinfo[gt.name] = std::make_pair(r, c);
            native_names.insert(gt.name);
            n_tern++;
        } else if (gt.type == 2) {
            // Q4_0: block = { fp16 d; uint8 qs[16] } (18 bytes, 32 weights). Kept as the raw
            // blocks; q4f decodes on the device. The MTP block of the "...MTP-Q4_0" model is
            // Q4_0, so without this it falls through to the 4-trit branch and is garbage.
            int bpr = c / 32;
            size_t nbytes = (size_t)r * bpr * 18;
            std::vector<uint32_t> words((nbytes + 3) / 4, 0u);
            memcpy(words.data(), data, nbytes);
            Buf cb = mkbuf(words.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&cb, words.data(), words.size() * 4);
            codebufs[gt.name] = cb;
            tinfo[gt.name] = std::make_pair(r, c);
            q4names.insert(gt.name);
            n_q4++;
        } else if (gt.type == 8) {
            // Q8_0: 32 signed bytes + fp16 scale per block (34 bytes). Split into
            // a byte-packed uint code array and a uint scale array (fp16 bits, low half).
            int bpr = c / 32;
            size_t nbytes = (size_t)r * c;
            std::vector<uint32_t> codes((nbytes + 3) / 4, 0u);
            std::vector<uint32_t> scales((size_t)r * bpr, 0u);
            // block_q8_0 is { ggml_half d; int8_t qs[32]; } (ggml-common.h): the fp16
            // scale comes FIRST. Reading the codes from bytes 0..31 and the scale from
            // bytes 32..33 shifted every weight by two bytes and turned the last two
            // codes into a "scale", which can be a NaN bit pattern.
            for (int row = 0; row < r; ++row) {
                for (int b = 0; b < bpr; ++b) {
                    const unsigned char* blk = data + ((size_t)row * bpr + b) * 34;
                    uint16_t sv; memcpy(&sv, blk, 2);
                    scales[(size_t)row * bpr + b] = (uint32_t)sv;
                    for (int i = 0; i < 32; ++i) {
                        size_t bi = (size_t)row * c + (size_t)b * 32 + i;
                        codes[bi >> 2] |= ((uint32_t)blk[2 + i]) << (8 * (bi & 3));
                    }
                }
            }
            Buf cb = mkbuf(codes.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&cb, codes.data(), codes.size() * 4);
            Buf sb = mkbuf(scales.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&sb, scales.data(), scales.size() * 4);
            codebufs[gt.name] = cb; scalebufs[gt.name] = sb;
            tinfo[gt.name] = std::make_pair(r, c);
            q8names.insert(gt.name);
            n_q8++;
        } else if (gt.type == 143) {
            int bpr = c / 128;
            int nb = r * bpr;
            std::vector<unsigned char> trits((size_t)nb * 128);
            for (int bi = 0; bi < nb; ++bi) {
                const unsigned char* raw = data + (size_t)bi * 28;
                unsigned char* out = trits.data() + (size_t)bi * 128;
                for (int n = 0; n < 5; ++n) {
                    for (int j = 0; j < 16; ++j) {
                        uint32_t q = ((uint32_t)raw[j] * pow3[n]) & 0xFF;
                        out[n*16 + j] = (unsigned char)((q * 3) >> 8);
                    }
                }
                for (int n = 0; n < 5; ++n) {
                    for (int j = 0; j < 8; ++j) {
                        uint32_t q = ((uint32_t)raw[16 + j] * pow3[n]) & 0xFF;
                        out[80 + n*8 + j] = (unsigned char)((q * 3) >> 8);
                    }
                }
                for (int n = 0; n < 4; ++n) {
                    for (int j = 0; j < 2; ++j) {
                        uint32_t q = ((uint32_t)raw[24 + j] * pow3[n]) & 0xFF;
                        out[120 + n*2 + j] = (unsigned char)((q * 3) >> 8);
                    }
                }
            }
            std::vector<unsigned char> codes((size_t)r * bpr * 32);
            std::vector<uint16_t> scales((size_t)r * bpr);
            for (int bi = 0; bi < nb; ++bi) {
                const unsigned char* raw = data + (size_t)bi * 28;
                const unsigned char* tr = trits.data() + (size_t)bi * 128;
                int row = bi / bpr;
                int phase = bi % bpr;
                unsigned char* cb = codes.data() + (size_t)row * (bpr * 32) + (size_t)phase * 32;
                for (int s = 0; s < 32; ++s) {
                    int t0 = tr[s*4+0], t1 = tr[s*4+1], t2 = tr[s*4+2], t3 = tr[s*4+3];
                    cb[s] = (unsigned char)(t0 + 3*t1 + 9*t2 + 27*t3);
                }
                uint16_t v; memcpy(&v, raw + 26, 2);
                scales[(size_t)row * bpr + phase] = v;
            }
            std::vector<uint32_t> scales32((size_t)r * bpr);
            for (size_t si = 0; si < scales32.size(); ++si) scales32[si] = (uint32_t)scales[si];
            Buf cb = mkbuf((size_t)r * bpr * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&cb, codes.data(), (size_t)r * bpr * 32);
            Buf sb = mkbuf((size_t)r * bpr * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&sb, scales32.data(), (size_t)r * bpr * 4);
            codebufs[gt.name] = cb;
            scalebufs[gt.name] = sb;
            tinfo[gt.name] = std::make_pair(r, c);
            n_tern++;
        } else if (gt.type == 0) {
            size_t n = (size_t)r * c;
            Buf f = mkbuf(n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&f, data, n * 4);
            f32bufs[gt.name] = f;
            tinfo[gt.name] = std::make_pair(r, c);
        } else if (gt.type == 30) {
            size_t n = (size_t)r * c;
            std::vector<uint32_t> f32bits(n);
            const uint16_t* src = (const uint16_t*)data;
            for (size_t i = 0; i < n; ++i) f32bits[i] = ((uint32_t)src[i]) << 16;
            Buf f = mkbuf(n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&f, f32bits.data(), n * 4);
            f32bufs[gt.name] = f;
            tinfo[gt.name] = std::make_pair(r, c);
        } else if (gt.type == 1) {
            size_t n = (size_t)r * c;
            std::vector<float> f32(n);
            const uint16_t* src = (const uint16_t*)data;
            for (size_t i = 0; i < n; ++i) f32[i] = half_to_float(src[i]);
            Buf f = mkbuf(n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&f, f32.data(), n * 4);
            f32bufs[gt.name] = f;
            tinfo[gt.name] = std::make_pair(r, c);
        } else {
            fprintf(stderr, "unsupported tensor type %u for %s\n", gt.type, gt.name.c_str());
            return 1;
        }
    }
    fprintf(stderr,"weights uploaded: %zu ternary + %zu f32 + %zu q8_0 + %zu q4_0 + %zu pq2_0\n", n_tern, f32bufs.size(), n_q8, n_q4, n_pq2);
    {
        // Context-sized buffers are allocated next. Clamp the context to what is actually
        // left on the device-local heap rather than trusting the pre-load estimate: the
        // 27B's 5.9 GB of weights is only known now, and an unclamped --ctx used to fail
        // with a bare "Vulkan error -2" at allocation time.
        double per_pos = (double)NATT*2*NHKV*HD*2 + NH*4.0 + (NH*(double)HD*4.0)/256.0;
        if(spec_n>0) per_pos += 2.0*NH*4.0;   // MTP layer KV
        double heap = 0;
        for(uint32_t hi=0; hi<G.mp.memoryHeapCount; ++hi)
            if(G.mp.memoryHeaps[hi].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                if((double)G.mp.memoryHeaps[hi].size > heap) heap = (double)G.mp.memoryHeaps[hi].size;
        // leave room for the scratch buffers, the spec snapshot and driver overhead.
        // Calibrated from measurement: the 27B runs at ctx 14336 (0.94 GB of KV) beside
        // 6.74 GB of weights on a 7.75 GiB heap, so everything that is not weights and not
        // KV comes to ~0.64 GB. The first estimate of 1.6 GB clamped --ctx 8192 to 512.
        double fixed = 0.66e9;
        if(spec_n>0) fixed += (double)n_gdn_est*(NV*HK*HK*4+3.0*CONV_CH*4) + 2.0*VOCAB*2*4;
        double freeb = heap - (double)g_alloc_total - fixed;
        int fit = (int)(freeb/per_pos);
        fit = (fit/256)*256;
        if(fit < 512) fit = 512;
        if(ctx > fit){
            fprintf(stderr,"ctx reduced %d -> %d to fit VRAM (%.2f GB heap, %.2f GB weights, %.2f GB free)\n",
                    ctx, fit, heap/1e9, (double)g_alloc_total/1e9, freeb/1e9);
            ctx = fit;
        } else {
            fprintf(stderr,"ctx %d fits (%.2f GB heap, %.2f GB weights, %.2f GB free, %d max)\n",
                    ctx, heap/1e9, (double)g_alloc_total/1e9, freeb/1e9, fit);
        }
    }

    std::string shdir="shaders";
    Pipe pg4=mkpipe((shdir+"/g4lut2.comp.spv").c_str(),4,8);
    Pipe pg4m=mkpipe((shdir+"/g4lut3.comp.spv").c_str(),10,16);
    Pipe pg4M=mkpipe((shdir+"/g4lutM.comp.spv").c_str(),4,16);   // push grew to {R,C,T,xstride}
    Pipe pg2M =mkpipe((shdir+"/g2pqM.comp.spv").c_str(),3,16);
    Pipe pq8f =mkpipe((shdir+"/q8f.comp.spv").c_str(),4,16);
    Pipe pq4f =mkpipe((shdir+"/q4f.comp.spv").c_str(),3,16);
    Pipe paddv =mkpipe((shdir+"/addv.comp.spv").c_str(),3,4);      // residual add
    Pipe psilu =mkpipe((shdir+"/silu_mul.comp.spv").c_str(),3,4);   // fold-free SwiGLU
    // T-token variants for the chunked prefill path (KINDI_CHUNK)
    Pipe prmsfoldT=mkpipe((shdir+"/rmsfold_T.comp.spv").c_str(),7,24);
    Pipe psmfoldT =mkpipe((shdir+"/smfold_T.comp.spv").c_str(),4,32);
    Pipe pconvT   =mkpipe((shdir+"/gdn_conv_T.comp.spv").c_str(),4,16);
    Pipe pdeltaT  =mkpipe((shdir+"/gdn_delta_T.comp.spv").c_str(),7,28);
    Pipe pgatesT  =mkpipe((shdir+"/gdn_gates_T.comp.spv").c_str(),7,12);
    Pipe pl2wT    =mkpipe((shdir+"/l2w_T.comp.spv").c_str(),2,20);
    Pipe pargmax=mkpipe((shdir+"/argmax.comp.spv").c_str(),2,4);
    Pipe prmsfold=mkpipe((shdir+"/rmsfold.comp.spv").c_str(),7,12);
    Pipe pdeq=mkpipe((shdir+"/deq_emb3.comp.spv").c_str(),4,8);
    Pipe psmfold=mkpipe((shdir+"/smfold.comp.spv").c_str(),4,16);
    Pipe pfwh=mkpipe((shdir+"/fold_fwht.comp.spv").c_str(),3,4);
    Pipe prmsw=mkpipe((shdir+"/rmsw.comp.spv").c_str(),3,8);
    Pipe pl2w=mkpipe((shdir+"/l2w.comp.spv").c_str(),2,8);
    Pipe prope=mkpipe((shdir+"/rope.comp.spv").c_str(),4,4);
    Pipe pgates=mkpipe((shdir+"/gdn_gates.comp.spv").c_str(),7,4);
    Pipe pconv=mkpipe((shdir+"/gdn_conv.comp.spv").c_str(),4,4);
    Pipe pdelta=mkpipe((shdir+"/gdn_delta.comp.spv").c_str(),7,8);
    Pipe psoft=mkpipe((shdir+"/softmax_rows.comp.spv").c_str(),1,4);
    Pipe pattns4=mkpipe((shdir+"/attn_s4w.comp.spv").c_str(),3,8);
    Pipe pattnv4=mkpipe((shdir+"/attn_v4p.comp.spv").c_str(),3,12);
    Pipe pattncomb=mkpipe((shdir+"/attn_comb.comp.spv").c_str(),3,8);
    Pipe pstore=mkpipe((shdir+"/store_h.comp.spv").c_str(),2,4);
    Pipe pcpq=mkpipe((shdir+"/cpq.comp.spv").c_str(),2,8);
    Pipe psumparts=mkpipe((shdir+"/sumparts.comp.spv").c_str(),2,8);
    Pipe pq8=mkpipe((shdir+"/g4q8.comp.spv").c_str(),4,8);
    Pipe pg5=mkpipe((shdir+"/g5ptq.comp.spv").c_str(),3,8);
    // Experiment: the same decode with a different workgroup geometry.
    //   KINDI_G5=byte  -> g5ptq_byte  (1 row per 64-thread workgroup)
    //   KINDI_G5=byte4 -> g5ptq_byte4 (4 rows per 256-thread workgroup, subgroupAdd)
    // Unset keeps the shipped g5ptq. Used to attribute the gap to geometry vs decode.
    Pipe pg5b =mkpipe((shdir+"/g5ptq_byte.comp.spv").c_str(),3,8);
    Pipe pg5b4=mkpipe((shdir+"/g5ptq_byte4.comp.spv").c_str(),3,8);
    Pipe pg5M=mkpipe((shdir+"/g5ptqM.comp.spv").c_str(),3,16);  // batched native PTQ1_0
    Pipe pdeq5=mkpipe((shdir+"/deq_emb5.comp.spv").c_str(),3,12);
    Pipe pfwhi=mkpipe((shdir+"/fwht_hi.comp.spv").c_str(),1,8);
    Pipe pg2=mkpipe((shdir+"/g2pq.comp.spv").c_str(),3,8);
    Pipe pdeq2=mkpipe((shdir+"/deq_emb2.comp.spv").c_str(),3,16);
    Pipe pattnsg=mkpipe((shdir+"/attn_sg.comp.spv").c_str(),3,16);
    Pipe pattnvg=mkpipe((shdir+"/attn_vg.comp.spv").c_str(),3,16);
    fprintf(stderr,"all 19 pipelines created\n");

    auto Z=[&](int n){ return mkbuf((size_t)n*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); };
    // Hadamard folding. engine.py fold_args/fold_hi: for HB > 512 the kernels do UNSCALED
    // 512-point chunks (HB = -512) and fwht_hi then combines the HB/512 chunks and applies
    // 1/sqrt(HB). 2B has HB=512 (single stage); 27B has HB=1024 (needs the second stage).
    const int hbArg = (HB > 512) ? -512 : HB;
    auto foldGrid=[&](int width){ int ch=(hbArg<0?512:hbArg); return width/ch; };
    auto fold_hi=[&](Buf&buf, int width){
        if(HB <= 512) return;
        std::vector<Buf> bufs={buf};
        bindbufs(pfwhi,bufs);
        struct{int HBv,n;}pc{HB,width};
        int nc=HB>>9; int limit=(width>>9)/nc*512;
        dispatch(pfwhi,G.q0,G.cp,(limit+255)/256,1,1,&pc,8);
    };
    // fold_hi for one token's slice of a token-major buffer: same second stage, bound at
    // the token's byte offset so the chunked path needs no copy.
    auto fold_hi_off=[&](Buf&buf, int width, int tok, int stride){
        if(HB <= 512) return;
        std::vector<Buf> bufs={buf};
        std::vector<VkDeviceSize> offs={ (VkDeviceSize)tok*stride*4 };
        bindbufs(pfwhi,bufs,offs);
        struct{int HBv,n;}pc{HB,width};
        int nc=HB>>9; int limit=(width>>9)/nc*512;
        dispatch(pfwhi,G.q0,G.cp,(uint32_t)((limit+255)/256),1,1,&pc,8);
    };
    // xp is the shared "activation workspace" that receives every folded vector, so it must hold the
    // widest one. engine.py: work = max(D, NFF, NV*HK). The 2B hid the omission because there
    // NV*HK == D == 2048; on the 27B NV*HK = 48*128 = 6144 > D = 5120 and smfold overran by 1024
    // floats, leaving the last Hadamard block of every GDN layer uninitialised.
    const int work = ((D > NFF) ? D : NFF) > (NV*HK) ? ((D > NFF) ? D : NFF) : (NV*HK);
    Buf y=Z(VOCAB), xp=Z(work), h=Z(D), xn=Z(D), zv=Z(NV*HK), h2=Z(D), part=Z(4096);
    Buf qkv=Z(CONV_CH), cs=Z(CONV_CH), qkbuf=Z(2*NK*HK), qkvbuf=Z(NHKV*HD);
    Buf kn=Z(NHKV*HD), qn=Z(NH*2*HD), qr=Z(NH*HD), kro=Z(NHKV*HD);
    Buf fo=Z(NH*HD), fo6=Z(NFF), outgdn=Z(NV*HK), att=Z(NH*HD), ffg=Z(NFF), ffu=Z(NFF);
    Buf gb=Z(NV), gg=Z(NV), gout=Z(NV*HK);
    // chunked-path scratch: T tokens, token-major
    // T width: the chunked prefill width, but never below 2, because the speculative
    // verify pass runs T=2 regardless of KINDI_CHUNK.
    const int CT = (CHUNK>2) ? CHUNK : ((spec_n>0) ? 2 : 1);
    Buf hT=Z(D*CT), xnT=Z(D*CT), xpT=Z(work*CT), zvT=Z(NV*HK*CT), outT=Z(NV*HK*CT);
    Buf qkvT=Z(CONV_CH*CT), csT=Z(CONV_CH*CT), gbT=Z(NV*CT), ggT=Z(NV*CT);
    Buf ffgT=Z(NFF*CT), ffuT=Z(NFF*CT), fo6T=Z(NFF*CT), yT=Z(VOCAB*CT);
    Buf logT = (spec_n>0) ? Z(VOCAB*CT) : Z(4);   // verify logits, one row per token
    Buf h2T=Z(D*CT), qkbufT=Z(2*NK*HK*CT), goutT=Z(NV*HK*CT);
    // per-layer copies of the GDN inputs a speculative verify pass must keep so the
    // commit can replay the accepted prefix (see spec_replay)
    // Only allocated for --spec: they are pure commit-replay state, and on the 27B PQ2_0
    // the 7.2 GB of weights leave no room for anything that is not needed.
    std::vector<Buf> qkvL, csL, qkL, ggL, gbL;
    if(spec_n>0) for(int il=0;il<NL;++il){
        qkvL.push_back(Z(CONV_CH*CT));
        csL.push_back(Z(CONV_CH*CT));
        qkL.push_back(Z(2*NK*HK*CT));
        ggL.push_back(Z(NV*CT));
        gbL.push_back(Z(NV*CT));
    }
    Buf aout=mkbuf(4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buf logitsbuf=mkbuf((size_t)VOCAB*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buf sbuf=Z(NH*ctx), apart=Z(((ctx+255)/256)*NH*HD);

    // ---- MTP draft head (blk.<NL>; Q8_0 in the procreations file, Q4_0 in the MTP-Q4_0 one) ----
    // A full attention + FFN layer whose activations are NOT Hadamard-folded: the reference
    // feeds the plain normalised vector to its GEMVs (no signs, no FWHT), unlike the ternary
    // trunk. embed() already returns the primal-basis embedding (deq_emb* applies the inverse
    // transform), which is what nextn.enorm consumes.
    const bool have_mtp = (nextn_layers > 0) && codebufs.count("blk."+std::to_string(NL)+".attn_q.weight");
    const std::string mtp_p = "blk."+std::to_string(NL)+".";
    const std::string head_name = codebufs.count("output.weight") ? "output.weight" : "token_embd.weight";
    Buf mtp_hprev=Z(D), mtp_emb=Z(D), mtp_en=Z(D), mtp_hn=Z(D), mtp_cat=Z(2*D);
    Buf mtp_cur=Z(D), mtp_inpSA=Z(D), mtp_xn=Z(D);
    Buf mtp_qn=Z(2*NH*HD), mtp_qr=Z(NH*HD), mtp_fo=Z(NH*HD);
    Buf mtp_kro=Z(NHKV*HD), mtp_kn=Z(NHKV*HD), mtp_v=Z(NHKV*HD);
    Buf mtp_att=Z(NH*HD), mtp_y=Z(D), mtp_res=Z(D);
    Buf mtp_ffnxn=Z(D), mtp_ffg=Z(NFF), mtp_ffu=Z(NFF), mtp_mul=Z(NFF), mtp_ffn=Z(D), mtp_res2=Z(D);
    Buf mtp_hin=Z(D), mtp_hfold=Z(D), mtp_hnorm=Z(D), mtp_logits=Z(VOCAB);
    Buf mtp_sbuf=Z(NH*ctx), mtp_apart=Z(((ctx+255)/256)*NH*HD);
    Buf KcM=mkbuf(have_mtp?(VkDeviceSize)ctx*NHKV*HD*2:4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf VcM=mkbuf(have_mtp?(VkDeviceSize)ctx*NHKV*HD*2:4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf mtp_aout=mkbuf(64,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    fillbuf(mtp_hprev); fillbuf(KcM); fillbuf(VcM);
    if(have_mtp){
        // The MTP block's dtype varies by file (Q8_0 in the procreations model, Q4_0 in the
        // Abliterated MTP-Q4_0 one). Report what was actually loaded rather than a guess.
        const std::string aq = mtp_p+"attn_q.weight";
        const char* fmt = q4names.count(aq) ? "Q4_0"
                        : q8names.count(aq) ? "Q8_0"
                        : native_names.count(aq) ? "PTQ1_0"
                        : pq2_names.count(aq) ? "PQ2_0" : "4-trit";
        fprintf(stderr,"MTP draft head: %s (%s, unfolded basis, own KV cache)\n", mtp_p.c_str(), fmt);
    }

    std::vector<Buf> M, convv, Kc, Vc;
    for(int i=0;i<NATT;++i){ Kc.push_back(mkbuf((size_t)ctx*NHKV*HD*2,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)); Vc.push_back(mkbuf((size_t)ctx*NHKV*HD*2,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)); }
    for(int i=0;i<NL;++i){ M.push_back(mkbuf((size_t)NV*HK*HK*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)); convv.push_back(mkbuf((size_t)3*CONV_CH*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)); }
    for(auto&m:M) fillbuf(m);
    for(auto&c:convv) fillbuf(c);
    for(auto&k:Kc) fillbuf(k);
    for(auto&v:Vc) fillbuf(v);

    // signs from GGUF prism.hadamard
    std::map<int,std::vector<float>> signsmap;
    {
        size_t off=0;
        for(int w : sign_widths){
            signsmap[w].assign(sign_values.begin()+off, sign_values.begin()+off+w);
            off += w;
        }
    }
    std::vector<float> sD = signsmap.count(D) ? signsmap[D] : std::vector<float>(D,1.0f);
    if((int)sD.size() < D) sD.resize(D,1.0f);
    std::vector<float> sF = signsmap.count(NFF) ? signsmap[NFF] : std::vector<float>(NFF,1.0f);
    if((int)sF.size() < NFF) sF.resize(NFF,1.0f);
    // sV/sA carry their own widths in prism.hadamard.sign_widths (2B: 2048/2048, 27B: 6144/6144);
    // only tile from sD when the GGUF does not carry that width.
    std::vector<float> sV = signsmap.count(NV*HK) ? signsmap[NV*HK] : std::vector<float>(NV*HK,1.0f);
    if((int)sV.size() < NV*HK){ size_t n0=sV.size(); sV.resize(NV*HK,1.0f); for(int i=(int)n0;i<NV*HK;++i) sV[i]=sD[i%D]; }
    std::vector<float> sA = signsmap.count(NH*HD) ? signsmap[NH*HD] : std::vector<float>(NH*HD,1.0f);
    if((int)sA.size() < NH*HD){ size_t n0=sA.size(); sA.resize(NH*HD,1.0f); for(int i=(int)n0;i<NH*HD;++i) sA[i]=sD[i%D]; }
    Buf sDb=mkbuf(D*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&sDb,sD.data(),D*4);

    Buf sVb=mkbuf(NV*HK*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&sVb,sV.data(),NV*HK*4);
    Buf sAb=mkbuf(NH*HD*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&sAb,sA.data(),NH*HD*4);
    Buf sFb=mkbuf(NFF*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&sFb,sF.data(),NFF*4);

    std::vector<float> cosv(ctx*(rope_dim/2)), sinv(ctx*(rope_dim/2));
    std::vector<double> invfreq(rope_dim/2);
    for(int i=0;i<rope_dim/2;++i) invfreq[i]=pow(1e7,-2.0*i/rope_dim);
    for(int p=0;p<ctx;++p) for(int i=0;i<rope_dim/2;++i){ double th=p*invfreq[i]; cosv[p*(rope_dim/2)+i]=cos(th); sinv[p*(rope_dim/2)+i]=sin(th); }
    Buf cost=mkbuf(cosv.size()*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&cost,cosv.data(),cosv.size()*4);
    Buf sint=mkbuf(sinv.size()*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&sint,sinv.data(),sinv.size()*4);

    // ---- helper: GEMV ----
    // engine.py column-split rule: single tensor, nsm>1, (R0>>4) < split_groups*nsm and (C>>9) >= nsm
    // -> g4 with gridY=nsm into part, return nsm (rmsfold consumes parts, else sumparts folds them).
    // g4simple.comp has no gridY support, so on the fallback path we dispatch the normal GEMV into
    // outbuf and the caller sets pending=1. True column-split needs g4.comp (exists in shaders/g4.comp
    // + .spv, 11 bindings incl. tables scratch, 36B push with gridY) which is not wired yet.
    const int nsm = 1;            // single compute queue path; g4simple grid is (R,1,1) so no split
    const int split_groups = 128; // KINDI_SPLIT_GROUPS default, kept for the rule check below
    auto gemv=[&](const std::string&wname, Buf&xbuf, Buf&outbuf){
        auto [rows,cols]=tinfo[wname];
        // Column-split check (engine.py rule; always false while nsm==1):
        if(nsm>1 && (rows>>4) < split_groups*nsm && (cols>>9) >= nsm){
            // True path needs g4.comp (gridY=nsm into part, pending=nsm, rmsfold consumes parts).
            // g4simple has no gridY support, so fall through to the normal dispatch below
            // (caller sets pending=1; rmsfold fuses the single residual).
        }
        // Parts path would be: bind g4.comp with gridY=nsm into part, pending=nsm (skip sumparts,
        // rmsfold sums parts). Fallback used here: normal g4simple dispatch, caller sets pending=1.
        if(pq2_names.count(wname)){
            std::vector<Buf> bufs={codebufs[wname],xbuf,outbuf};
            bindbufs(pg2,bufs);
            struct{int R,C;}pc{rows,cols};
            dispatch(pg2,G.q0,G.cp,rows,1,1,&pc,8);
            return;
        }
        if(native_names.count(wname)){
            std::vector<Buf> bufs={codebufs[wname],xbuf,outbuf};
            struct{int R,C;}pc{rows,cols};
            // Default: byte-granular decode (one load per byte, five weights).
            // KINDI_G5=byte4 selects the 4-rows-per-workgroup shape, which is a
            // further 1.09x but sums the lanes with subgroupAdd instead of the
            // shared tree, so it is not bit-identical to g5ptqM and would need its
            // own spec-equivalence proof. KINDI_G5=old keeps the previous shader.
            const char* g5 = getenv("KINDI_G5");
            if(g5 && !strcmp(g5,"byte4")){
                bindbufs(pg5b4,bufs);
                dispatch(pg5b4,G.q0,G.cp,(uint32_t)((rows+3)/4),1,1,&pc,8);
            } else if(g5 && !strcmp(g5,"old")){
                bindbufs(pg5,bufs);
                dispatch(pg5,G.q0,G.cp,rows,1,1,&pc,8);
            } else {
                bindbufs(pg5b,bufs);
                dispatch(pg5b,G.q0,G.cp,rows,1,1,&pc,8);
            }
            return;
        }
        if(q4names.count(wname)){
            std::vector<Buf> bufs={codebufs[wname],xbuf,outbuf};
            bindbufs(pq4f,bufs);
            struct{int R,C,T,xstride;}pc{rows,cols,1,cols};
            dispatch(pq4f,G.q0,G.cp,rows,1,1,&pc,16);
            return;
        }
        if(q8names.count(wname)){
            std::vector<Buf> bufs={codebufs[wname],scalebufs[wname],xbuf,outbuf};
            bindbufs(pq8f,bufs);
            struct{int R,C,T,xstride;}pc{rows,cols,1,cols};
            dispatch(pq8f,G.q0,G.cp,(uint32_t)((rows+3)/4),1,1,&pc,16);
            return;
        }
        std::vector<Buf> bufs={codebufs[wname],scalebufs[wname],xbuf,outbuf};
        bindbufs(pg4,bufs);
        struct{int R,C;}pc{rows,cols};
        // g4lut2: 4 rows per workgroup (one per 64-wide wave), so ceil(rows/4) groups
        dispatch(pg4,G.q0,G.cp,(uint32_t)((rows+3)/4),1,1,&pc,8);
    };
    // Up to three ternary tensors with the SAME input in one launch (engine.py gemv).
    // Tensors that share an input have no dependency on each other, so the per-dispatch
    // barrier between them was pure cost. Falls back to individual gemv calls when any
    // tensor is not on the 4-trit path or the column counts differ.
    auto gemv3=[&](Buf&xbuf, const std::string&n0, Buf*o0,
                             const std::string&n1, Buf*o1,
                             const std::string&n2, Buf*o2)->bool{
        auto usable=[&](const std::string&n){ return !n.empty() && codebufs.count(n)
                    && scalebufs.count(n) && !pq2_names.count(n) && !native_names.count(n)
                    && tinfo.count(n); };
        std::vector<std::string> ns; std::vector<Buf*> os;
        for(int i=0;i<3;++i){
            const std::string&n = i==0?n0:(i==1?n1:n2);
            Buf*o = i==0?o0:(i==1?o1:o2);
            if(n.empty()) continue;
            if(!usable(n)) return false;
            if(!ns.empty() && tinfo[n].second != tinfo[ns[0]].second) return false;
            ns.push_back(n); os.push_back(o);
        }
        if(ns.size()<2) return false;
        const int C = tinfo[ns[0]].second;
        int R[3]={0,0,0};
        for(size_t i=0;i<ns.size();++i) R[i]=tinfo[ns[i]].first;
        std::vector<Buf> bufs={
            codebufs[ns[0]], scalebufs[ns[0]], *os[0],
            codebufs[ns.size()>1?ns[1]:ns[0]], scalebufs[ns.size()>1?ns[1]:ns[0]], *(os.size()>1?os[1]:os[0]),
            codebufs[ns.size()>2?ns[2]:ns[0]], scalebufs[ns.size()>2?ns[2]:ns[0]], *(os.size()>2?os[2]:os[0]),
            xbuf };
        bindbufs(pg4m,bufs);
        struct{int R0,R1,R2,C;}pc{R[0],R[1],R[2],C};
        int total=R[0]+R[1]+R[2];
        dispatch(pg4m,G.q0,G.cp,(uint32_t)((total+3)/4),1,1,&pc,16);
        return true;
    };
    // T tokens against ONE weight stream (g4lutM). x is token-major [T][C], y is [T][R].
    // With T=1 this is arithmetically identical to gemv on the 4-trit path, which is the
    // regression gate for the batched path.
    auto gemvM=[&](const std::string&wname, Buf&xbuf, Buf&outbuf, int T, int xstride=-1)->bool{
        if(T < 1 || T > 4) return false;
        if(!codebufs.count(wname) || !scalebufs.count(wname)) return false;
        if(pq2_names.count(wname) || native_names.count(wname)) return false;
        auto it = tinfo.find(wname);
        if(it == tinfo.end()) return false;
        int rows = it->second.first, cols = it->second.second;
        std::vector<Buf> bufs={codebufs[wname],scalebufs[wname],outbuf,xbuf};
        bindbufs(pg4M,bufs);
        if(xstride<0) xstride=cols;
        struct{int R,C,T,xstride;}pc{rows,cols,T,xstride};
        dispatch(pg4M,G.q0,G.cp,(uint32_t)((rows+3)/4),1,1,&pc,16);
        return true;
    };
    // Same contract for PQ2_0 (27B): T tokens against one packed block stream.
    auto gemvM2pq=[&](const std::string&wname, Buf&xbuf, Buf&outbuf, int T, int xstride=-1)->bool{
        if(T < 1 || T > 4) return false;
        if(!pq2_names.count(wname)) return false;
        auto it = tinfo.find(wname);
        if(it == tinfo.end()) return false;
        int rows = it->second.first, cols = it->second.second;
        if(xstride<0) xstride=cols;
        std::vector<Buf> bufs={codebufs[wname],xbuf,outbuf};
        bindbufs(pg2M,bufs);
        struct{int R,C,T,xstride;}pc{rows,cols,T,xstride};
        dispatch(pg2M,G.q0,G.cp,(uint32_t)((rows+3)/4),1,1,&pc,16);
        return true;
    };
    // sumparts fold for pending>1 produced outside rmsfold (engine.py gemv parts_ok=false path).
    // Not triggered while nsm==1 (pending is only ever 0/1), wired here so the pipeline is live.
    auto sumparts=[&](Buf&p, Buf&yout, int R, int n){
        std::vector<Buf> bufs={p,yout};
        bindbufs(psumparts,bufs);
        struct{int Rr,nn;}pc{R,n};
        dispatch(psumparts,G.q0,G.cp,(R+255)/256,1,1,&pc,8);
    };

    // ---- embed ----
    // one reusable staging buffer (the 27B is ~5.8 GB resident on an 8 GB card; a per-probe
    // allocation OOMs). Sized on first use to the largest probe seen.
    Buf stage{};
    auto readbuf=[&](Buf&b, int n)->std::vector<float>{
        if(stage.buf==VK_NULL_HANDLE || stage.size < (VkDeviceSize)n*4){
            if(stage.buf) freebuf(stage);
            stage=mkbuf((VkDeviceSize)n*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        copybuf(b,stage,(size_t)n*4);
        std::vector<float> v((float*)stage.mapped, (float*)stage.mapped+n);
        return v;
    };
    auto dumpv=[&](const char*tag, Buf&b, int n){
        std::vector<float> v=readbuf(b,n);
        fprintf(stderr,"G %s: %.4f %.4f %.4f\n",tag,v[0],v[1],v[2]);
    };
    // full-vector dump for offline verification of the Hadamard fold
    auto dumpfull=[&](const char*tag, Buf&b, int n){
        if(!dumpvec) return;
        static std::map<std::string,int> seen;
        if(seen[tag]++ > 0) return;      // first occurrence only (token 0, layer 0)
        std::vector<float> v=readbuf(b,n);
        std::string fn=std::string("dv_")+tag+".bin";
        FILE*f=fopen(fn.c_str(),"wb");
        if(f){ fwrite(v.data(),4,v.size(),f); fclose(f); fprintf(stderr,"G full %s: %d floats -> %s\n",tag,n,fn.c_str()); }
    };
    // dst: where the row lands (default the trunk residual h). The MTP draft needs the
    // embedding without clobbering h, so it passes its own buffer.
    auto embed=[&](int tok, Buf* dst=nullptr){
        Buf& tgt = dst ? *dst : h;
        static bool emb_reported = false;
        if(!emb_reported){
            emb_reported = true;
            fprintf(stderr, "embed path: %s\n",
                pq2_names.count("token_embd.weight") ? "deq_emb2 (PQ2_0)" :
                native_names.count("token_embd.weight") ? "deq_emb5 (native PTQ1_0)" : "deq_emb3 (expanded)");
        }
        if(pq2_names.count("token_embd.weight")){
            std::vector<Buf> bufs={codebufs["token_embd.weight"],sDb,tgt};
            bindbufs(pdeq2,bufs);
            struct{int row,C,HBv,raw;}pc{tok,D,HB,raw_embed};
            dispatch(pdeq2,G.q0,G.cp,1,1,1,&pc,16);
            if(dump) dumpv("embed_h",tgt,D);
            return;
        }
        if(native_names.count("token_embd.weight")){
            std::vector<Buf> bufs={codebufs["token_embd.weight"],sDb,tgt};
            bindbufs(pdeq5,bufs);
            struct{int row,C,HBv;}pc{tok,D,HB};
            dispatch(pdeq5,G.q0,G.cp,1,1,1,&pc,12);
            if(dump) dumpv("embed_h",tgt,D);
            return;
        }
        std::vector<Buf> bufs={codebufs["token_embd.weight"],scalebufs["token_embd.weight"],sDb,tgt};
        bindbufs(pdeq,bufs);
        struct{int row,C;}pc{tok,D};
        dispatch(pdeq,G.q0,G.cp,1,1,1,&pc,8);
        if(dump) dumpv("embed_h",tgt,D);
    };

    // ---- rmsfold ----
    int pending=0;
    auto rmsfold=[&](Buf&x, Buf&r, Buf&xw, Buf&xn, Buf&xo, const std::string&wname, int C, int HBv){
        // pending: 0 = no residual, 1 = single GEMV output in y, >1 = column-split in part
        Buf& resbuf = (pending>1) ? part : y;
        std::vector<Buf> bufs={x,resbuf,xw,f32bufs[wname],sDb,xn,xo};
        bindbufs(prmsfold,bufs);
        (void)HBv;
        struct{int Cc,add,HB;}pc{C,pending,hbArg};
        dispatch(prmsfold,G.q0,G.cp,foldGrid(C),1,1,&pc,12);
        fold_hi(xo, C);
        if(pending>0){ std::swap(h,h2); }
        pending=0;
    };

    // ---- gdn layer ----
    auto gdn=[&](int il, int pos){
        std::string p="blk."+std::to_string(il)+".";
        rmsfold(h, part, h2, xn, xp, p+"attn_norm.weight", D, HB);
        if(dump&&il==0){ dumpv("xn",xn,D); dumpv("xp",xp,D); }
        // Three independent dispatches. qkv and z both read xp; the gate scalars read
        // xn. None of them reads another's output, so they need no barrier between
        // them: one barrier after the group covers the consumers below (gdn_conv reads
        // qkv, gdn_delta reads gb/gg). On the 5-trit path gemv3 never fires because it
        // requires scalebufs, so without this the 27B paid a drain per projection.
        // Opt-in only. Measured 6082 vs 6132 ms, 0.8%, for removing two barriers
        // per GDN layer. Too small to justify dropping a memory-ordering guarantee
        // by default, and 1920 of 26660 barriers is only 7% of them, so the linear
        // ceiling for this approach is about 694 ms of the 1180 ms NO_BARRIER win.
        const bool g5sib = getenv("KINDI_SIB_BARRIER") != nullptr;
        if(!gemv3(xp, p+"attn_qkv.weight",&qkv, p+"attn_gate.weight",&zv, "", nullptr)){
            if(g5sib){
                // gemv() ends in dispatch(); set the flag so the first one keeps its
                // barrier (it follows the norm) and the second does not need one.
                g_nobar_next = false;
                gemv(p+"attn_qkv.weight", xp, qkv);
                g_nobar_next = true;
                gemv(p+"attn_gate.weight", xp, zv);
                g_nobar_next = false;
            } else {
                gemv(p+"attn_qkv.weight", xp, qkv);
                gemv(p+"attn_gate.weight", xp, zv);
            }
        }
        if(dump&&il==0){ dumpv("qkv",qkv,CONV_CH); dumpv("z",zv,NV*HK); }
        // gdn_gates: reads xn only, so it is the third member of the same group.
        {
            std::vector<Buf> bufs={xn, f32bufs[p+"ssm_beta.weight"], f32bufs[p+"ssm_alpha.weight"],
                f32bufs[p+"ssm_dt.bias"], f32bufs[p+"ssm_a"], gb, gg};
            bindbufs(pgates,bufs);
            struct{int n;}pc{D};
            if(g5sib) dispatch_nb(pgates,G.q0,G.cp,NV,1,1,&pc,4);
            else      dispatch(pgates,G.q0,G.cp,NV,1,1,&pc,4);
            if(dump&&il==0){ dumpv("beta",gb,NV); dumpv("gate",gg,NV); }
        }
        if(g5sib) barrier_now();
        // gdn_conv
        {
            std::vector<Buf> bufs={qkv, convv[il], f32bufs[p+"ssm_conv1d.weight"], cs};
            bindbufs(pconv,bufs);
            struct{int C;}pc{CONV_CH};
            dispatch(pconv,G.q0,G.cp,(CONV_CH+255)/256,1,1,&pc,4);
            if(dump&&il==0) dumpv("conv_silu",cs,CONV_CH);
        }
        // l2w
        {
            std::vector<Buf> bufs={cs, qkbuf};
            bindbufs(pl2w,bufs);
            struct{int n,rows;}pc{HK, 2*NK};
            dispatch(pl2w,G.q0,G.cp,(2*NK+3)/4,1,1,&pc,8);
            if(dump&&il==0) dumpv("l2w_qk",qkbuf,2*NK*HK);
        }
        // gdn_delta
        {
            std::vector<Buf> bufs={M[il], qkbuf, qkbuf, cs, gg, gb, gout};
            std::vector<VkDeviceSize> offs={0, 0, (VkDeviceSize)NK*HK*4, (VkDeviceSize)2*NK*HK*4, 0, 0, 0};
            bindbufs(pdelta,bufs,offs);
            struct{float scale; int nkh;}pc{1.0f/sqrtf((float)HK), NK};
            dispatch(pdelta,G.q0,G.cp,NV,1,1,&pc,8);
            if(dump&&il==0) dumpv("delta_out",gout,NV*HK);
        }
        // rmsw
        {
            std::vector<Buf> bufs={gout, f32bufs[p+"ssm_norm.weight"], outgdn};
            bindbufs(prmsw,bufs);
            struct{int n,rows;}pc{HK, NV};
            dispatch(prmsw,G.q0,G.cp,(NV+3)/4,1,1,&pc,8);
            if(dump&&il==0) dumpv("gdn_norm",outgdn,NV*HK);
            if(il==0){ dumpfull("z",zv,NV*HK); dumpfull("gdn_norm",outgdn,NV*HK); }
        }
        // smfold
        {
            std::vector<Buf> bufs={zv, outgdn, sVb, xp};
            bindbufs(psmfold,bufs);
            // perm_rep = NV/NK: the grouped-V-head permutation. Confirmed from the reference graph
            // (llama-eval-callback, 27B PQ2_0, token 0):
            //   node_61 SWIGLU(z, norm)   {128, 48, 5}
            //   RESHAPE                   {128, 16, 3, 5}   [hd, num_k_heads=16, rep=3]
            //   PERMUTE                   {128, 3, 16, 5}   <- swaps nk and rep
            //   MUL(..., signs.6144) then MUL_MAT(hadamard.1024) then MUL_MAT(ssm_out)
            // That maps dst d = i0 + 128*rep + 384*nk from src i0 + 128*nk + 2048*rep, which is
            // exactly smfold's src = hd + perm_hd*(nk + perm_nk*rep) with
            // perm_hd = HK = 128, perm_rep = NV/NK = 3, perm_nk = NK = 16.
            // The 2B has NV == NK, so perm_rep = 1 and the branch is identity there.
            struct{int HB,phd,pnk,prep;}pc{hbArg, HK, NK, NV/NK};
            dispatch(psmfold,G.q0,G.cp,foldGrid(NV*HK),1,1,&pc,16);
            if(dump&&il==0) dumpv("gdn_swiglu_unfolded",xp,NV*HK);
            if(il==0){ dumpfull("swiglu_unfolded",xp,NV*HK); }
            fold_hi(xp, NV*HK);
            if(dump&&il==0) dumpv("gdn_folded",xp,NV*HK);
            if(il==0){ dumpfull("gdn_folded",xp,NV*HK); }
        }
        // gemv ssm_out
        if(!gemvM(p+"ssm_out.weight", xp, y, 1)) gemv(p+"ssm_out.weight", xp, y);
        if(dump&&il==0) dumpv("ssm_out",y,D);
        if(il==0){ dumpfull("ssm_out",y,D); }
        pending=1;
    };

    // ---- attention layer ----
    auto attn=[&](int il, int pos, int kvi){
        std::string p="blk."+std::to_string(il)+".";
        rmsfold(h, part, h2, xn, xp, p+"attn_norm.weight", D, HB);
        if(!gemv3(xp, p+"attn_q.weight",&qn, p+"attn_k.weight",&kro, p+"attn_v.weight",&qkvbuf)){
            gemv(p+"attn_q.weight", xp, qn);
            gemv(p+"attn_k.weight", xp, kro);
            gemv(p+"attn_v.weight", xp, qkvbuf);
        }
        // cpq: qn rows [q|gate] -> q half only (qr gets raw q, gate stays in qn second half)
        {
            std::vector<Buf> bufs={qn, qr};
            bindbufs(pcpq,bufs);
            struct{int rows,hd;}pc{NH, HD};
            dispatch(pcpq,G.q0,G.cp,NH,1,1,&pc,8);
        }
        // rms q norm: qr -> fo (CUDA: q = rms_rows(qc, q_norm, NH, HD, fo))
        {
            std::vector<Buf> bufs={qr, f32bufs[p+"attn_q_norm.weight"], fo};
            bindbufs(prmsw,bufs);
            struct{int n,rows;}pc{HD, NH};
            dispatch(prmsw,G.q0,G.cp,(NH+7)/8,1,1,&pc,8);
        }
        // rope on q (NH rows: fo -> qr) with pos offset
        {
            std::vector<Buf> bufs={fo, cost, sint, qr};
            std::vector<VkDeviceSize> offs={0, (VkDeviceSize)pos*(rope_dim/2)*4, (VkDeviceSize)pos*(rope_dim/2)*4, 0};
            bindbufs(prope,bufs,offs);
            struct{int v;}pc{HD};
            dispatch(prope,G.q0,G.cp,NH,1,1,&pc,4);
        }
        // rms k norm: kro -> kn (CUDA: Kn = rms_rows(kro, k_norm, NHKV, HD, kn))
        {
            std::vector<Buf> bufs={kro, f32bufs[p+"attn_k_norm.weight"], kn};
            bindbufs(prmsw,bufs);
            struct{int n,rows;}pc{HD, NHKV};
            dispatch(prmsw,G.q0,G.cp,(NHKV+7)/8,1,1,&pc,8);
        }
        // rope on k (NHKV rows: kn -> kro) with pos offset; kro then holds roped Kn for store
        {
            std::vector<Buf> bufs={kn, cost, sint, kro};
            std::vector<VkDeviceSize> offs={0, (VkDeviceSize)pos*(rope_dim/2)*4, (VkDeviceSize)pos*(rope_dim/2)*4, 0};
            bindbufs(prope,bufs,offs);
            struct{int v;}pc{HD};
            dispatch(prope,G.q0,G.cp,NHKV,1,1,&pc,4);
        }
        // store K and V into cache (fp16)
        // K cache: Kc[kvi][pos*NHKV*HD:(pos+1)*NHKV*HD] = kro
        // store_h: float* x -> uint* dst, n elements
        {
            std::vector<Buf> bufs={kro, Kc[kvi]};
            bindbufs(pstore,bufs,{0, (VkDeviceSize)pos*NHKV*HD*2});
            struct{int v;}pc{NHKV*HD};
            dispatch(pstore,G.q0,G.cp,(NHKV*HD/2+255)/256,1,1,&pc,4);
        }
        {
            std::vector<Buf> bufs={qkvbuf, Vc[kvi]};
            bindbufs(pstore,bufs,{0, (VkDeviceSize)pos*NHKV*HD*2});
            struct{int v;}pc{NHKV*HD};
            dispatch(pstore,G.q0,G.cp,(NHKV*HD/2+255)/256,1,1,&pc,4);
        }
        // attn_s4w: scores = Q @ K
        // sbuf layout: [NH*n] (n=pos+1)
        int n = pos+1;
        int Grp=NH/NHKV;   // q heads per KV head (G is the GPU context)
        if(Grp==4 && !force_gen){
            // specialised 2B path (NH=8, NHKV=2, G=4): 16-value butterfly, fastest
            std::vector<Buf> bufs={Kc[kvi], qr, sbuf};
            bindbufs(pattns4,bufs);
            struct{int nn,nkv;}pc{n, NHKV};
            dispatch(pattns4,G.q0,G.cp,NHKV,(n+31)/32,1,&pc,8);
        } else {
            // generic GQA (4B/9B/27B: G=6 or 8): per-head dot + subgroup reduce
            std::vector<Buf> bufs={Kc[kvi], qr, sbuf};
            bindbufs(pattnsg,bufs);
            struct{int nn,nkv,g; float sc;}pc{n, NHKV, Grp, 1.0f/sqrtf((float)HD)};
            dispatch(pattnsg,G.q0,G.cp,NHKV,(n+31)/32,1,&pc,16);
        }
        // softmax_rows: sbuf (NH rows x n)
        {
            std::vector<Buf> bufs={sbuf};
            bindbufs(psoft,bufs);
            struct{int v;}pc{n};
            dispatch(psoft,G.q0,G.cp,NH,1,1,&pc,4);
        }
        // attn_v4p: sbuf + Vc -> apart
        // nch = (n+255)/256
        int nch = (n+255)/256;
        if(Grp==4 && !force_gen){
            std::vector<Buf> bufs={sbuf, Vc[kvi], apart};
            bindbufs(pattnv4,bufs);
            struct{int nn,nh,nkv;}pc{n, NH, NHKV};
            dispatch(pattnv4,G.q0,G.cp,NHKV,1,nch,&pc,12);
        } else {
            std::vector<Buf> bufs={sbuf, Vc[kvi], apart};
            bindbufs(pattnvg,bufs);
            struct{int nn,nh,nkv,g;}pc{n, NH, NHKV, Grp};
            dispatch(pattnvg,G.q0,G.cp,NHKV,1,nch,&pc,16);
        }
        // attn_comb: apart -> att (NH*HD); gatev = qn base (qn rows are [q|gate],
        // shader reads gatev[hh*512+256+f] = row hh gate half; qn untouched since gemv)
        {
            std::vector<Buf> bufs={apart, qn, att};
            bindbufs(pattncomb,bufs);
            struct{int nch,nh;}pc{nch, NH};
            dispatch(pattncomb,G.q0,G.cp,NH,1,1,&pc,8);
        }
        // fold_fwht: att -> xp (CUDA: k_fold_fwht(att, sA, xp, hb); HB=512 so C/HB groups)
        if(dump&&pos==0&&il==(interval-1)){ dumpv("q_rope",qr,NH*HD); dumpv("k_rope",kro,NHKV*HD); dumpv("scores",sbuf,NH); dumpv("comb",att,NH*HD); }
        {
            std::vector<Buf> bufs={att, sAb, xp};
            bindbufs(pfwh,bufs);
            struct{int v;}pc{hbArg};
            dispatch(pfwh,G.q0,G.cp,foldGrid(NH*HD),1,1,&pc,4);
            fold_hi(xp, NH*HD);
        }
        if(!gemvM(p+"attn_output.weight", xp, y, 1)) gemv(p+"attn_output.weight", xp, y);
        pending=1;
    };

    // ---- ffn ----
    auto ffn=[&](int il){
        std::string p="blk."+std::to_string(il)+".";
        rmsfold(h, part, h2, xn, xp, p+"post_attention_norm.weight", D, HB);
        if(!gemv3(xp, p+"ffn_gate.weight",&ffg, p+"ffn_up.weight",&ffu, "", nullptr)){
            gemv(p+"ffn_gate.weight", xp, ffg);
            gemv(p+"ffn_up.weight", xp, ffu);
        }
        if(dump&&il==0){ dumpv("ffn_norm_xp",xp,D); dumpv("ffn_gate",ffg,NFF); dumpv("ffn_up",ffu,NFF); }
        // smfold
        {
            std::vector<Buf> bufs={ffg, ffu, sFb, fo6};
            bindbufs(psmfold,bufs);
            struct{int HB,phd,pnk,prep;}pc{hbArg, 0, 0, 0};
            dispatch(psmfold,G.q0,G.cp,foldGrid(NFF),1,1,&pc,16);
            fold_hi(fo6, NFF);
        }
        if(dump&&il==0) dumpv("ffn_swiglu_folded",fo6,NFF);
        if(!gemvM(p+"ffn_down.weight", fo6, y, 1)) gemv(p+"ffn_down.weight", fo6, y);
        if(dump&&il==0) dumpv("ffn_out",y,D);
        pending=1;
    };

    // ---- MTP draft block ----
    // Mirrors llama_model_qwen35::graph_mtp (qwen35.cpp:694-944):
    //   e_norm = rms(emb(x_p),  nextn.enorm)      h_norm = rms(h_{p-1}, nextn.hnorm)
    //   cur = eh_proj @ [e_norm ; h_norm]         (Q8_0)
    //   q_full = attn_q @ rms(cur, attn_norm)     q = first half, gate = second half
    //   q,k = rms(per-head) + rope(pos);  v = attn_v @ xn
    //   att = attention(q,k,v) * sigmoid(gate);  cur += attn_output @ att
    //   cur += ffn_down @ (silu(ffn_gate @ xn2) * (ffn_up @ xn2))
    //   logits = head @ rms(cur, nextn.shared_head_norm)
    // No fold anywhere: the Q8_0 weights were not trained folded.
    auto mtp_rmsw=[&](Buf& x, const std::string& wname, Buf& out, int rows, int n, size_t outOff=0){
        std::vector<Buf> bufs={x, f32bufs[wname], out};
        bindbufs(prmsw,bufs,{0,0,outOff});
        struct{int n,rows;}pc{n,rows};
        dispatch(prmsw,G.q0,G.cp,(uint32_t)((rows+7)/8),1,1,&pc,8);
    };
    auto mtp_addv=[&](Buf& a, Buf& b, Buf& out, int n){
        std::vector<Buf> bufs={a,b,out};
        bindbufs(paddv,bufs);
        struct{int n;}pc{n};
        dispatch(paddv,G.q0,G.cp,(uint32_t)((n+255)/256),1,1,&pc,4);
    };
    auto dbgv=[&](const char* tag, Buf& b, int n){
        if(!dump) return;
        std::vector<float> v=readbuf(b,n);
        double mx=0,sum=0; int nf=0;
        for(float x: v){ if(!std::isfinite(x)) nf++; double a=std::fabs(x); if(a>mx) mx=a; sum+=a; }
        fprintf(stderr,"M %-12s n=%d first %.5f %.5f %.5f | max %.4f mean %.5f nonfinite %d\n",
                tag,n,v[0],v[1],v[2],mx,sum/n,nf);
    };
    // The draft's h_input is the trunk's t_h_nextn, i.e. rms(h, output_norm) (qwen35.cpp:287-290),
    // NOT the raw residual: the reference publishes it after the final output norm.
    auto mtp_hinput=[&](Buf& hres, const std::string& wname, Buf& out){
        mtp_rmsw(hres, wname, out, 1, D);
    };
    auto mtp_block=[&](Buf& hprev, int tok, int pos, bool want_logits)->int{
        embed(tok, &mtp_emb);
        // concat = [e_norm ; h_norm]: write the two halves straight into the GEMV input.
        // (Copying them with copyB needs a compute->transfer barrier; the earlier version
        //  read uninitialised device memory and fed NaNs to eh_proj.)
        mtp_rmsw(mtp_emb, mtp_p+"nextn.enorm.weight", mtp_cat, 1, D, 0);
        mtp_rmsw(hprev,  mtp_p+"nextn.hnorm.weight", mtp_cat, 1, D, (size_t)D*4);
        if(dump&&pos<10){ dbgv("emb",mtp_emb,D); dbgv("cat",mtp_cat,2*D); dbgv("hprev",hprev,D); }
        gemv(mtp_p+"nextn.eh_proj.weight", mtp_cat, mtp_cur);
        if(dumpvec&&pos==4){ dumpfull("mtp_hprev4",hprev,D); dumpfull("mtp_cat4",mtp_cat,2*D);
                             dumpfull("mtp_cur4",mtp_cur,D); }
        if(dump&&pos<10) dbgv("cur",mtp_cur,D);
        mtp_rmsw(mtp_cur, mtp_p+"attn_norm.weight", mtp_xn, 1, D);
        if(dump&&pos<10) dbgv("xn",mtp_xn,D);
        gemv(mtp_p+"attn_q.weight", mtp_xn, mtp_qn);
        gemv(mtp_p+"attn_k.weight", mtp_xn, mtp_kro);
        gemv(mtp_p+"attn_v.weight", mtp_xn, mtp_v);
        if(dump&&pos<10){ dbgv("qfull",mtp_qn,2*NH*HD); dbgv("kraw",mtp_kro,NHKV*HD); dbgv("vraw",mtp_v,NHKV*HD); }
        {   // q half of [q|gate]
            std::vector<Buf> bufs={mtp_qn, mtp_qr};
            bindbufs(pcpq,bufs);
            struct{int rows,hd;}pc{NH, HD};
            dispatch(pcpq,G.q0,G.cp,NH,1,1,&pc,8);
        }
        mtp_rmsw(mtp_qr, mtp_p+"attn_q_norm.weight", mtp_fo, NH, HD);
        {   // rope q at pos
            std::vector<Buf> bufs={mtp_fo, cost, sint, mtp_qr};
            std::vector<VkDeviceSize> offs={0, (VkDeviceSize)pos*(rope_dim/2)*4, (VkDeviceSize)pos*(rope_dim/2)*4, 0};
            bindbufs(prope,bufs,offs);
            struct{int v;}pc{HD};
            dispatch(prope,G.q0,G.cp,NH,1,1,&pc,4);
        }
        mtp_rmsw(mtp_kro, mtp_p+"attn_k_norm.weight", mtp_kn, NHKV, HD);
        {   // rope k at pos
            std::vector<Buf> bufs={mtp_kn, cost, sint, mtp_kro};
            std::vector<VkDeviceSize> offs={0, (VkDeviceSize)pos*(rope_dim/2)*4, (VkDeviceSize)pos*(rope_dim/2)*4, 0};
            bindbufs(prope,bufs,offs);
            struct{int v;}pc{HD};
            dispatch(prope,G.q0,G.cp,NHKV,1,1,&pc,4);
        }
        {   // K/V into the MTP layer's own fp16 cache
            std::vector<Buf> bufs={mtp_kro, KcM};
            bindbufs(pstore,bufs,{0,(VkDeviceSize)pos*NHKV*HD*2});
            struct{int v;}pc{NHKV*HD};
            dispatch(pstore,G.q0,G.cp,(NHKV*HD/2+255)/256,1,1,&pc,4);
        }
        {
            std::vector<Buf> bufs={mtp_v, VcM};
            bindbufs(pstore,bufs,{0,(VkDeviceSize)pos*NHKV*HD*2});
            struct{int v;}pc{NHKV*HD};
            dispatch(pstore,G.q0,G.cp,(NHKV*HD/2+255)/256,1,1,&pc,4);
        }
        int nn=pos+1, Grp=NH/NHKV, nch=(nn+255)/256;
        {   // scores = q . k
            std::vector<Buf> bufs={KcM, mtp_qr, mtp_sbuf};
            bindbufs(pattnsg,bufs);
            struct{int n,nkv,g; float sc;}pc{nn, NHKV, Grp, 1.0f/sqrtf((float)HD)};
            dispatch(pattnsg,G.q0,G.cp,NHKV,(uint32_t)((nn+31)/32),1,&pc,16);
        }
        {
            std::vector<Buf> bufs={mtp_sbuf};
            bindbufs(psoft,bufs);
            struct{int v;}pc{nn};
            dispatch(psoft,G.q0,G.cp,NH,1,1,&pc,4);
        }
        {
            std::vector<Buf> bufs={mtp_sbuf, VcM, mtp_apart};
            bindbufs(pattnvg,bufs);
            struct{int n,nh,nkv,g;}pc{nn, NH, NHKV, Grp};
            dispatch(pattnvg,G.q0,G.cp,NHKV,1,(uint32_t)nch,&pc,16);
        }
        {   // chunk sum * sigmoid(gate); the gate half sits at qn[hh*2*HD + HD + f]
            std::vector<Buf> bufs={mtp_apart, mtp_qn, mtp_att};
            bindbufs(pattncomb,bufs);
            struct{int nch,nh;}pc{nch, NH};
            dispatch(pattncomb,G.q0,G.cp,NH,1,1,&pc,8);
        }
        if(dumpvec&&pos==4){
            dumpfull("mtp_qr4",mtp_qr,NH*HD); dumpfull("mtp_kro4",mtp_kro,NHKV*HD);
            dumpfull("mtp_v4",mtp_v,NHKV*HD); dumpfull("mtp_sbuf4",mtp_sbuf,NH*(pos+1));
            dumpfull("mtp_att4",mtp_att,NH*HD); dumpfull("mtp_y4",mtp_y,D);
            dumpfull("mtp_res4",mtp_res,D); dumpfull("mtp_ffnxn4",mtp_ffnxn,D);
            dumpfull("mtp_ffg4",mtp_ffg,NFF); dumpfull("mtp_ffu4",mtp_ffu,NFF);
            dumpfull("mtp_mul4",mtp_mul,NFF); dumpfull("mtp_ffn4",mtp_ffn,D);
            dumpfull("mtp_res24",mtp_res2,D); dumpfull("mtp_hnorm4",mtp_hnorm,D);
            dumpfull("mtp_logits4",mtp_logits,VOCAB);
        }
        if(dump&&pos<10) dbgv("att",mtp_att,NH*HD);
        gemv(mtp_p+"attn_output.weight", mtp_att, mtp_y);
        mtp_addv(mtp_y, mtp_cur, mtp_res, D);   // inpSA = the eh_proj output
        if(dump&&pos<10) dbgv("attnout",mtp_y,D);
        mtp_rmsw(mtp_res, mtp_p+"post_attention_norm.weight", mtp_ffnxn, 1, D);
        gemv(mtp_p+"ffn_gate.weight", mtp_ffnxn, mtp_ffg);
        gemv(mtp_p+"ffn_up.weight",   mtp_ffnxn, mtp_ffu);
        if(dump&&pos<10){ dbgv("ffngate",mtp_ffg,NFF); dbgv("ffnup",mtp_ffu,NFF); }
        {   // fold-free SwiGLU
            std::vector<Buf> bufs={mtp_ffg, mtp_ffu, mtp_mul};
            bindbufs(psilu,bufs);
            struct{int n;}pc{NFF};
            dispatch(psilu,G.q0,G.cp,(uint32_t)((NFF+255)/256),1,1,&pc,4);
        }
        gemv(mtp_p+"ffn_down.weight", mtp_mul, mtp_ffn);
        mtp_addv(mtp_ffn, mtp_res, mtp_res2, D);
        if(dump&&pos<10){ dbgv("mul",mtp_mul,NFF); dbgv("ffnout",mtp_ffn,D); dbgv("res2",mtp_res2,D); }
        if(!want_logits) return -1;
        mtp_rmsw(mtp_res2, mtp_p+"nextn.shared_head_norm.weight", mtp_hnorm, 1, D);
        if(dump&&pos<10) dbgv("headnorm",mtp_hnorm,D);
        // output.weight is Hadamard-folded on disk (it is in prism.hadamard.weight_names),
        // so build_lora_mm folds the activation before the GEMV. blk.64.* are NOT folded,
        // which is why the rest of this block takes the plain vector.
        {
            std::vector<Buf> bufs={mtp_hnorm, sDb, mtp_hfold};
            bindbufs(pfwh,bufs);
            struct{int v;}pc{hbArg};
            dispatch(pfwh,G.q0,G.cp,foldGrid(D),1,1,&pc,4);
            fold_hi(mtp_hfold, D);
        }
        if(dumpvec&&pos==4) dumpfull("mtp_hfold4",mtp_hfold,D);
        gemv(head_name, mtp_hfold, mtp_logits);
        if(dump&&pos<10) dbgv("logits",mtp_logits,VOCAB);
        {
            std::vector<Buf> bufs={mtp_logits, mtp_aout};
            bindbufs(pargmax,bufs);
            struct{int n;}pc{VOCAB};
            dispatch(pargmax,G.q0,G.cp,1,1,1,&pc,4);
        }
        batch_flush();                       // readback of the draft token
        return ((int*)mtp_aout.mapped)[0];
    };

    // Batched Q8_0 GEMV (MTP block weights).
    auto gemvMQ8=[&](const std::string&wname, Buf&xbuf, Buf&outbuf, int T, int xstride)->bool{
        if(T < 1 || T > 4) return false;
        if(!q8names.count(wname)) return false;
        auto it = tinfo.find(wname);
        if(it == tinfo.end()) return false;
        int rows = it->second.first, cols = it->second.second;
        if(xstride<0) xstride=cols;
        std::vector<Buf> bufs={codebufs[wname],scalebufs[wname],xbuf,outbuf};
        bindbufs(pq8f,bufs);
        struct{int R,C,T,xstride;}pc{rows,cols,T,xstride};
        dispatch(pq8f,G.q0,G.cp,(uint32_t)((rows+3)/4),1,1,&pc,16);
        return true;
    };
    // Batched Q4_0 GEMV (the MTP block of the Abliterated MTP-Q4_0 model).
    auto gemvMQ4=[&](const std::string&wname, Buf&xbuf, Buf&outbuf, int T, int xstride)->bool{
        if(T < 1 || T > 4) return false;
        if(!q4names.count(wname)) return false;
        auto it = tinfo.find(wname);
        if(it == tinfo.end()) return false;
        int rows = it->second.first, cols = it->second.second;
        if(xstride<0) xstride=cols;
        std::vector<Buf> bufs={codebufs[wname],xbuf,outbuf};
        bindbufs(pq4f,bufs);
        struct{int R,C,T,xstride;}pc{rows,cols,T,xstride};
        dispatch(pq4f,G.q0,G.cp,(uint32_t)rows,1,1,&pc,16);
        return true;
    };
    // Batched native PTQ1_0 GEMV (27B). Mirrors g5ptq exactly, so T=1 matches it bit
    // for bit and row t of a T>1 launch matches a separate T=1 launch.
    auto gemvM5ptq=[&](const std::string&wname, Buf&xbuf, Buf&outbuf, int T, int xstride)->bool{
        if(T < 1 || T > 4) return false;
        if(!native_names.count(wname)) return false;
        auto it = tinfo.find(wname);
        if(it == tinfo.end()) return false;
        int rows = it->second.first, cols = it->second.second;
        if(xstride<0) xstride=cols;
        std::vector<Buf> bufs={codebufs[wname],xbuf,outbuf};
        bindbufs(pg5M,bufs);
        struct{int R,C,T,xstride;}pc{rows,cols,T,xstride};
        dispatch(pg5M,G.q0,G.cp,(uint32_t)rows,1,1,&pc,16);
        return true;
    };
    // Batched GEMV for any supported format: Q8_0, PQ2_0, native 5-trit or 4-trit.
    auto gemvT=[&](const std::string&wname, Buf&xbuf, Buf&outbuf, int T, int xstride)->bool{
        if(gemvM(wname, xbuf, outbuf, T, xstride)) return true;
        if(gemvM2pq(wname, xbuf, outbuf, T, xstride)) return true;
        if(gemvMQ8(wname, xbuf, outbuf, T, xstride)) return true;
        if(gemvM5ptq(wname, xbuf, outbuf, T, xstride)) return true;
        if(gemvMQ4(wname, xbuf, outbuf, T, xstride)) return true;
        return false;
    };

    // ---- speculative verify: state snapshot, commit replay, per-row head ----
    // Only the GDN recurrent state (M, conv) needs rollback. The KV cache is append-only
    // and position-indexed, so a rejected row is overwritten by the next pass.
    std::vector<Buf> specM, specC;
    int n_gdn=0;
    for(int il=0;il<NL;++il) if((il+1)%interval!=0) n_gdn++;
    if(spec_n>0){
        for(int i=0;i<n_gdn;++i){
            specM.push_back(mkbuf((VkDeviceSize)NV*HK*HK*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
            specC.push_back(mkbuf((VkDeviceSize)3*CONV_CH*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        }
        fprintf(stderr,"spec: state snapshot %.1f MiB (%d GDN layers)\n",
                (double)n_gdn*(NV*HK*HK*4+3.0*CONV_CH*4)/1048576.0, n_gdn);
    }
    auto spec_snapshot=[&](){
        bulk_bar_compute_to_transfer();
        int idx=0;
        for(int il=0;il<NL;++il){
            if((il+1)%interval==0) continue;
            copyB_nb(M[il], specM[idx], (size_t)NV*HK*HK*4);
            copyB_nb(convv[il], specC[idx], (size_t)3*CONV_CH*4);
            idx++;
        }
        bulk_bar_transfer_to_compute();
    };
    auto spec_restore=[&](){
        bulk_bar_compute_to_transfer();
        int idx=0;
        for(int il=0;il<NL;++il){
            if((il+1)%interval==0) continue;
            copyB_nb(specM[idx], M[il], (size_t)NV*HK*HK*4);
            copyB_nb(specC[idx], convv[il], (size_t)3*CONV_CH*4);
            idx++;
        }
        bulk_bar_transfer_to_compute();
    };
    // Commit: advance the recurrent state by the first n tokens the verify pass fed.
    // Re-running the T kernels with T = n on the restored state is bit-identical to
    // having run the pass with T = n (both are serial in token order).
    auto spec_replay=[&](int ntok){
        int idx=0;
        for(int il=0;il<NL;++il){
            if((il+1)%interval==0) continue;
            const std::string pfx="blk."+std::to_string(il)+".";
            // Restore this layer's state, then order the transfer before the compute that
            // reads it. A single barrier for the whole loop was wrong: each layer's conv
            // and delta dispatch raced the copy restoring that same layer, so the recurrent
            // state drifted after the first rejected draft.
            copyB(specM[idx], M[il], (size_t)NV*HK*HK*4);
            copyB(specC[idx], convv[il], (size_t)3*CONV_CH*4);
            bulk_bar_transfer_to_compute();
            {
                std::vector<Buf> bufs={qkvL[il], convv[il], f32bufs[pfx+"ssm_conv1d.weight"], csL[il]};
                bindbufs(pconvT,bufs);
                struct{int C,ldq,ldc,T;}pc{CONV_CH,CONV_CH,CONV_CH,ntok};
                dispatch(pconvT,G.q0,G.cp,(uint32_t)((CONV_CH+255)/256),1,1,&pc,16);
            }
            {
                std::vector<Buf> bufs={M[il], qkL[il], qkL[il], csL[il], ggL[il], gbL[il], goutT};
                std::vector<VkDeviceSize> offs={0,0,(VkDeviceSize)NK*HK*4,(VkDeviceSize)2*NK*HK*4,0,0,0};
                bindbufs(pdeltaT,bufs,offs);
                struct{float scale;int nkh,ldqk,ldv,ldg,T,ldo;}pc{1.0f/sqrtf((float)HK),NK,2*NK*HK,CONV_CH,NV,ntok,NV*HK};
                dispatch(pdeltaT,G.q0,G.cp,(uint32_t)NV,1,1,&pc,28);
            }
            idx++;
        }
    };
    // Residual add, final norm and fold for all T rows in one pass (rmsfold_T), then ONE
    // head GEMV over the T rows (the head is a GEMV like any other, so batching it reads
    // its 5 GB of weights once instead of T times), then a per-row argmax.
    auto head_rows=[&](int T){
        {
            std::vector<Buf> bufs={hT,yT,h2T,f32bufs["output_norm.weight"],sDb,xnT,xpT};
            bindbufs(prmsfoldT,bufs);
            int chunkLen=(hbArg<0?512:hbArg);
            struct{int Cc,add,HB,nchunk,stride,ostride;}pc{D,pending,hbArg,D/chunkLen,D,work};
            dispatch(prmsfoldT,G.q0,G.cp,(uint32_t)(T*pc.nchunk),1,1,&pc,24);
            for(int t=0;t<T;++t) fold_hi_off(xpT, D, t, work);
            // deliberately no hT/h2T swap: row_h() reads the split residual hT[t] + yT[t]
            // after the head, so hT and yT must keep their pre-norm contents.
        }
        if(!gemvT(head_name, xpT, logT, T, work)){
            // non-batchable head (should not happen): fall back to one row at a time
            for(int t=0;t<T;++t){
                copyB(hT, h, (size_t)D*4, (size_t)t*D*4, 0);
                copyB(yT, y, (size_t)D*4, (size_t)t*D*4, 0);
                pending=1;
                rmsfold(h, part, h2, xn, xp, "output_norm.weight", D, HB);
                gemv(head_name, xp, y);
                std::vector<Buf> bufs={y, mtp_aout};
                bindbufs(pargmax,bufs,{0,(VkDeviceSize)t*4});
                struct{int n;}pc{VOCAB};
                dispatch(pargmax,G.q0,G.cp,1,1,1,&pc,4);
            }
            return;
        }
        for(int t=0;t<T;++t){
            std::vector<Buf> bufs={logT, mtp_aout};
            bindbufs(pargmax,bufs,{(VkDeviceSize)t*VOCAB*4,(VkDeviceSize)t*4});
            struct{int n;}pc{VOCAB};
            dispatch(pargmax,G.q0,G.cp,1,1,1,&pc,4);
        }
    };
    // hidden state of row t of the T buffers: hT[t] + yT[t]
    auto row_h=[&](int t){
        copyB(hT, h, (size_t)D*4, (size_t)t*D*4, 0);
        copyB(yT, y, (size_t)D*4, (size_t)t*D*4, 0);
        mtp_addv(h, y, mtp_hprev, D);
    };

    // ---- chunked prefill over T tokens (KINDI_CHUNK) ----
    // Mirrors chunk1.py: one GEMV per weight over T tokens, GDN conv/delta serial in
    // token order inside one launch. Attention runs per token on the verified path,
    // moving the residual with in-batch copies so no flush is needed.
    auto body_T=[&](const std::vector<int>& toks, int pos, bool force=false)->bool{
        const int T=(int)toks.size();
        if((CHUNK<=0 && !force) || T<1 || T>CT) return false;
        const int CTt=T;
        // ---- embed: one launch per token straight into its hT slot.
        // Goes through embed(), which picks the decoder that matches the file format
        // (deq_emb2 PQ2_0 / deq_emb5 native PTQ1_0 / deq_emb3 expanded). The old code
        // bailed out for native PTQ1_0, which is exactly the 27B MTP model.
        for(int t=0;t<T;++t){
            embed(toks[t], &h);
            copyB(h, hT, (size_t)D*4, 0, (size_t)t*D*4);
        }
        pending=0;   // fresh chunk: drop the previous token's deferred residual
        // ---- layers ----
        int kvi=0;
        for(int il=0; il<NL; ++il){
            const std::string p="blk."+std::to_string(il)+".";
            const bool is_attn = ((il+1)%interval==0);
            if(!is_attn){
                // ---------------- GDN layer, batched ----------------
                // rmsfold_T: residual add + norm + fold for all T tokens
                {
                    std::vector<Buf> bufs={hT,yT,h2T,f32bufs[p+"attn_norm.weight"],sDb,xnT,xpT};
                    bindbufs(prmsfoldT,bufs);
                    // token stride in xpT is `work`, not D
                    int chunkLen=(hbArg<0?512:hbArg);
                    struct{int Cc,add,HB,nchunk,stride,ostride;}pc{D,pending,hbArg,D/chunkLen,D,work};
                    dispatch(prmsfoldT,G.q0,G.cp,(uint32_t)(T*pc.nchunk),1,1,&pc,24);
                    for(int t=0;t<T;++t) fold_hi_off(xpT, D, t, work);
                    if(pending>0) std::swap(hT,h2T);   // only when a residual was added
                    pending=0;
                }
                const bool cap = !qkvL.empty();   // per-layer capture for the commit replay
                Buf& Lqkv = cap ? qkvL[il] : qkvT;
                Buf& Lcs  = cap ? csL[il]  : csT;
                Buf& Lqk  = cap ? qkL[il]  : qkbufT;
                Buf& Lgg  = cap ? ggL[il]  : ggT;
                Buf& Lgb  = cap ? gbL[il]  : gbT;
                if(!gemvT(p+"attn_qkv.weight", xpT, Lqkv, T, work)) return false;
                if(!gemvT(p+"attn_gate.weight", xpT, zvT, T, work)) return false;
                // gates
                {
                    std::vector<Buf> bufs={xnT, f32bufs[p+"ssm_beta.weight"], f32bufs[p+"ssm_alpha.weight"],
                        f32bufs[p+"ssm_dt.bias"], f32bufs[p+"ssm_a"], Lgb, Lgg};
                    bindbufs(pgatesT,bufs);
                    struct{int n,T,nv;}pc{D,T,NV};
                    dispatch(pgatesT,G.q0,G.cp,(uint32_t)(T*NV),1,1,&pc,12);
                }
                // conv (serial in-kernel)
                {
                    std::vector<Buf> bufs={Lqkv, convv[il], f32bufs[p+"ssm_conv1d.weight"], Lcs};
                    bindbufs(pconvT,bufs);
                    struct{int C,ldq,ldc,T;}pc{CONV_CH,CONV_CH,CONV_CH,T};
                    dispatch(pconvT,G.q0,G.cp,(uint32_t)((CONV_CH+255)/256),1,1,&pc,16);
                }
                // l2 norm of q/k per token
                {
                    std::vector<Buf> bufs={Lcs, Lqk};
                    bindbufs(pl2wT,bufs);
                    struct{int n,rows,stride,T,ostride;}pc{HK,2*NK,CONV_CH,T,2*NK*HK};
                    dispatch(pl2wT,G.q0,G.cp,(uint32_t)((2*NK*T+7)/8),1,1,&pc,20);
                }
                // delta (serial in-kernel)
                {
                    std::vector<Buf> bufs={M[il], Lqk, Lqk, Lcs, Lgg, Lgb, goutT};
                    std::vector<VkDeviceSize> offs={0,0,(VkDeviceSize)NK*HK*4,(VkDeviceSize)2*NK*HK*4,0,0,0};
                    bindbufs(pdeltaT,bufs,offs);
                    struct{float scale;int nkh,ldqk,ldv,ldg,T,ldo;}pc{1.0f/sqrtf((float)HK),NK,2*NK*HK,CONV_CH,NV,T,NV*HK};
                    dispatch(pdeltaT,G.q0,G.cp,(uint32_t)NV,1,1,&pc,28);
                }
                // rmsw over T*NV rows (existing kernel already handles rows)
                {
                    std::vector<Buf> bufs={goutT, f32bufs[p+"ssm_norm.weight"], outT};
                    bindbufs(prmsw,bufs);
                    struct{int n,rows;}pc{HK, NV*T};
                    dispatch(prmsw,G.q0,G.cp,(uint32_t)((NV*T+3)/4),1,1,&pc,8);
                }
                // smfold_T: silu-gate + fold, with the grouped-V permute
                {
                    std::vector<Buf> bufs={zvT, outT, sVb, xpT};
                    bindbufs(psmfoldT,bufs);
                    int chunkLen=(hbArg<0?512:hbArg);
                    struct{int HB,phd,pnk,prep,nchunk,C,stride,ostride;}pc{hbArg,HK,NK,NV/NK,(NV*HK)/chunkLen,NV*HK,NV*HK,work};
                    dispatch(psmfoldT,G.q0,G.cp,(uint32_t)(T*pc.nchunk),1,1,&pc,32);
                    for(int t=0;t<T;++t) fold_hi_off(xpT, NV*HK, t, work);
                }
                if(!gemvT(p+"ssm_out.weight", xpT, yT, T, work)) return false;
                pending=1;
            } else {
                // ---------------- attention layer, per token ----------------
                // Move token t's residual into the serial buffers, run the verified path,
                // move it back. copyB is in-batch, so this costs no flush.
                for(int t=0;t<T;++t){
                    copyB(hT, h, (size_t)D*4, (size_t)t*D*4, 0);
                    copyB(yT, y, (size_t)D*4, (size_t)t*D*4, 0);
                    pending = 1;
                    attn(il, pos+t, kvi);
                    copyB(h, hT, (size_t)D*4, 0, (size_t)t*D*4);
                    copyB(y, yT, (size_t)D*4, 0, (size_t)t*D*4);
                }
                kvi++;
                pending = 1;
                // FFN falls through to the batched block below. It used to run per token
                // here, which made the 16 attention layers of the 27B pay single-token
                // GEMVs for gate/up/down while the 48 GDN layers batched theirs.
            }
            // ---------------- FFN, batched (GDN layers) ----------------
            {
                std::vector<Buf> bufs={hT,yT,h2T,f32bufs[p+"post_attention_norm.weight"],sDb,xnT,xpT};
                bindbufs(prmsfoldT,bufs);
                int chunkLen=(hbArg<0?512:hbArg);
                struct{int Cc,add,HB,nchunk,stride,ostride;}pc{D,pending,hbArg,D/chunkLen,D,work};
                dispatch(prmsfoldT,G.q0,G.cp,(uint32_t)(T*pc.nchunk),1,1,&pc,24);
                for(int t=0;t<T;++t) fold_hi_off(xpT, D, t, work);
                if(pending>0) std::swap(hT,h2T);
                pending=0;
            }
            if(!gemvT(p+"ffn_gate.weight", xpT, ffgT, T, work)) return false;
            if(!gemvT(p+"ffn_up.weight", xpT, ffuT, T, work)) return false;
            {
                std::vector<Buf> bufs={ffgT, ffuT, sFb, fo6T};
                bindbufs(psmfoldT,bufs);
                int chunkLen=(hbArg<0?512:hbArg);
                struct{int HB,phd,pnk,prep,nchunk,C,stride,ostride;}pc{hbArg,0,0,0,NFF/chunkLen,NFF,NFF,NFF};
                dispatch(psmfoldT,G.q0,G.cp,(uint32_t)(T*pc.nchunk),1,1,&pc,32);
                for(int t=0;t<T;++t) fold_hi_off(fo6T, NFF, t, NFF);
            }
            if(!gemvT(p+"ffn_down.weight", fo6T, yT, T, NFF)) return false;
            pending=1;
        }
        return true;
    };

    // ---- body ----
    auto body=[&](int tok, int pos){
        embed(tok);
        pending=0;   // fresh token: drop the previous token's deferred FFN residual (engine.py body())
        int kvi=0;
        for(int il=0;il<NL;++il){
            if((il+1)%interval!=0){ gdn(il,pos); }
            else { attn(il,pos,kvi); kvi++; }
            ffn(il);
            if(dump&&pos==0){
                std::vector<float> hv=readbuf(h,3), yv=readbuf(y,3);
                fprintf(stderr,"G L%02d res %.4f %.4f %.4f\n",il,hv[0]+yv[0],hv[1]+yv[1],hv[2]+yv[2]);
            }
        }
    };

    // ---- step ----
    auto step=[&](int tok, int pos)->int{
        body(tok,pos);
        rmsfold(h, part, h2, xn, xp, "output_norm.weight", D, HB);
        gemv(head_name, xp, y);
        if(temperature <= 0.0f){
            std::vector<Buf> bufs={y,aout};
            bindbufs(pargmax,bufs);
            struct{int n;}pc{VOCAB};
            dispatch(pargmax,G.q0,G.cp,1,1,1,&pc,4);
            batch_flush();   // readback of the argmax slot
            int* res=(int*)aout.mapped;
            return res[0];
        }
        // sampling path: read back logits
        copybuf(y, logitsbuf, (size_t)VOCAB*4);
        float* L=(float*)logitsbuf.mapped;
        // temperature scaling
        for(int i=0;i<VOCAB;++i) L[i]/=temperature;
        // top-k filtering
        if(topk>0 && topk<VOCAB){
            // find k-th largest
            std::vector<float> tmp(L, L+VOCAB);
            std::nth_element(tmp.begin(), tmp.begin()+(VOCAB-topk), tmp.end());
            float thr=tmp[VOCAB-topk];
            for(int i=0;i<VOCAB;++i) if(L[i]<thr) L[i]=-1e30f;
        }
        // softmax
        float mx=-1e30f;
        for(int i=0;i<VOCAB;++i) if(L[i]>mx) mx=L[i];
        double sum=0;
        for(int i=0;i<VOCAB;++i){ L[i]=expf(L[i]-mx); sum+=L[i]; }
        for(int i=0;i<VOCAB;++i) L[i]/=(float)sum;
        // sample
        double r=(double)rand()/RAND_MAX;
        double acc=0;
        for(int i=0;i<VOCAB;++i){ acc+=L[i]; if(r<acc) return i; }
        return VOCAB-1;
    };

    // ---- generate ----
    if(selftest){
        bool ok = true;
        auto rd=[&](Buf&b,int n){ Buf stg=mkbuf((VkDeviceSize)n*4,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
              copybuf(b,stg,(size_t)n*4); std::vector<float> v((float*)stg.mapped,(float*)stg.mapped+n);
              freebuf(stg); return v; };
        auto mkx=[&](int C, Buf&out){ std::vector<float> v((size_t)C*2);
              for(int i=0;i<C;++i){ float x=std::sin(0.01f*(float)i)*0.5f; v[i]=x; v[C+i]=x; }
              out=mkbuf((VkDeviceSize)v.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
              upload(&out, v.data(), v.size()*4); };

        // ---- 4-trit batched GEMV (g4lutM) ----
        std::string wn;
        for(auto&kv:codebufs){ if(scalebufs.count(kv.first) && !pq2_names.count(kv.first)
                                 && !native_names.count(kv.first) && tinfo.count(kv.first)
                                 && kv.first.find("ssm_out")!=std::string::npos){ wn=kv.first; break; } }
        if(wn.empty()){
            fprintf(stderr,"selftest 4-trit: skipped (no 4-trit tensor)\n");
        } else {
            int R=tinfo[wn].first, C=tinfo[wn].second;
            fprintf(stderr,"selftest 4-trit on %s  R=%d C=%d\n", wn.c_str(), R, C);
            Buf xb; mkx(C, xb);
            Buf y1=mkbuf((VkDeviceSize)R*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Buf y2=mkbuf((VkDeviceSize)R*2*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            { std::vector<Buf> b={codebufs[wn],scalebufs[wn],y1,xb}; bindbufs(pg4M,b);
              struct{int R,C,T,xstride;}pc{R,C,1,C}; dispatch(pg4M,G.q0,G.cp,(uint32_t)((R+3)/4),1,1,&pc,16); }
            batch_flush();
            { std::vector<Buf> b={codebufs[wn],scalebufs[wn],y2,xb}; bindbufs(pg4M,b);
              struct{int R,C,T,xstride;}pc{R,C,2,C}; dispatch(pg4M,G.q0,G.cp,(uint32_t)((R+3)/4),1,1,&pc,16); }
            batch_flush();
            auto a=rd(y1,R), bb=rd(y2,R*2);
            double e0=0,e1=0,nrm=0;
            for(int i=0;i<R;++i){
                double d0=(double)std::fabs(a[i]-bb[i]);       if(d0>e0) e0=d0;
                double d1=(double)std::fabs(a[i]-bb[R+i]);     if(d1>e1) e1=d1;
                nrm += std::fabs(a[i]);
            }
            fprintf(stderr,"selftest 4-trit T=2: max|y1-y2[0]|=%.3e  max|y1-y2[1]|=%.3e  sum=%.3f\n", e0,e1,nrm);
            bool o1 = (e0 < 1e-4) && (e1 < 1e-4);
            fprintf(stderr,"selftest 4-trit: %s\n", o1 ? "PASS" : "FAIL");
            ok = ok && o1;
        }

        // ---- PQ2_0 batched GEMV (g2pqM) ----
        std::string pn;
        for(auto&kv:codebufs){ if(pq2_names.count(kv.first) && tinfo.count(kv.first)
                                 && kv.first.find("ssm_out")!=std::string::npos){ pn=kv.first; break; } }
        if(pn.empty()){
            fprintf(stderr,"selftest PQ2_0: skipped (no PQ2_0 tensor)\n");
        } else {
            int R=tinfo[pn].first, C=tinfo[pn].second;
            fprintf(stderr,"selftest PQ2_0 on %s  R=%d C=%d\n", pn.c_str(), R, C);
            Buf xb; mkx(C, xb);
            Buf y1=mkbuf((VkDeviceSize)R*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Buf y2=mkbuf((VkDeviceSize)R*2*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            // reference: the shipped single-token g2pq
            { std::vector<Buf> b={codebufs[pn],xb,y1}; bindbufs(pg2,b);
              struct{int R,C;}pc{R,C}; dispatch(pg2,G.q0,G.cp,(uint32_t)R,1,1,&pc,8); }
            batch_flush();
            { std::vector<Buf> b={codebufs[pn],xb,y2}; bindbufs(pg2M,b);
              struct{int R,C,T,xstride;}pc{R,C,2,C}; dispatch(pg2M,G.q0,G.cp,(uint32_t)((R+3)/4),1,1,&pc,16); }
            batch_flush();
            auto a=rd(y1,R), bb=rd(y2,R*2);
            double e0=0,e1=0,nrm=0;
            for(int i=0;i<R;++i){
                double d0=(double)std::fabs(a[i]-bb[i]);   if(d0>e0) e0=d0;
                double d1=(double)std::fabs(a[i]-bb[R+i]); if(d1>e1) e1=d1;
                nrm += std::fabs(a[i]);
            }
            fprintf(stderr,"selftest PQ2_0 T=2: max|y1-y2[0]|=%.3e  max|y1-y2[1]|=%.3e  sum=%.3f\n", e0,e1,nrm);
            bool o2 = (e0 < 1e-4) && (e1 < 1e-4);
            fprintf(stderr,"selftest PQ2_0: %s\n", o2 ? "PASS" : "FAIL");
            ok = ok && o2;
        }

        // ---- Q8_0 GEMV (MTP block weights) ----
        std::string qn;
        for(auto&kv:codebufs){ if(q8names.count(kv.first) && tinfo.count(kv.first)
                                 && kv.first.find("attn_q.weight")!=std::string::npos){ qn=kv.first; break; } }
        if(qn.empty()){
            fprintf(stderr,"selftest Q8_0: skipped (no Q8_0 tensor)\n");
        } else {
            int R=tinfo[qn].first, C=tinfo[qn].second;
            fprintf(stderr,"selftest Q8_0 on %s  R=%d C=%d\n", qn.c_str(), R, C);
            Buf xb; mkx(C, xb);
            Buf y1=mkbuf((VkDeviceSize)R*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            { std::vector<Buf> b={codebufs[qn],scalebufs[qn],xb,y1}; bindbufs(pq8f,b);
              struct{int R,C,T,xstride;}pc{R,C,1,C}; dispatch(pq8f,G.q0,G.cp,(uint32_t)((R+3)/4),1,1,&pc,16); }
            batch_flush();
            auto gv=rd(y1,R);
            // host reference straight from the GGUF: block_q8_0 is { fp16 d; int8 qs[32]; }.
            // This also checks the loader's block layout, which the old version (reading the
            // device code/scale buffers back) could not: it agreed with the loader by
            // construction and so hid a two-byte shift.
            const GT* gq=nullptr;
            for(auto&g:gtensors) if(g.name==qn) gq=&g;
            if(!gq){ fprintf(stderr,"selftest Q8_0: no GGUF tensor entry for %s\n",qn.c_str()); return 1; }
            std::vector<float> xv((size_t)C);
            for(int i=0;i<C;++i) xv[i]=std::sin(0.01f*(float)i)*0.5f;
            const int bpr=C/32;
            double worst=0; int nz=0;
            for(int r=0;r<R;r+=97){          // sample rows
                const unsigned char* rp=filebase + data_start + gq->offset + (size_t)r*bpr*34;
                double h=0;
                for(int k=0;k<C;++k){
                    const unsigned char* blk=rp + (size_t)(k>>5)*34;
                    uint16_t hb; memcpy(&hb, blk, 2);
                    float d=half_to_float(hb);
                    int q=(int)(int8_t)blk[2+(k&31)];
                    h += (double)(d*(float)q)*xv[k];
                }
                double e=std::fabs(h-gv[r]); if(e>worst) worst=e; nz++;
            }
            fprintf(stderr,"selftest Q8_0: sampled %d rows, max abs err vs host = %.4e\n", nz, worst);
            bool o3 = worst < 5e-3;
            fprintf(stderr,"selftest Q8_0: %s\n", o3 ? "PASS" : "FAIL");
            ok = ok && o3;
        }

        // ---- Q4_0 GEMV (the MTP block of the MTP-Q4_0 model) ----
        std::string q4n;
        for(auto&kv:codebufs){ if(q4names.count(kv.first) && tinfo.count(kv.first)){ q4n=kv.first; break; } }
        if(q4n.empty()){
            fprintf(stderr,"selftest Q4_0: skipped (no Q4_0 tensor)\n");
        } else {
            int R=tinfo[q4n].first, C=tinfo[q4n].second;
            fprintf(stderr,"selftest Q4_0 on %s  R=%d C=%d\n", q4n.c_str(), R, C);
            Buf xb; mkx(C, xb);
            Buf y1=mkbuf((VkDeviceSize)R*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Buf y2=mkbuf((VkDeviceSize)R*2*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            { std::vector<Buf> b={codebufs[q4n],xb,y1}; bindbufs(pq4f,b);
              struct{int R,C,T,xstride;}pc{R,C,1,C}; dispatch(pq4f,G.q0,G.cp,(uint32_t)R,1,1,&pc,16); }
            batch_flush();
            { std::vector<Buf> b={codebufs[q4n],xb,y2}; bindbufs(pq4f,b);
              struct{int R,C,T,xstride;}pc{R,C,2,C}; dispatch(pq4f,G.q0,G.cp,(uint32_t)R,1,1,&pc,16); }
            batch_flush();
            auto gv=rd(y1,R), gv2=rd(y2,R*2);
            // host reference straight from the GGUF: block_q4_0 = { fp16 d; uint8 qs[16] }
            const GT* g4=nullptr;
            for(auto&g:gtensors) if(g.name==q4n) g4=&g;
            std::vector<float> xv((size_t)C);
            for(int i=0;i<C;++i) xv[i]=std::sin(0.01f*(float)i)*0.5f;
            const int bpr=C/32;
            double worst=0, worst2=0; int nz=0;
            for(int r=0;r<R;r+=97){
                const unsigned char* rp=filebase + data_start + g4->offset + (size_t)r*bpr*18;
                double h=0;
                for(int k=0;k<C;++k){
                    const unsigned char* blk=rp + (size_t)(k>>5)*18;
                    uint16_t hb; memcpy(&hb, blk, 2);
                    float d=half_to_float(hb);
                    uint8_t byte=blk[2+(k&15)];
                    int nib=(k&16)?(byte>>4):(byte&0xF);
                    h += (double)((float)(nib-8)*d)*xv[k];
                }
                double e=std::fabs(h-gv[r]); if(e>worst) worst=e;
                double e2=std::fabs(h-gv2[r]); if(e2>worst2) worst2=e2;
                nz++;
            }
            fprintf(stderr,"selftest Q4_0: sampled %d rows, max abs err vs host = %.4e (T=2 row0 %.4e)\n", nz, worst, worst2);
            bool o6 = worst < 5e-3 && worst2 < 5e-3;
            fprintf(stderr,"selftest Q4_0: %s\n", o6 ? "PASS" : "FAIL");
            ok = ok && o6;
            freebuf(y1); freebuf(y2);
        }

        // ---- native PTQ1_0 batched GEMV (g5ptqM): T=1 must equal g5ptq exactly ----
        std::string n5;
        for(auto&kv:codebufs){ if(native_names.count(kv.first) && tinfo.count(kv.first)
                                 && kv.first.find("ssm_out")!=std::string::npos){ n5=kv.first; break; } }
        if(n5.empty()){
            fprintf(stderr,"selftest PTQ1_0: skipped (no native tensor)\n");
        } else {
            int R=tinfo[n5].first, C=tinfo[n5].second;
            fprintf(stderr,"selftest PTQ1_0 on %s  R=%d C=%d\n", n5.c_str(), R, C);
            Buf xb; mkx(C, xb);
            Buf y1=mkbuf((VkDeviceSize)R*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Buf y2=mkbuf((VkDeviceSize)R*2*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            { std::vector<Buf> b={codebufs[n5],xb,y1}; bindbufs(pg5,b);
              struct{int R,C;}pc{R,C}; dispatch(pg5,G.q0,G.cp,(uint32_t)R,1,1,&pc,8); }
            batch_flush();
            { std::vector<Buf> b={codebufs[n5],xb,y2}; bindbufs(pg5M,b);
              struct{int R,C,T,xstride;}pc{R,C,2,C}; dispatch(pg5M,G.q0,G.cp,(uint32_t)R,1,1,&pc,16); }
            batch_flush();
            auto a=rd(y1,R), bb=rd(y2,R*2);
            double e0=0,e1=0;
            for(int i=0;i<R;++i){
                double d0=std::fabs(a[i]-bb[i]);   if(d0>e0) e0=d0;
                double d1=std::fabs(a[i]-bb[R+i]); if(d1>e1) e1=d1;
            }
            // Informational only: the per-trit shader is kept for A/B against the
            // old decode (KINDI_G5=old). It no longer shares arithmetic with the
            // batched kernel, so this difference is expected, not a regression.
            fprintf(stderr,"selftest PTQ1_0 per-trit vs batched: max|dy|=%.3e (informational)\n",
                    e0 > e1 ? e0 : e1);

            // The byte-granular pair is what the engine actually runs now. The spec
            // verify pass compares trunk rows produced by g5ptqM against tokens the
            // serial path produced with g5ptq_byte, so these two must agree exactly
            // or an accepted draft could differ from the plain run.
            {
                Buf z1=mkbuf((VkDeviceSize)R*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                Buf z2=mkbuf((VkDeviceSize)R*2*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                { std::vector<Buf> b={codebufs[n5],xb,z1}; bindbufs(pg5b,b);
                  struct{int R,C;}pc{R,C}; dispatch(pg5b,G.q0,G.cp,(uint32_t)R,1,1,&pc,8); }
                batch_flush();
                { std::vector<Buf> b={codebufs[n5],xb,z2}; bindbufs(pg5M,b);
                  struct{int R,C,T,xstride;}pc{R,C,2,C}; dispatch(pg5M,G.q0,G.cp,(uint32_t)R,1,1,&pc,16); }
                batch_flush();
                auto c1=rd(z1,R), c2=rd(z2,R*2);
                double f0=0,f1=0;
                for(int i=0;i<R;++i){
                    double d0=std::fabs(c1[i]-c2[i]);   if(d0>f0) f0=d0;
                    double d1=std::fabs(c1[i]-c2[R+i]); if(d1>f1) f1=d1;
                }
                fprintf(stderr,"selftest PTQ1_0 byte T=2: max|y1-y2[0]|=%.3e  max|y1-y2[1]|=%.3e\n", f0,f1);
                bool o5b = (f0 == 0.0) && (f1 == 0.0);
                fprintf(stderr,"selftest PTQ1_0 byte: %s\n", o5b ? "PASS (bit-identical)" : "FAIL");
                ok = ok && o5b;
                freebuf(z1); freebuf(z2);
            }
            freebuf(y1); freebuf(y2);
        }

        // ---- fold-free SwiGLU (silu_mul): the MTP block's activation ----
        {
            int n = (NFF < 4096) ? NFF : 4096;
            std::vector<float> gv((size_t)n), uv((size_t)n);
            for(int i=0;i<n;++i){ gv[i]=std::sin(0.013f*(float)i)*3.0f; uv[i]=std::cos(0.007f*(float)i)*2.0f; }
            Buf gb2=mkbuf((VkDeviceSize)n*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Buf ub2=mkbuf((VkDeviceSize)n*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Buf ob2=mkbuf((VkDeviceSize)n*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            upload(&gb2,gv.data(),(size_t)n*4); upload(&ub2,uv.data(),(size_t)n*4);
            { std::vector<Buf> b={gb2,ub2,ob2}; bindbufs(psilu,b);
              struct{int n;}pc{n}; dispatch(psilu,G.q0,G.cp,(uint32_t)((n+255)/256),1,1,&pc,4); }
            batch_flush();
            auto ov=rd(ob2,n);
            double worst=0;
            for(int i=0;i<n;++i){
                double ref=(double)(gv[i]/(1.0+std::exp(-gv[i])))*uv[i];
                double e=std::fabs(ref-ov[i]); if(e>worst) worst=e;
            }
            fprintf(stderr,"selftest silu_mul: n=%d max abs err vs host = %.3e\n", n, worst);
            bool o4 = worst < 1e-5;
            fprintf(stderr,"selftest silu_mul: %s\n", o4 ? "PASS" : "FAIL");
            ok = ok && o4;
            freebuf(gb2); freebuf(ub2); freebuf(ob2);
        }

        fprintf(stderr,"selftest: %s\n", ok ? "PASS - batched GEMV matches single-token" : "FAIL");
        return ok ? 0 : 1;
    }

    // ---- tokenizer ----
    kindi_tok::BpeTokenizer tok;
    std::string tokerr;
    if(!tok.load_from_strings(gguf_tokens, gguf_merges, &tokerr)){ fprintf(stderr,"tokenizer load failed: %s\n", tokerr.c_str()); return 1; }
    std::string promptstr = prompt ? prompt : "The sky is blue because";
    std::vector<int> ids = tok.encode(promptstr);
    if(ids.empty()) ids.push_back(0);
    fprintf(stderr,"prompt ids (%zu):", ids.size());
    for(int t: ids) fprintf(stderr," %d", t);
    fprintf(stderr,"\n");
    // ================================ OpenAI-compatible API ================================
    if(serve){
        const int n_gdn = n_gdn_est;
        // ---- reset to an empty sequence. The KV caches are position-indexed and only ever
        // read below the current position, so they need no clearing; the recurrent state
        // (M, conv) is genuine state and must be zeroed.
        auto reset_state=[&](){
            for(int il=0;il<NL;++il){ fillbuf(M[il]); fillbuf(convv[il]); }
            fillbuf(KcM); fillbuf(VcM);
            // h_{-1} for the MTP row 0 is a zero vector, as the reference starts pending_h
            if(have_mtp) fillbuf(mtp_hin);
            pending=0;
        };
        // ---- prefix cache: one checkpoint of the recurrent state, taken after a prompt
        // prefill. KV needs no copy at all, so this is the whole cost of prompt reuse.
        struct Pc { std::vector<int> ids; bool valid=false; std::vector<Buf> M, C; bool ok=false; };
        Pc pc;
        auto pc_alloc=[&]()->bool{
            if(pc.ok) return true;
            pc.M.resize(n_gdn); pc.C.resize(n_gdn);
            for(int i=0;i<n_gdn;++i){
                if(!mkbuf_try((VkDeviceSize)NV*HK*HK*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,pc.M[i]) ||
                   !mkbuf_try((VkDeviceSize)3*CONV_CH*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,pc.C[i])){
                    for(auto&b:pc.M) if(b.buf) freebuf(b);
                    for(auto&b:pc.C) if(b.buf) freebuf(b);
                    pc.M.clear(); pc.C.clear();
                    return false;
                }
            }
            pc.ok=true;
            return true;
        };
        auto pc_save=[&](const std::vector<int>& ids){
            if(!prefix_cache) return;
            if(!pc_alloc()) return;
            bulk_bar_compute_to_transfer();
            int i=0;
            for(int il=0;il<NL;++il){
                if((il+1)%interval==0) continue;
                copyB_nb(M[il], pc.M[i], (size_t)NV*HK*HK*4);
                copyB_nb(convv[il], pc.C[i], (size_t)3*CONV_CH*4);
                i++;
            }
            bulk_bar_transfer_to_compute();
            pc.ids=ids; pc.valid=true;
        };
        auto pc_restore=[&](int ntok)->bool{
            if(!prefix_cache || !pc.valid || ntok<=0) return false;
            bulk_bar_compute_to_transfer();
            int i=0;
            for(int il=0;il<NL;++il){
                if((il+1)%interval==0) continue;
                copyB_nb(pc.M[i], M[il], (size_t)NV*HK*HK*4);
                copyB_nb(pc.C[i], convv[il], (size_t)3*CONV_CH*4);
                i++;
            }
            bulk_bar_transfer_to_compute();
            return true;
        };

        // Speculative decode needs the MTP head, greedy sampling and T=2 buffers, all of
        // which --spec already arranges; in serve mode it means "use it", not a token count.
        const bool spec_ok = (spec_n>0 && have_mtp);
        if(spec_n>0 && !have_mtp)
            fprintf(stderr,"api: --spec ignored, this model has no MTP block\n");

        int im_end_id=-1;
        { auto v=tok.encode("<|im_end|>"); if(v.size()==1) im_end_id=v[0]; }
        if(im_end_id<0) fprintf(stderr,"api: <|im_end|> is not a single token; only EOS stops generation\n");

        // ---- generate. on_piece gets the decoded text of each step.
        struct GStats { int prompt=0, completion=0; double prefill_ms=0, decode_ms=0; int accepted=0, passes=0; };
        auto generate=[&](const std::vector<int>& in_ids, int n_predict, float temp, int tk,
                          const std::function<void(int,const std::string&)>& on_piece)->GStats{
            GStats st;
            temperature=temp; topk=tk;
            // Speculative decode is greedy-only: the draft is one argmax, and accepting it
            // under sampling would change the distribution.
            const bool use_spec = spec_ok && (temp<=0.0f);
            std::vector<int> ids=in_ids;
            if(ids.empty()) ids.push_back(0);
            int room=ctx-2;
            if((int)ids.size()>room) ids.erase(ids.begin(), ids.end()-room);
            st.prompt=(int)ids.size();

            double t0=ktrace::now_us();
            // prefix reuse: resume after the longest common prefix
            int shared=0;
            if(prefix_cache && pc.valid){
                size_t m=(ids.size()<pc.ids.size())?ids.size():pc.ids.size();
                while((size_t)shared<m && ids[shared]==pc.ids[shared]) shared++;
                // Keep at least one token to feed. With speculative decode keep two:
                // the draft at the last position needs h_{N-2}, which is a residual-stream
                // value the checkpoint does not store, so position N-2 must be fed.
                int keep = use_spec ? 2 : 1;
                int cap  = (int)ids.size()-keep;
                if(cap<0) cap=0;
                if(shared>cap) shared=cap;
                if(shared>0 && !pc_restore(shared)) shared=0;
            }
            if(shared>0) fprintf(stderr,"api: prefix reuse %d/%zu tokens\n", shared, ids.size());
            else { reset_state(); shared=0; }

            // prefill ids[shared .. n-2]
            size_t last=ids.size()-1;
            if(use_spec){
                // The MTP layer keeps its own KV and its attention needs a row at every
                // position. Row q is written by the block at q, which consumes (h_{q-1}, x_q),
                // so the catch-up runs alongside the trunk prefill. Row 0 pairs with a zero
                // h, matching the reference's pending_h initialisation.
                //
                // The trunk prefill stays chunked: the catch-up reads the chunk's per-token
                // hidden states out of hT/yT, so it does not force a serial pass. Running
                // the catch-up per token with body() instead cost 9.4 s on a 17-token prompt.
                if(shared==0 && ids.size()>=2){
                    ktrace::set_tag("mtp:");
                    mtp_block(mtp_hin, ids[0], 0, false);
                    ktrace::set_tag("");
                }
                size_t i=(size_t)shared;
                while(i<last){
                    size_t remain=last-i;
                    int T=(CHUNK>0 && remain>(size_t)1)
                          ? (int)((remain<(size_t)CHUNK)?remain:(size_t)CHUNK) : 1;
                    std::vector<int> ch(ids.begin()+i, ids.begin()+i+T);
                    bool chunked = (T>1) && body_T(ch,(int)i);
                    if(!chunked) for(int t=0;t<T;++t) body(ch[t],(int)(i+t));
                    for(int t=0;t<T;++t){
                        size_t g=i+(size_t)t;
                        if(chunked){
                            // h_g = residual + deferred FFN output
                            copyB(hT, h, (size_t)D*4, (size_t)t*D*4, 0);
                            copyB(yT, y, (size_t)D*4, (size_t)t*D*4, 0);
                        }
                        mtp_addv(h, y, mtp_hprev, D);          // h_g
                        if(g+1<last){
                            mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
                            ktrace::set_tag("mtp:");
                            mtp_block(mtp_hin, ids[g+1], (int)g+1, false);   // MTP KV row g+1
                            ktrace::set_tag("");
                        }
                    }
                    i+=T;
                }
                // mtp_hprev now holds h_{N-2}, which is the draft's h_input for position N-1
                mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
            } else if(CHUNK>0 && !shared){
                size_t i=0;
                while(i<last){
                    size_t remain=last-i;
                    int T=(int)((remain<(size_t)CHUNK)?remain:(size_t)CHUNK);
                    std::vector<int> ch(ids.begin()+i, ids.begin()+i+T);
                    if(!body_T(ch,(int)i)) for(int t=0;t<T;++t) body(ch[t],(int)i+t);
                    i+=T;
                }
            } else {
                for(size_t i=(size_t)shared;i<last;++i) body(ids[i],(int)i);
            }
            st.prefill_ms=(ktrace::now_us()-t0)/1000.0;

            // Advance the checkpoint whenever the prompt extended it, so a multi-turn
            // conversation reuses more each turn instead of only the first prompt.
            bool did_work = (int)ids.size()-1 > shared;
            if(prefix_cache && (int)ids.size()>=64 && (did_work || !pc.valid)) pc_save(ids);

            ktrace::mark_decode_start();   // --trace-decode: table starts here
            double t1=ktrace::now_us();
            int pos=(int)ids.size()-1;
            int cur=ids.back();          // `tok` is the tokenizer, so the token is `cur`
            std::string acc;          // bytes not yet emitted (UTF-8 boundaries)
            std::vector<int> gen;

            // Push one token: false means "stop generating". Emits only complete UTF-8
            // sequences, so a multi-byte character split across two tokens never reaches
            // the client as mojibake.
            auto emit_tok=[&](int t)->bool{
                if(t==eos_id) return false;
                if(t==im_end_id) return false;
                gen.push_back(t);
                acc+=tok.decode({t});
                size_t cut=acc.size();
                for(size_t i=0;i<acc.size();){
                    unsigned char c=(unsigned char)acc[i];
                    size_t need = (c<0x80)?1:((c>>5)==0x6)?2:((c>>4)==0xE)?3:((c>>3)==0x1E)?4:1;
                    if(i+need>acc.size()){ cut=i; break; }
                    i+=need;
                }
                if(cut>0){ on_piece(t, acc.substr(0,cut)); acc.erase(0,cut); }
                return true;
            };

            if(use_spec){
                // One draft, one T=2 verify pass, accept the matching prefix. Same loop as
                // the CLI's --spec (see MTP-FINDINGS.md); the trunk path is unchanged, so a
                // rejected draft falls back to exactly the serial token.
                int d=0;
                { ktrace::set_tag("mtp:");
                  d=mtp_block(mtp_hin, cur, pos, true);
                  ktrace::set_tag(""); }
                bool ok=true;
                while((int)gen.size()<n_predict && ok){
                    if(pos+2>=ctx) break;
                    std::vector<int> fed{cur, d};
                    spec_snapshot();
                    ktrace::set_tag("spec:");
                    bool okT=body_T(fed, pos, true);
                    ktrace::set_tag("");
                    if(!okT){ ok=false; break; }      // fall back to serial below
                    head_rows(2);
                    batch_flush();                    // one readback for both rows
                    int r0=((int*)mtp_aout.mapped)[0];
                    int r1=((int*)mtp_aout.mapped)[1];
                    int j=(r0==d)?1:0;
                    int ntok=1+j;
                    st.passes++; st.accepted+=j;
                    if(ntok<2){
                        spec_restore();
                        ktrace::set_tag("spec:");
                        spec_replay(ntok);
                        ktrace::set_tag("");
                    }
                    if(!emit_tok(r0)) break;
                    if(j && (int)gen.size()<n_predict){
                        if(!emit_tok(r1)) break;
                    }
                    cur = j ? r1 : r0;
                    pos += ntok;
                    if((int)gen.size()>=n_predict) break;
                    // MTP catch-up for the committed position, then the next draft
                    if(j){
                        row_h(0);
                        mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
                        ktrace::set_tag("mtp:");
                        mtp_block(mtp_hin, r0, pos-1, false);
                        ktrace::set_tag("");
                        row_h(1);
                    } else {
                        row_h(0);
                    }
                    mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
                    ktrace::set_tag("mtp:");
                    d=mtp_block(mtp_hin, cur, pos, true);
                    ktrace::set_tag("");
                }
                if(!ok){
                    // body_T declined mid-request: finish serially from the committed state
                    fprintf(stderr,"api: body_T declined, finishing this request serially\n");
                    while((int)gen.size()<n_predict){
                        if(pos>=ctx) break;
                        int next=step(cur,pos);
                        pos++;
                        if(!emit_tok(next)) break;
                        cur=next;
                    }
                }
            } else {
                for(int k=0;k<n_predict;++k){
                    if(pos>=ctx) break;
                    int next=step(cur,pos);
                    pos++;
                    if(!emit_tok(next)) break;
                    cur=next;
                }
            }
            if(!acc.empty()) on_piece(-1, acc);
            st.completion=(int)gen.size();
            st.decode_ms=(ktrace::now_us()-t1)/1000.0;
            return st;
        };

        // ---- ChatML, text-only, matching tokenizer.chat_template
        auto render_chat=[&](const json& msgs, bool add_gen, bool thinking)->std::string{
            std::string out;
            bool first=true;
            for(auto& m: msgs){
                std::string role=m.value("role","user");
                std::string content;
                if(m.contains("content") && !m["content"].is_null()){
                    if(m["content"].is_string()) content=m["content"].get<std::string>();
                    else if(m["content"].is_array()){
                        for(auto& part: m["content"]){
                            if(part.is_object() && part.value("type",std::string())=="text")
                                content+=part.value("text",std::string());
                        }
                    }
                }
                if(role=="system"){
                    if(!first) continue;                 // the template requires system first
                    out+="<|im_start|>system\n"+content+"<|im_end|>\n";
                } else if(role=="user"){
                    out+="<|im_start|>user\n"+content+"<|im_end|>\n";
                } else if(role=="assistant"){
                    out+="<|im_start|>assistant\n"+content+"<|im_end|>\n";
                } else if(role=="tool"){
                    out+="<|im_start|>user\n<tool_response>\n"+content+"\n</tool_response><|im_end|>\n";
                }
                first=false;
            }
            if(add_gen){
                out+="<|im_start|>assistant\n";
                if(!thinking) out+="<think>\n\n</think>\n\n";
                else          out+="<think>\n";
            }
            return out;
        };

        // ---- server loop
        SOCKET ls=kindi_http::listen_on(srv_host, srv_port);
        if(ls==INVALID_SOCKET){ fprintf(stderr,"api: cannot listen on %s:%d\n",srv_host,srv_port); return 1; }
        std::string model_id=model;
        { size_t a=model_id.find_last_of("/\\"); if(a!=std::string::npos) model_id=model_id.substr(a+1);
          size_t d=model_id.rfind(".gguf"); if(d!=std::string::npos) model_id=model_id.substr(0,d); }
        fprintf(stderr,"api: listening on http://%s:%d  (model %s, ctx %d, %s)\n",
                srv_host, srv_port, model_id.c_str(), ctx,
                spec_ok?"speculative k=1 + chunked prefill":(CHUNK>0?"chunked prefill":"serial"));
        fprintf(stderr,"api: endpoints /v1/models /v1/chat/completions /v1/completions /health\n");
        if(spec_ok)
            fprintf(stderr,"api: speculative decode ON (k=1; greedy requests only, sampling runs serially)\n");

        for(;;){
            SOCKET cs=accept(ls,nullptr,nullptr);
            if(cs==INVALID_SOCKET) continue;
            kindi_http::Req rq;
            if(!kindi_http::read_req(cs,rq)){ closesocket(cs); continue; }
            const std::string& path=rq.path;

            if(path=="/health"){
                kindi_http::respond_json(cs,200,"{\"status\":\"ok\"}");
                closesocket(cs); continue;
            }
            if(path=="/v1/models"){
                std::string b="{\"object\":\"list\",\"data\":[{\"id\":\""+kindi_http::json_escape(model_id)
                             +"\",\"object\":\"model\",\"owned_by\":\"kindi\"}]}";
                kindi_http::respond_json(cs,200,b);
                closesocket(cs); continue;
            }
            if(path!="/v1/chat/completions" && path!="/v1/completions"){
                kindi_http::respond_err(cs,404,"unknown path "+path);
                closesocket(cs); continue;
            }
            if(rq.method!="POST"){
                kindi_http::respond_err(cs,405,"POST required");
                closesocket(cs); continue;
            }

            json j;
            try { j=json::parse(rq.body); }
            catch(const std::exception& e){
                kindi_http::respond_err(cs,400,std::string("invalid JSON: ")+e.what());
                closesocket(cs); continue;
            }

            bool chat = (path=="/v1/chat/completions");
            bool stream = j.value("stream",false);
            int n_predict = j.contains("max_tokens") ? j["max_tokens"].get<int>()
                          : j.contains("max_completion_tokens") ? j["max_completion_tokens"].get<int>()
                          : 256;
            if(n_predict<=0 || n_predict>ctx) n_predict=(256<ctx)?256:ctx;
            float temp = j.contains("temperature") ? j["temperature"].get<float>() : 1.0f;
            int tk = j.contains("top_k") ? j["top_k"].get<int>() : 0;
            if(j.contains("top_p")) { /* not implemented: accepted and ignored */ }
            bool thinking = true;
            if(j.contains("enable_thinking")) thinking=j["enable_thinking"].get<bool>();
            if(j.contains("chat_template_kwargs") && j["chat_template_kwargs"].is_object() &&
               j["chat_template_kwargs"].contains("enable_thinking"))
                thinking=j["chat_template_kwargs"]["enable_thinking"].get<bool>();

            std::string prompt;
            if(chat){
                if(!j.contains("messages") || !j["messages"].is_array()){
                    kindi_http::respond_err(cs,400,"messages[] required");
                    closesocket(cs); continue;
                }
                prompt=render_chat(j["messages"],true,thinking);
            } else {
                if(j.contains("prompt")){
                    if(j["prompt"].is_string()) prompt=j["prompt"].get<std::string>();
                    else if(j["prompt"].is_array() && !j["prompt"].empty() && j["prompt"][0].is_string())
                        prompt=j["prompt"][0].get<std::string>();
                }
                if(prompt.empty()){ kindi_http::respond_err(cs,400,"prompt required"); closesocket(cs); continue; }
            }

            std::vector<int> ids=tok.encode(prompt);
            if(ids.empty()) ids.push_back(0);

            char cid[64]; snprintf(cid,sizeof(cid),"chatcmpl-%08x",(unsigned)((unsigned long long)ktrace::now_us()&0xffffffffULL));
            long long created=(long long)time(nullptr);

            std::string content, reasoning;
            bool in_think = thinking;      // the template opened a think block
            kindi_http::Sse sse; sse.s=cs;
            if(stream){
                sse.begin();
                // role chunk first, as OpenAI clients expect
                std::string d = "{\"id\":\""+std::string(cid)+"\",\"object\":\"chat.completion.chunk\",\"created\":"
                              + std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                              +"\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}";
                if(!chat) d="{\"id\":\""+std::string(cid)+"\",\"object\":\"text_completion\",\"created\":"
                            +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                            +"\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":null}]}";
                sse.data(d);
            }

            auto emit=[&](int, const std::string& piece){
                // split at </think> so reasoning is reported separately. Only for the chat
                // endpoint: a text completion has no reasoning field, so its output must
                // not be swallowed by the think block.
                std::string text=piece;
                if(in_think && chat){
                    size_t e=text.find("</think>");
                    if(e==std::string::npos){
                        reasoning+=text;
                        if(stream){
                            std::string d="{\"id\":\""+std::string(cid)+"\",\"object\":\"chat.completion.chunk\",\"created\":"
                                         +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                                         +"\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\""
                                         +kindi_http::json_escape(text)+"\"},\"finish_reason\":null}]}";
                            sse.data(d);
                        }
                        return;
                    }
                    std::string pre=text.substr(0,e);
                    reasoning+=pre;
                    if(stream && !pre.empty()){
                        std::string d="{\"id\":\""+std::string(cid)+"\",\"object\":\"chat.completion.chunk\",\"created\":"
                                     +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                                     +"\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\""
                                     +kindi_http::json_escape(pre)+"\"},\"finish_reason\":null}]}";
                        sse.data(d);
                    }
                    in_think=false;
                    text=text.substr(e+9);
                    while(!text.empty() && (text[0]=='\n'||text[0]=='\r')) text.erase(0,1);
                }
                if(text.empty()) return;
                content+=text;
                if(stream){
                    std::string d = chat
                        ? "{\"id\":\""+std::string(cid)+"\",\"object\":\"chat.completion.chunk\",\"created\":"
                          +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                          +"\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
                          +kindi_http::json_escape(text)+"\"},\"finish_reason\":null}]}"
                        : "{\"id\":\""+std::string(cid)+"\",\"object\":\"text_completion\",\"created\":"
                          +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                          +"\",\"choices\":[{\"index\":0,\"text\":\""
                          +kindi_http::json_escape(text)+"\",\"finish_reason\":null}]}";
                    sse.data(d);
                }
            };

            GStats st=generate(ids,n_predict,temp,tk,emit);

            if(stream){
                std::string fin = chat
                    ? "{\"id\":\""+std::string(cid)+"\",\"object\":\"chat.completion.chunk\",\"created\":"
                      +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                      +"\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}"
                    : "{\"id\":\""+std::string(cid)+"\",\"object\":\"text_completion\",\"created\":"
                      +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                      +"\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}]}";
                sse.data(fin);
                sse.data("[DONE]");
                sse.done();
            } else {
                std::string msg="{\"role\":\"assistant\",\"content\":\""+kindi_http::json_escape(content)+"\"";
                if(!reasoning.empty()) msg+=",\"reasoning_content\":\""+kindi_http::json_escape(reasoning)+"\"";
                msg+="}";
                std::string body;
                if(chat){
                    body="{\"id\":\""+std::string(cid)+"\",\"object\":\"chat.completion\",\"created\":"
                        +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                        +"\",\"choices\":[{\"index\":0,\"message\":"+msg+",\"finish_reason\":\"stop\"}],"
                        "\"usage\":{\"prompt_tokens\":"+std::to_string(st.prompt)
                        +",\"completion_tokens\":"+std::to_string(st.completion)
                        +",\"total_tokens\":"+std::to_string(st.prompt+st.completion)+"}}";
                } else {
                    body="{\"id\":\""+std::string(cid)+"\",\"object\":\"text_completion\",\"created\":"
                        +std::to_string(created)+",\"model\":\""+kindi_http::json_escape(model_id)
                        +"\",\"choices\":[{\"index\":0,\"text\":\""+kindi_http::json_escape(content)
                        +"\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":"
                        +std::to_string(st.prompt)+",\"completion_tokens\":"+std::to_string(st.completion)
                        +",\"total_tokens\":"+std::to_string(st.prompt+st.completion)+"}}";
                }
                kindi_http::respond_json(cs,200,body);
            }
            double tot=st.prefill_ms+st.decode_ms;
            fprintf(stderr,"api: %d prompt + %d gen in %.0f ms (prefill %.0f, decode %.0f) = %.2f tok/s\n",
                    st.prompt, st.completion, tot, st.prefill_ms, st.decode_ms,
                    st.decode_ms>0 ? st.completion/(st.decode_ms/1000.0) : 0.0);
            closesocket(cs);
        }
        return 0;
    }

    if(spec_n>0){
        if(!have_mtp){ fprintf(stderr,"--spec: no usable MTP block in this model\n"); return 1; }
        { const char* g5=getenv("KINDI_G5");
          if(g5 && !strcmp(g5,"byte4")){
              fprintf(stderr,"--spec: KINDI_G5=byte4 is not bit-identical to the batched "
                             "kernel, so it cannot be used for speculative verify\n");
              return 1;
          } }
        if(temperature>0.0f){ fprintf(stderr,"--spec: needs --temp 0 (greedy)\n"); return 1; }
        if(CHUNK>0){ fprintf(stderr,"--spec: set KINDI_CHUNK=0\n"); return 1; }
        if(ctx < (int)ids.size()+spec_n+4){ fprintf(stderr,"--spec: context too small\n"); return 1; }
    }
    if(mtp_accept>0){
        if(!have_mtp){ fprintf(stderr,"--mtp-accept: no usable MTP block in this model\n"); return 1; }
        if(temperature>0.0f){ fprintf(stderr,"--mtp-accept: needs --temp 0 (greedy)\n"); return 1; }
        if(CHUNK>0){ fprintf(stderr,"--mtp-accept: set KINDI_CHUNK=0\n"); return 1; }
        if(HB>512 && !native5){ fprintf(stderr,"--mtp-accept: this model needs --native5 (HB=%d embedding fold)\n",HB); return 1; }
        if(ids.empty()) return 1;
    }
    if(CHUNK>0){
        // chunked prefill: T tokens per pass, falling back to serial per token if the
        // chunked path declines (e.g. an unsupported tensor format).
        size_t i=0, last=ids.size()-1;
        int nchunk_used=0, nserial=0;
        while(i<last){
            size_t remain=last-i;
            int T=(int)((remain<(size_t)CHUNK)?remain:(size_t)CHUNK);
            std::vector<int> ch(ids.begin()+i, ids.begin()+i+T);
            if(body_T(ch,(int)i)){ nchunk_used++; }
            else { for(int t=0;t<T;++t) body(ch[t],(int)i+t); nserial+=T; }
            i+=T;
        }
        fprintf(stderr,"prefill: %d chunked passes, %d serial tokens\n", nchunk_used, nserial);
    } else if(mtp_accept>0 || spec_n>0){
        // MTP catch-up interleaved with the trunk prefill: the draft layer's attention needs
        // its own K/V at every prompt position, built from (h_{p-1}, x_p). Position 0 pairs
        // with a zero h, as the reference starts pending_h at 0.
        if(ids.size()>=2){ ktrace::set_tag("mtp:"); mtp_block(mtp_hprev, ids[0], 0, false); ktrace::set_tag(""); }
        for(size_t i=0;i+1<ids.size();++i){
            body(ids[i], (int)i);
            if(i+2<ids.size()){
                ktrace::set_tag("mtp:");
                mtp_addv(h, y, mtp_hprev, D);            // h_i = h + y (body() defers the add)
                mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
                if(dumpvec) dumpfull(("mtp_h"+std::to_string(i)).c_str(), mtp_hin, D);
                mtp_block(mtp_hin, ids[i+1], (int)i+1, false);
                ktrace::set_tag("");
            }
        }
        if(ids.size()>=2){
            ktrace::set_tag("mtp:");
            mtp_addv(h, y, mtp_hprev, D);
            mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
            ktrace::set_tag("");
        }
    } else {
        for(size_t i=0;i+1<ids.size();++i) body(ids[i], (int)i);
    }
    int pos=(int)ids.size();
    std::vector<int> out;
    if(spec_n>0){
        // Speculative decode, k=1. The draft at position p predicts x_{p+1} from
        // (h_{p-1}, x_p); the trunk verifies [x_p, draft] in one T=2 pass. If the draft
        // matches the trunk's row 0 the draft is accepted and row 1 is a valid extra
        // token, so one pass yields 2 tokens; otherwise 1.
        int p=(int)ids.size()-1;
        int last=ids.back();
        // mtp_hin already holds rms(h_{N-2}, output_norm) from the prefill catch-up, and
        // the MTP KV holds positions 0..N-2. The draft at p writes position p.
        int nacc=0, ndraft=0, npass=0, ntok_spec=0;
        ktrace::mark_decode_start();   // --trace-decode: table starts here
        double t0=ktrace::now_us();
        double t_draft=0, t_verify=0, t_head=0, t_commit=0, t_snap=0;
        auto tick=[&](){ return ktrace::now_us(); };
        ktrace::set_tag("mtp:");
        int d=mtp_block(mtp_hin, last, p, true);
        ktrace::set_tag("");
        double t_loop=0;
        while((int)out.size()<spec_n){
            if(p+2>=ctx) break;
            double w0=tick();
            std::vector<int> fed{last, d};
            double s0=tick();
            spec_snapshot();
            double s1=tick(); t_snap+=s1-s0;
            ktrace::set_tag("spec:");
            bool okT=body_T(fed, p, true);
            ktrace::set_tag("");
            double s2=tick(); t_verify+=s2-s1;
            if(!okT){ fprintf(stderr,"spec: body_T declined (T=2)\n"); break; }
            head_rows(2);
            batch_flush();                       // one readback for both rows
            double s3=tick(); t_head+=s3-s2;
            int r0=((int*)mtp_aout.mapped)[0];
            int r1=((int*)mtp_aout.mapped)[1];
            int j=(r0==d)?1:0;
            static const bool force_reject = getenv("KINDI_SPEC_FORCE_REJECT") != nullptr;
            if(force_reject) j=0;
            int ntok=1+j;
            ndraft++; npass++; nacc+=j;
            if(ntok<2){ spec_restore(); ktrace::set_tag("spec:"); spec_replay(ntok); ktrace::set_tag(""); }
            double s4=tick(); t_commit+=s4-s3;
            out.push_back(r0); ktrace::accuracy_token(r0); ntok_spec++;
            if(j && (int)out.size()<spec_n){ out.push_back(r1); ktrace::accuracy_token(r1); ntok_spec++; }
            // next pass: feed the token at p+ntok
            last = j ? r1 : r0;
            p += ntok;
            if((int)out.size()>=spec_n){ t_loop+=tick()-w0; break; }
            // hidden state of the committed position, for the next draft
            if(j){
                // catch-up: the MTP KV needs position p-1 (the accepted r0)
                row_h(0);
                mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
                ktrace::set_tag("mtp:");
                mtp_block(mtp_hin, r0, p-1, false);
                ktrace::set_tag("");
                row_h(1);
            } else {
                row_h(0);
            }
            mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
            ktrace::set_tag("mtp:");
            d=mtp_block(mtp_hin, last, p, true);
            ktrace::set_tag("");
            t_draft+=tick()-s4;
            t_loop+=tick()-w0;
        }
        double t1=ktrace::now_us();
        fprintf(stderr,"spec: %d passes, %d tokens, %d/%d drafts accepted = %.1f%%\n",
                npass, ntok_spec, nacc, ndraft, ndraft?100.0*nacc/(double)ndraft:0.0);
        double t_sum=t_verify+t_head+t_commit+t_draft+t_snap;
        fprintf(stderr,"spec: phases per pass: verify %.0f + head %.0f + commit %.0f + draft %.0f + snap %.0f = %.0f ms, loop body %.0f ms\n",
                t_verify/1000.0/npass, t_head/1000.0/npass, t_commit/1000.0/npass,
                t_draft/1000.0/npass, t_snap/1000.0/npass,
                t_sum/1000.0/npass, t_loop/1000.0/npass);
        fprintf(stderr,"spec: %.1f ms total, %.1f ms/pass, %.3f tok/pass, %.2f tok/s\n",
                (t1-t0)/1000.0, npass?(t1-t0)/1000.0/npass:0.0,
                npass?(double)ntok_spec/npass:0.0, (t1-t0)>0?ntok_spec/((t1-t0)/1e6):0.0);
        fprintf(stderr,"\nGenerated %zu tokens: ", out.size());
        for(int t: out) fprintf(stderr,"%d ", t);
        fprintf(stderr,"\nDecoded: %s\n", tok.decode(out).c_str());
        batch_flush();
        ktrace::dump();
        return 0;
    }
    if(mtp_accept>0){
        // k=1 acceptance oracle: the draft at position p predicts the token at p+1 from
        // (h_{p-1}, x_p); the trunk's greedy step then produces the token at p+1 from the
        // same inputs. The trunk path is unchanged, so the emitted tokens must match the
        // plain run exactly; only the hit count is new information.
        int p=(int)ids.size()-1;
        int last=ids.back();
        // h_{N-2} for the first draft; after each step() the engine's h holds h_p
        mtp_hinput(mtp_hprev, "output_norm.weight", mtp_hin);
        Buf* hp=&mtp_hin;
        int nacc=0, ndraft=0;
        double t_draft=0, t_trunk=0;
        ktrace::mark_decode_start();   // --trace-decode: table starts here
        fprintf(stderr,"mtp: acceptance harness, k=1, greedy (draft at p predicts p+1)\n");
        fprintf(stderr,"  %5s %8s %8s %6s\n","pos","draft","trunk","hit");
        for(int k=0;k<mtp_accept;++k){
            double a0=ktrace::now_us();
            ktrace::set_tag("mtp:");
            int d=mtp_block(*hp, last, p, true);
            ktrace::set_tag("");
            double a1=ktrace::now_us();
            int t=step(last, p);
            double a2=ktrace::now_us();
            t_draft+=a1-a0; t_trunk+=a2-a1;
            bool hit=(d==t);
            ndraft++; if(hit) nacc++;
            if(k<12 || !hit) fprintf(stderr,"  %5d %8d %8d %6s\n",p,d,t,hit?"yes":"NO");
            if(dump&&k<4){
                std::vector<float> L=readbuf(mtp_logits,VOCAB);
                int rank=0; for(int i=0;i<VOCAB;++i) if(L[i]>L[t]) rank++;
                int b1=0,b2=0,b3=0;
                for(int i=1;i<VOCAB;++i){
                    if(L[i]>L[b1]){b3=b2;b2=b1;b1=i;}
                    else if(L[i]>L[b2]){b3=b2;b2=i;}
                    else if(L[i]>L[b3]) b3=i;
                }
                fprintf(stderr,"mtp: rank(trunk %d)=%d  top3=%d(%.3f) %d(%.3f) %d(%.3f)\n",
                        t,rank,b1,L[b1],b2,L[b2],b3,L[b3]);
            }
            out.push_back(t); ktrace::accuracy_token(t);
            last=t; p++;
            mtp_hinput(h, "output_norm.weight", mtp_hin);   // step() left h = h_p
            hp=&mtp_hin;
            if(p>=ctx) break;
        }
        fprintf(stderr,"mtp: accepted %d/%d = %.1f%%  (draft %.2f ms/tok, trunk %.2f ms/tok, +%.1f%%)\n",
                nacc, ndraft, ndraft?100.0*nacc/(double)ndraft:0.0,
                t_draft/1000.0/(ndraft?ndraft:1), t_trunk/1000.0/(ndraft?ndraft:1),
                t_trunk>0.0?100.0*t_draft/t_trunk:0.0);
        fprintf(stderr,"\nGenerated %zu tokens: ", out.size());
        for(int t: out) fprintf(stderr,"%d ", t);
        fprintf(stderr,"\nDecoded: %s\n", tok.decode(out).c_str());
        batch_flush();
        ktrace::dump();
        return 0;
    }
    ktrace::mark_decode_start();   // --trace-decode: table starts here
    double g0=ktrace::now_us();
    int next=step(ids.back(), pos-1);
    out.push_back(next); ktrace::accuracy_token(next);
    pos++;
    while((int)out.size()<ngt){
        next=step(out.back(), pos);
        out.push_back(next); ktrace::accuracy_token(next);
        pos++;
        if(pos>=ctx) break;
    }
    double g1=ktrace::now_us();
    fprintf(stderr,"decode: %zu tokens in %.1f ms = %.3f tok/s (%.1f ms/token)\n",
            out.size(), (g1-g0)/1000.0,
            (g1-g0)>0 ? out.size()/((g1-g0)/1e6) : 0.0,
            out.size() ? (g1-g0)/1000.0/out.size() : 0.0);
    fprintf(stderr,"\nGenerated %zu tokens: ", out.size());
    for(int t: out) fprintf(stderr,"%d ", t);
    if(getenv("KINDI_CMP")){
        std::vector<float> hv=readbuf(h,3), yv=readbuf(y,3);
        fprintf(stderr,"CMP serial h: %.5f %.5f %.5f | y: %.5f %.5f %.5f\n",
                hv[0],hv[1],hv[2],yv[0],yv[1],yv[2]);
        std::vector<float> ht=readbuf(hT,D*CT), yt=readbuf(yT,D*CT);
        for(int t=0;t<CT;++t)
            fprintf(stderr,"CMP chunk  t%d h: %.5f %.5f %.5f | y: %.5f %.5f %.5f\n",t,
                ht[t*D+0],ht[t*D+1],ht[t*D+2],yt[t*D+0],yt[t*D+1],yt[t*D+2]);
    }
    fprintf(stderr,"\nDecoded: %s\n", tok.decode(out).c_str());
    batch_flush();
    ktrace::dump();
    return 0;
}
