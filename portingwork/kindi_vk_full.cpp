// kindi-vk full 2B decode engine
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
#include <deque>
#include "trace.h"
#include <map>
#include <unordered_map>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){fprintf(stderr,"Vulkan error %d at %s:%d\n",(int)_r,__FILE__,__LINE__);exit(1);} } while(0)

// ---- device helpers ----
struct DeviceCtx { VkInstance inst; VkPhysicalDevice phys; VkDevice dev; VkQueue q0,q1,qt; uint32_t cf,tf; VkCommandPool cp,tp; VkPhysicalDeviceMemoryProperties mp; VkPhysicalDeviceProperties pp; };
static DeviceCtx G;
static uint32_t findmt(uint32_t bits, VkMemoryPropertyFlags f){ for(uint32_t i=0;i<G.mp.memoryTypeCount;++i) if(((bits>>i)&1)&&((G.mp.memoryTypes[i].propertyFlags&f)==f)) return i; fprintf(stderr,"nomemtype\n");exit(1);}
struct Buf { VkBuffer buf=VK_NULL_HANDLE; VkDeviceMemory mem=VK_NULL_HANDLE; VkDeviceSize size=0; void* mapped=nullptr; };
static Buf mkbuf(VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags f){
    Buf b; b.size=sz; VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=sz; ci.usage=u; ci.sharingMode=VK_SHARING_MODE_CONCURRENT; uint32_t qf[2]={G.cf,G.tf}; ci.queueFamilyIndexCount=2; ci.pQueueFamilyIndices=qf;
    VK_CHECK(vkCreateBuffer(G.dev,&ci,nullptr,&b.buf));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(G.dev,b.buf,&r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=r.size; ai.memoryTypeIndex=findmt((uint32_t)r.memoryTypeBits,f);
    VK_CHECK(vkAllocateMemory(G.dev,&ai,nullptr,&b.mem)); VK_CHECK(vkBindBufferMemory(G.dev,b.buf,b.mem,0));
    if(f&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VK_CHECK(vkMapMemory(G.dev,b.mem,0,sz,0,&b.mapped));
    return b;
}
static void freebuf(Buf&b){ if(b.mapped)vkUnmapMemory(G.dev,b.mem); if(b.buf)vkDestroyBuffer(G.dev,b.buf,nullptr); if(b.mem)vkFreeMemory(G.dev,b.mem,nullptr); b={}; }
static void initdev(){
    VkApplicationInfo a{VK_STRUCTURE_TYPE_APPLICATION_INFO}; a.pApplicationName="kindi-vk"; a.applicationVersion=1; a.pEngineName="kindi-vk"; a.engineVersion=1; a.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&a; VK_CHECK(vkCreateInstance(&ici,nullptr,&G.inst));
    uint32_t n=0; vkEnumeratePhysicalDevices(G.inst,&n,nullptr); std::vector<VkPhysicalDevice> ds(n); vkEnumeratePhysicalDevices(G.inst,&n,ds.data()); G.phys=ds[0];
    vkGetPhysicalDeviceProperties(G.phys,&G.pp); vkGetPhysicalDeviceMemoryProperties(G.phys,&G.mp); fprintf(stderr,"GPU: %s\n",G.pp.deviceName);
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(G.phys,&qn,nullptr); std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(G.phys,&qn,qf.data());
    G.cf=UINT32_MAX; G.tf=UINT32_MAX;
    for(uint32_t i=0;i<qn;++i){ if(!(qf[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)&&(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT)&&qf[i].queueCount>=2&&G.cf==UINT32_MAX)G.cf=i; if(!(qf[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)&&(qf[i].queueFlags&VK_QUEUE_TRANSFER_BIT)&&G.tf==UINT32_MAX&&i!=G.cf)G.tf=i; }
    assert(G.cf!=UINT32_MAX&&G.tf!=UINT32_MAX);
    std::vector<VkDeviceQueueCreateInfo> qcis;
    for(int qi=0;qi<2;++qi){ VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueFamilyIndex=G.cf; q.queueCount=1; static float p=1.0f; q.pQueuePriorities=&p; qcis.push_back(q); }
    { VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueFamilyIndex=G.tf; q.queueCount=1; static float p=1.0f; q.pQueuePriorities=&p; qcis.push_back(q); }
    VkPhysicalDeviceFeatures f{}; f.shaderFloat64=VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=(uint32_t)qcis.size(); dci.pQueueCreateInfos=qcis.data(); dci.pEnabledFeatures=&f;
    VK_CHECK(vkCreateDevice(G.phys,&dci,nullptr,&G.dev));
    vkGetDeviceQueue(G.dev,G.cf,0,&G.q0); vkGetDeviceQueue(G.dev,G.cf,1,&G.q1); vkGetDeviceQueue(G.dev,G.tf,0,&G.qt);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex=G.cf; cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(G.dev,&cpci,nullptr,&G.cp)); cpci.queueFamilyIndex=G.tf; VK_CHECK(vkCreateCommandPool(G.dev,&cpci,nullptr,&G.tp));
    if(ktrace::enabled()) ktrace::init(G.phys,G.dev);
}
static std::vector<uint32_t> readspv(const char*p){ std::ifstream f(p,std::ios::binary|std::ios::ate); if(!f){fprintf(stderr,"nospv %s\n",p);exit(1);} size_t sz=(size_t)f.tellg(); f.seekg(0); std::vector<uint32_t> b(sz/4); f.read((char*)b.data(),sz); return b; }

struct Pipe { VkShaderModule mod; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe;
             VkDescriptorPool pool; VkDescriptorSet dset;
             const char* name=nullptr; uint32_t nbind=0; uint64_t brange[16]={0}; uint32_t lx=0; };
static Pipe mkpipe(const char* spv, int nbind, int pushsz){
    Pipe P{}; auto code=readspv(spv);
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
        static std::deque<std::string> keep;
        keep.push_back(std::string(b));
        std::string& k=keep.back(); size_t dot=k.find(".comp"); if(dot!=std::string::npos) k=k.substr(0,dot);
        P.name=k.c_str();
    }
    return P;
}

static void bindbufs(Pipe&P, const std::vector<Buf>& bufs){
    std::vector<VkDescriptorBufferInfo> infos(bufs.size()); std::vector<VkWriteDescriptorSet> w(bufs.size());
    for(size_t i=0;i<bufs.size();++i){ infos[i].buffer=bufs[i].buf; infos[i].offset=0; infos[i].range=bufs[i].size;
        w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].pNext=nullptr; w[i].dstSet=P.dset; w[i].dstBinding=(uint32_t)i; w[i].dstArrayElement=0; w[i].descriptorCount=1; w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pImageInfo=nullptr; w[i].pBufferInfo=&infos[i]; w[i].pTexelBufferView=nullptr; }
    vkUpdateDescriptorSets(G.dev,(uint32_t)w.size(),w.data(),0,nullptr);
    for(size_t i=0;i<bufs.size() && i<16;++i) P.brange[i]=infos[i].range;
}
static uint64_t pipe_bytes(const Pipe&P){
    uint64_t b=0; for(uint32_t i=0;i<P.nbind && i<16;++i) b+=P.brange[i]; return b;
}
static void dispatch(const Pipe&P, VkQueue q, VkCommandPool pool, uint32_t gx,uint32_t gy,uint32_t gz, const void*push,uint32_t psz){
    const bool tr = ktrace::enabled();
    double th0 = tr ? ktrace::now_us() : 0.0;
    uint32_t pair = UINT32_MAX;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=pool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(G.dev,&ai,&cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VK_CHECK(vkBeginCommandBuffer(cb,&bi));
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P.pipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,P.pl,0,1,&P.dset,0,nullptr);
    if(push&&psz) vkCmdPushConstants(cb,P.pl,VK_SHADER_STAGE_COMPUTE_BIT,0,psz,push);
    if(tr) ktrace::mark_before(cb,pair);
    vkCmdDispatch(cb,gx,gy,gz);
    if(tr) ktrace::mark_after(cb,pair);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VK_CHECK(vkCreateFence(G.dev,&fi,nullptr,&fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb; VK_CHECK(vkQueueSubmit(q,1,&si,fence));
    ktrace::count_submit();
    VK_CHECK(vkWaitForFences(G.dev,1,&fence,VK_TRUE,60000000000ULL));
    ktrace::count_fence();
    vkDestroyFence(G.dev,fence,nullptr); vkFreeCommandBuffers(G.dev,pool,1,&cb);
    if(tr){
        double th1 = ktrace::now_us();
        ktrace::record(P.name?P.name:"?", gx,gy,gz, P.lx, psz, pipe_bytes(P), th1-th0, -1.0, pair);
    }
}

static void upload(Buf*dst, const void*data, size_t nb){
    Buf st=mkbuf(nb,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(st.mapped,data,nb);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=G.tp; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer c; VK_CHECK(vkAllocateCommandBuffers(G.dev,&ai,&c));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VK_CHECK(vkBeginCommandBuffer(c,&bi));
    VkBufferCopy cp{0,0,(VkDeviceSize)nb}; vkCmdCopyBuffer(c,st.buf,dst->buf,1,&cp); VK_CHECK(vkEndCommandBuffer(c));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; VK_CHECK(vkCreateFence(G.dev,&fi,nullptr,&fe));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c; VK_CHECK(vkQueueSubmit(G.qt,1,&si,fe));
    VK_CHECK(vkWaitForFences(G.dev,1,&fe,VK_TRUE,60000000000ULL)); vkDestroyFence(G.dev,fe,nullptr); vkFreeCommandBuffers(G.dev,G.tp,1,&c); freebuf(st);
}

// ---- main: load weights, print status ----
int main(int argc, char**argv){
    const char* pack=nullptr;
    for(int i=1;i<argc;++i) if(!strcmp(argv[i],"--pack")&&i+1<argc) pack=argv[++i];
    if(!pack){ fprintf(stderr,"usage: --pack <dir>\n"); return 1; }
    initdev();
    std::ifstream mf(std::string(pack)+"/manifest.json"); json man; mf>>man;
    int D=man["dims"]["width"], NL=man["dims"]["layers"], NFF=man["dims"]["ffn"];
    int NH=man["dims"]["heads"], NHKV=man["dims"]["kv_heads"], HD=man["dims"]["head_dim"];
    int NK=man["dims"]["groups"], HK=man["dims"]["state_size"], NV=man["dims"]["gdn_inner"]/HK;
    int rope_dim=man["dims"]["rope_dim"], interval=man["dims"]["full_attention_interval"];
    int HB=man["dims"]["hb"]; int NATT=NL/interval; int VOCAB=248320; int ctx=4096;
    fprintf(stderr,"dims: L=%d D=%d FFN=%d H=%d KV=%d HD=%d G=%d HK=%d NV=%d HB=%d\n",NL,D,NFF,NH,NHKV,HD,NK,HK,NV,HB);
    auto loadfile=[&](const char*name){ std::ifstream f(std::string(pack)+"/"+name,std::ios::binary|std::ios::ate); size_t sz=f.tellg(); f.seekg(0); std::vector<char> b(sz); f.read(b.data(),sz); return b; };
    auto codes_data=loadfile("codes.bin"); auto scales_data=loadfile("scales.bin"); auto f32_data=loadfile("f32.bin");
    std::map<std::string,Buf> codebufs, scalebufs, f32bufs;
    std::map<std::string,std::pair<int,int>> tinfo;
    for(auto&[name,t]:man["tensors"].items()){
        if(t["type"]=="ptq"){
            int rows=t["rows"], cols=t["cols"]; int co=t["code_off"], so=t["scale_off"]; int cb=t["code_bytes"], sb=t["scale_bytes"];
            Buf c=mkbuf(cb,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&c,codes_data.data()+co,cb);
            Buf s=mkbuf(sb,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&s,scales_data.data()+so,sb);
            codebufs[name]=c; scalebufs[name]=s; tinfo[name]={rows,cols};
        } else {
            int rows=t["rows"], cols=t["cols"]; int fo=t["f32_off"]; int fb=t["f32_bytes"];
            Buf f=mkbuf(fb,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); upload(&f,f32_data.data()+fo,fb);
            f32bufs[name]=f; tinfo[name]={rows,cols};
        }
    }
    fprintf(stderr,"weights uploaded: %zu ternary + %zu f32\n", codebufs.size(), f32bufs.size());

    // ---- decode loop (simplified: single token, no KV cache persistence across calls) ----
    // For now, demonstrate that the pipeline works by running one GEMV
    std::string shdir="shaders";
    Pipe pg4=mkpipe((shdir+"/g4simple.comp.spv").c_str(),4,8);
    // test GEMV on token_embd.weight
    {
        auto [rows,cols]=tinfo["token_embd.weight"];
        Buf x=mkbuf(cols*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        for(int i=0;i<cols;++i) ((float*)x.mapped)[i]=1.0f; // dummy input
        Buf out=mkbuf(rows*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::vector<Buf> bufs={codebufs["token_embd.weight"],scalebufs["token_embd.weight"],x,out};
        bindbufs(pg4,bufs);
        struct{int R,C;}pc{rows,cols};
        dispatch(pg4,G.q0,G.cp,rows,1,1,&pc,8);
        fprintf(stderr,"GEMV token_embd: rows=%d cols=%d out[0]=%.4f\n",rows,cols,((float*)out.mapped)[0]);
    }
    fprintf(stderr,"engine ready. Full decode loop: use --n to generate (not yet wired).\n");
    ktrace::dump();
    return 0;
    }
