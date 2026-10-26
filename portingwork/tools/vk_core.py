"""vk_core: buffers, parallel staging upload, pipeline + dispatch helpers. Compute-only."""
import os
import subprocess
import concurrent.futures
import numpy as np
import vulkan as vk

SHADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
GLSLANG = r"C:\VulkanSDK\1.4.363.0\Bin\glslangValidator.exe"

def compile_all():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
    ok = True
    for f in sorted(os.listdir(d)):
        if not f.endswith(".comp"):
            continue
        src = os.path.join(d, f)
        dst = src + ".spv"
        # skip if up to date
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            print(f"  {f}: cached")
            continue
        r = subprocess.run([GLSLANG, "--target-env", "vulkan1.3", "-V", src, "-o", dst],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dst):
            print(f"  {f}: FAIL\n{r.stdout}\n{r.stderr}")
            ok = False
        else:
            print(f"  {f}: ok ({os.path.getsize(dst)} B)")
    return ok

def mem_props(gpu):
    return vk.vkGetPhysicalDeviceMemoryProperties(gpu)

def find_mem_type(gpu, bits, flags):
    mp = mem_props(gpu)
    for i in range(mp.memoryTypeCount):
        if (bits >> i) & 1 and (int(mp.memoryTypes[i].propertyFlags) & flags) == flags:
            return i
    return -1

class Buf:
    def __init__(self, dev, gpu, size, usage, mem_flags, heap_pref=None):
        self.dev, self.size = dev, size
        ci = vk.VkBufferCreateInfo(sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext=None, flags=0, size=size, usage=usage, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0, pQueueFamilyIndices=None)
        self.buf = vk.vkCreateBuffer(dev, ci, None)
        req = vk.vkGetBufferMemoryRequirements(dev, self.buf)
        idx = find_mem_type(gpu, int(req.memoryTypeBits), mem_flags)
        assert idx >= 0, f"no memory type flags={mem_flags:#x}"
        ai = vk.VkMemoryAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None, allocationSize=int(req.size), memoryTypeIndex=idx)
        self.mem = vk.vkAllocateMemory(dev, ai, None)
        vk.vkBindBufferMemory(dev, self.buf, self.mem, 0)

    def write(self, data: bytes):
        ptr = vk.vkMapMemory(self.dev, self.mem, 0, len(data), 0)
        vk.ffi.memmove(ptr, data, len(data))
        vk.vkUnmapMemory(self.dev, self.mem)

    def read(self, n: int) -> bytes:
        ptr = vk.vkMapMemory(self.dev, self.mem, 0, n, 0)
        out = bytes(vk.ffi.buffer(ptr, n))
        vk.vkUnmapMemory(self.dev, self.mem)
        return out

HOST_COHERENT = int(vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
DEVICE_LOCAL = int(vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

def upload_parallel(dev, gpu, queue, cmdpool, pairs, workers=4):
    """pairs: list of (device_buf: Buf, host_bytes). One staging buf + cmdbuf per chunk,
    submitted together; waits on one fence. Returns after transfer completes."""
    import threading
    n = len(pairs)
    per = (n + workers - 1) // workers
    fns = []
    def _chunk(idxs):
        for i in idxs:
            dbuf, data = pairs[i]
            st = Buf(dev, gpu, len(data), int(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT), HOST_COHERENT)
            st.write(data)
            ai = vk.VkCommandBufferAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext=None, commandPool=cmdpool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1)
            (cb,) = vk.vkAllocateCommandBuffers(dev, ai)
            bi = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext=None, flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo=None)
            vk.vkBeginCommandBuffer(cb, bi)
            cp = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=len(data))
            vk.vkCmdCopyBuffer(cb, st.buf, dbuf.buf, 1, [cp])
            vk.vkEndCommandBuffer(cb)
            fns.append((cb, st))
    idx = list(range(n))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_chunk, [idx[i:i+per] for i in range(0, n, per)]))
    si = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, pNext=None,
        waitSemaphoreCount=0, pWaitSemaphores=None, pWaitDstStageMask=None,
        commandBufferCount=len(fns), pCommandBuffers=[c for c, _ in fns],
        signalSemaphoreCount=0, pSignalSemaphores=None)
    fi = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext=None, flags=0)
    fence = vk.vkCreateFence(dev, fi, None)
    vk.vkQueueSubmit(queue, 1, [si], fence)
    vk.vkWaitForFences(dev, 1, [fence], True, 10_000_000_000)
    vk.vkDestroyFence(dev, fence, None)
    for cb, st in fns:
        vk.vkFreeCommandBuffers(dev, cmdpool, 1, [cb])
        vk.vkDestroyBuffer(dev, st.buf, None)
        vk.vkFreeMemory(dev, st.mem, None)

def make_pipeline(dev, spv_path, set_layout_bindings, push_size):
    code = open(spv_path, "rb").read()
    mod = vk.vkCreateShaderModule(dev, vk.VkShaderModuleCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, pNext=None,
        flags=0, codeSize=len(code), pCode=bytearray(code)), None)
    binds = []
    for b, dtype in set_layout_bindings:
        binds.append(vk.VkDescriptorSetLayoutBinding(binding=b,
            descriptorType=dtype, descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None))
    dsl = vk.vkCreateDescriptorSetLayout(dev, vk.VkDescriptorSetLayoutCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pNext=None,
        flags=0, bindingCount=len(binds), pBindings=binds), None)
    pr = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=push_size)
    pl = vk.vkCreatePipelineLayout(dev, vk.VkPipelineLayoutCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, pNext=None, flags=0,
        setLayoutCount=1, pSetLayouts=[dsl], pushConstantRangeCount=1, pPushConstantRanges=[pr]), None)
    stage = vk.VkPipelineShaderStageCreateInfo(sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        pNext=None, flags=0, stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main", pSpecializationInfo=None)
    pipe = vk.vkCreateComputePipelines(dev, None, 1, [vk.VkComputePipelineCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, pNext=None, flags=0,
        stage=stage, layout=pl, basePipelineHandle=None, basePipelineIndex=0)], None)[0]
    pool = vk.vkCreateDescriptorPool(dev, vk.VkDescriptorPoolCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, pNext=None, flags=0,
        maxSets=1, poolSizeCount=1, pPoolSizes=[vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=len(binds))]), None)
    dset = vk.vkAllocateDescriptorSets(dev, vk.VkDescriptorSetAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, pNext=None,
        descriptorPool=pool, descriptorSetCount=1, pSetLayouts=[dsl]))[0]
    return mod, dsl, pl, pipe, pool, dset

def dispatch(dev, queue, cmdpool, pipe, pl, dset, groups, push_bytes):
    ai = vk.VkCommandBufferAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        pNext=None, commandPool=cmdpool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1)
    (cb,) = vk.vkAllocateCommandBuffers(dev, ai)
    vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext=None,
        flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo=None))
    vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe)
    vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, [dset], 0, None)
    if push_bytes:
        vk.vkCmdPushConstants(cb, pl, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(push_bytes), push_bytes)
    vk.vkCmdDispatch(cb, *groups)
    vk.vkEndCommandBuffer(cb)
    fi = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext=None, flags=0)
    fence = vk.vkCreateFence(dev, fi, None)
    vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        pNext=None, waitSemaphoreCount=0, pWaitSemaphores=None, pWaitDstStageMask=None,
        commandBufferCount=1, pCommandBuffers=[cb], signalSemaphoreCount=0, pSignalSemaphores=None)], fence)
    vk.vkWaitForFences(dev, 1, [fence], True, 30_000_000_000)
    vk.vkDestroyFence(dev, fence, None)
    vk.vkFreeCommandBuffers(dev, cmdpool, 1, [cb])
