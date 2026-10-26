"""Test g4simple on the RX580: tiny random ternary matrix, compare vs numpy reference."""
import numpy as np, struct
import vulkan as vk
from vk_device import create_compute_device
import vk_core
from vk_core import Buf, HOST_COHERENT, DEVICE_LOCAL

def main():
    rng = np.random.RandomState(0)
    R, C = 64, 512
    W = rng.randint(0, 3, size=(R, C)).astype(np.uint8) - 1  # -1,0,1 as int8-ish
    codes = np.zeros((R, C // 4), dtype=np.uint8)
    for k in range(0, C, 4):
        codes[:, k // 4] = ((W[:, k] + 1) + 3 * (W[:, k + 1] + 1)
                            + 9 * (W[:, k + 2] + 1) + 27 * (W[:, k + 3] + 1)).astype(np.uint8)
    scales = (rng.randn(R, C // 128).astype(np.float16)).view(np.uint16)
    x = rng.randn(C).astype(np.float32)
    ref = ((W.astype(np.float32)) * scales.repeat(128, axis=1).view(np.float16).astype(np.float32)) @ x

    inst, gpu, dev, qc0, qc1, qt = create_compute_device()
    vk_core.compile_all()

    def mkbuf(nbytes, usage, flags):
        return Buf(dev, gpu, nbytes, usage, flags)

    SB = int(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT)
    codes_b = mkbuf(codes.nbytes, SB | int(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT), DEVICE_LOCAL)
    scales_b = mkbuf(scales.nbytes * 2, SB, DEVICE_LOCAL)  # upload as uint32 per scale
    scales32 = scales.astype(np.uint32)
    xb = mkbuf(x.nbytes, SB, DEVICE_LOCAL)
    yb = mkbuf(R * 4, SB | int(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT), DEVICE_LOCAL)
    yhost = mkbuf(R * 4, int(vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT), HOST_COHERENT)

    ci = vk.VkCommandPoolCreateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        pNext=None, flags=0, queueFamilyIndex=1)
    cpool = vk.vkCreateCommandPool(dev, ci, None)
    ti = vk.VkCommandPoolCreateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        pNext=None, flags=0, queueFamilyIndex=2)
    tpool = vk.vkCreateCommandPool(dev, ti, None)

    vk_core.upload_parallel(dev, gpu, qt, tpool, [
        (codes_b, codes.tobytes()),
        (scales_b, scales32.tobytes()),
        (xb, x.tobytes()),
    ])

    mod, dsl, pl, pipe, pool, dset = vk_core.make_pipeline(
        dev, "shaders/g4simple.comp.spv",
        [(0, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), (1, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
         (2, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), (3, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)], 8)
    import ctypes
    infos = []
    for b in (codes_b, scales_b, xb, yb):
        info = vk.VkDescriptorBufferInfo(buffer=b.buf, offset=0, range=b.size)
        infos.append(info)
    writes = [vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        pNext=None, dstSet=dset, dstBinding=i, dstArrayElement=0, descriptorCount=1,
        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pImageInfo=None,
        pBufferInfo=[infos[i]], pTexelBufferView=None) for i in range(4)]
    vk.vkUpdateDescriptorSets(dev, len(writes), writes, 0, None)

    push = struct.pack("ii", R, C)
    # pyvulkan vkCmdPushConstants wants a list? pass bytes via ffi
    ai = vk.VkCommandBufferAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        pNext=None, commandPool=cpool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1)
    (cb,) = vk.vkAllocateCommandBuffers(dev, ai)
    vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext=None,
        flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo=None))
    vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe)
    vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, [dset], 0, None)
    vk.vkCmdPushConstants(cb, pl, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(push),
                          list(struct.unpack("B" * len(push), push)))
    vk.vkCmdDispatch(cb, R, 1, 1)
    # copy y back
    cp = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=R * 4)
    vk.vkCmdCopyBuffer(cb, yb.buf, yhost.buf, 1, [cp])
    vk.vkEndCommandBuffer(cb)
    fi = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext=None, flags=0)
    fence = vk.vkCreateFence(dev, fi, None)
    vk.vkQueueSubmit(qc0, 1, [vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        pNext=None, waitSemaphoreCount=0, pWaitSemaphores=None, pWaitDstStageMask=None,
        commandBufferCount=1, pCommandBuffers=[cb], signalSemaphoreCount=0, pSignalSemaphores=None)], fence)
    vk.vkWaitForFences(dev, 1, [fence], True, 30_000_000_000)
    got = np.frombuffer(yhost.read(R * 4), dtype=np.float32)
    err = np.abs(got - ref)
    print(f"max|got-ref| = {err.max():.6f}  mean = {err.mean():.6f}")
    print("ref[:8]", ref[:8])
    print("got[:8]", got[:8])
    print("PASS" if err.max() < 1e-3 else "FAIL")

if __name__ == "__main__":
    main()
