"""Compute-only Vulkan device for kindi-vk. Family 1 (COMPUTE, no GRAPHICS) + family 2 TRANSFER.
Priority: REALTIME on first compute queue (high priority compute component), HIGH on second.
Family 0 (3D/graphics) is NEVER created or used."""
import vulkan as vk

def _make_prio_chain(prio):
    try:
        return vk.ffi.new(
            "VkDeviceQueueGlobalPriorityCreateInfoKHR*",
            {"sType": vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
             "pNext": vk.ffi.NULL, "globalPriority": prio},
        )
    except Exception:
        return None

def create_compute_device():
    inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="kindi-vk", applicationVersion=1,
            pEngineName="kindi-vk", engineVersion=1,
            apiVersion=vk.VK_MAKE_VERSION(1, 3, 0))), None)
    gpus = vk.vkEnumeratePhysicalDevices(inst)
    assert len(gpus) >= 1, "no Vulkan GPUs"
    gpu = gpus[0]
    props = vk.vkGetPhysicalDeviceProperties(gpu)
    print(f"GPU: {props.deviceName}")
    qfams = vk.vkGetPhysicalDeviceQueueFamilyProperties(gpu)
    for qi, q in enumerate(qfams):
        print(f"  family {qi}: flags={int(q.queueFlags)} count={q.queueCount}")
    # verify family 1 is compute-only (no graphics)
    f1 = qfams[1]
    assert not (int(f1.queueFlags) & vk.VK_QUEUE_GRAPHICS_BIT), "family 1 has GRAPHICS - wrong family!"
    assert int(f1.queueFlags) & vk.VK_QUEUE_COMPUTE_BIT, "family 1 lacks COMPUTE!"

    # device extensions: enable global priority if present
    exts = [e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(gpu, None)]
    use_prio = "VK_KHR_global_priority" in exts or "VK_EXT_global_priority" in exts
    print(f"  global-priority ext: {use_prio}")

    qinfos = []
    _keepalive = []
    # family 1: 2 compute queues. First with REALTIME, second with HIGH.
    for qi, prio in [(0, vk.VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR),
                     (1, vk.VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR)]:
        qinfo = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext=None, flags=0, queueFamilyIndex=1,
            queueCount=1, pQueuePriorities=[1.0],
        )
        if use_prio:
            chain = _make_prio_chain(prio)
            if chain is not None:
                # pass pNext at construction: cffi pointer stays valid via _keepalive
                qinfo = vk.VkDeviceQueueCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    pNext=chain, flags=0, queueFamilyIndex=1,
                    queueCount=1, pQueuePriorities=[1.0])
                _keepalive.append(chain)
        qinfos.append(qinfo)
    # family 2: 1 transfer queue for parallel staging uploads (never graphics)
    qinfos.append(vk.VkDeviceQueueCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        pNext=None, flags=0, queueFamilyIndex=2,
        queueCount=1, pQueuePriorities=[1.0]))

    feats = vk.VkPhysicalDeviceFeatures(
        shaderFloat64=True,  # fp64 norm sums like CUDA
        shaderInt64=True, shaderInt16=True,
        shaderStorageImageWriteWithoutFormat=True,
    )
    # enable 8/16-bit storage + float16/int8 features via pNext chain
    try:
        f8 = vk.VkPhysicalDevice8BitStorageFeatures(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
            pNext=None, storageBuffer8BitAccess=True, uniformAndStorageBuffer8BitAccess=True)
        f16 = vk.VkPhysicalDevice16BitStorageFeatures(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
            pNext=f8, storageBuffer16BitAccess=True, uniformAndStorageBuffer16BitAccess=True)
        f_f16i8 = vk.VkPhysicalDeviceShaderFloat16Int8Features(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
            pNext=f16, shaderInt8=True)
        feats_pnext = f_f16i8
    except Exception as e:
        print("  (feature chain skipped:", e, ")")
        feats_pnext = None

    enabled_exts = []
    if use_prio:
        if "VK_KHR_global_priority" in exts:
            enabled_exts.append("VK_KHR_global_priority")

    dinfo = vk.VkDeviceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        pNext=feats_pnext,
        flags=0, queueCreateInfoCount=len(qinfos), pQueueCreateInfos=qinfos,
        enabledLayerCount=0, ppEnabledLayerNames=None,
        enabledExtensionCount=len(enabled_exts),
        ppEnabledExtensionNames=enabled_exts if enabled_exts else None,
        pEnabledFeatures=feats,
    )
    dev = vk.vkCreateDevice(gpu, dinfo, None)
    q_compute0 = vk.vkGetDeviceQueue(dev, 1, 0)
    q_compute1 = vk.vkGetDeviceQueue(dev, 1, 1)
    q_transfer = vk.vkGetDeviceQueue(dev, 2, 0)
    print("device created: 2x COMPUTE (family 1, REALTIME+HIGH) + 1x TRANSFER (family 2). Family 0 never touched.")
    return inst, gpu, dev, q_compute0, q_compute1, q_transfer

if __name__ == "__main__":
    create_compute_device()
