"""Vulkan compute-only device probe: family 1 (COMPUTE, no GRAPHICS), queue selection, props."""
import vulkan as vk

def main():
    inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="kindi-vk", applicationVersion=1, pEngineName="kindi-vk",
            engineVersion=1, apiVersion=vk.VK_MAKE_VERSION(1, 3, 0))), None)
    gpus = vk.vkEnumeratePhysicalDevices(inst)
    print(f"gpus: {len(gpus)}")
    for gi, gpu in enumerate(gpus):
        props = vk.vkGetPhysicalDeviceProperties(gpu)
        print(f"[{gi}] {props.deviceName} api={props.apiVersion>>22}.{(props.apiVersion>>12)&0x3FF}.{props.apiVersion&0xFFF}")
        qfams = vk.vkGetPhysicalDeviceQueueFamilyProperties(gpu)
        for qi, q in enumerate(qfams):
            flags = int(q.queueFlags)
            names = []
            if flags & vk.VK_QUEUE_GRAPHICS_BIT: names.append("GRAPHICS")
            if flags & vk.VK_QUEUE_COMPUTE_BIT: names.append("COMPUTE")
            if flags & vk.VK_QUEUE_TRANSFER_BIT: names.append("TRANSFER")
            if flags & vk.VK_QUEUE_SPARSE_BINDING_BIT: names.append("SPARSE")
            print(f"    family {qi}: {'|'.join(names)} count={q.queueCount} tbits={q.timestampValidBits}")
        mem = vk.vkGetPhysicalDeviceMemoryProperties(gpu)
        for hi in range(mem.memoryHeapCount):
            h = mem.memoryHeaps[hi]
            print(f"    heap {hi}: {h.size/2**30:.2f} GiB flags={int(h.flags)}")
        # subgroup + float16/int8/dot features
        try:
            f11 = vk.VkPhysicalDeviceVulkan11Properties(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES)
            f12 = vk.VkPhysicalDeviceVulkan12Properties(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES)
            f13 = vk.VkPhysicalDeviceVulkan13Properties(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES)
            feats = vk.VkPhysicalDeviceFeatures2(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
            # chain query via vkGetPhysicalDeviceProperties2 if available
            vk.vkGetPhysicalDeviceProperties2(gpu, vk.VkPhysicalDeviceProperties2(sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, pNext=None))
        except Exception as e:
            print("    (extended props query skipped:", e, ")")

if __name__ == "__main__":
    main()
