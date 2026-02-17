#include "renderDevice.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_core.h>

using namespace renderApi::device;

QueueFamilies renderApi::device::findQueueFamilies(VkPhysicalDevice device) {
	QueueFamilies families;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	for (uint32_t i = 0; i < queueFamilyCount; i++) {
		if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && families.graphicsFamily < 0) families.graphicsFamily = i;
		if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT && families.computeFamily < 0) families.computeFamily = i;
		if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT && !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
			families.transferFamily < 0)
			families.transferFamily = i;
	}

	if (families.transferFamily < 0) families.transferFamily = families.graphicsFamily;

	return families;
}

uint32_t renderApi::device::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("Failed to find suitable memory type!");
}

std::vector<PhysicalDeviceInfo> renderApi::device::enumeratePhysicalDevices(VkInstance instance) {
	std::vector<PhysicalDeviceInfo> devices;

	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	if (deviceCount == 0) return devices;

	std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

	for (auto& device : physicalDevices) {
		PhysicalDeviceInfo info;
		info.device = device;
		vkGetPhysicalDeviceProperties(device, &info.properties);
		vkGetPhysicalDeviceFeatures(device, &info.features);
		info.name		 = info.properties.deviceName;
		info.discreteGPU = (info.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(device, &memProperties);
		VkDeviceSize totalMemory = 0;
		for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
			if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
				totalMemory += memProperties.memoryHeaps[i].size;
			}
		}
		info.memoryMB = static_cast<uint32_t>(totalMemory / (1024 * 1024));

		devices.push_back(info);
	}

	return devices;
}

VkPhysicalDevice renderApi::device::selectBestPhysicalDevice(VkInstance instance) {
	auto devices = enumeratePhysicalDevices(instance);

	if (devices.empty())
		return VK_NULL_HANDLE;

	std::sort(devices.begin(), devices.end(), [](const PhysicalDeviceInfo& a, const PhysicalDeviceInfo& b) {
		if (a.discreteGPU != b.discreteGPU) return a.discreteGPU > b.discreteGPU;
		return a.memoryMB > b.memoryMB;
	});
	return devices[0].device;
}

InitDeviceResult renderApi::device::finishDeviceInitialization(GPU& gpu) {
	if (!gpu.device) return VK_CREATE_DEVICE_FAILED;

	gpu.queueFamilies = findQueueFamilies(gpu.physicalDevice);

	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType			  = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = gpu.queueFamilies.graphicsFamily >= 0 ? gpu.queueFamilies.graphicsFamily : gpu.queueFamilies.computeFamily;
	poolInfo.flags			  = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (vkCreateCommandPool(gpu.device, &poolInfo, nullptr, &gpu.commandPool) != VK_SUCCESS)
		return VK_CREATE_DEVICE_FAILED;

	std::vector<VkDescriptorPoolSize> poolSizes = {
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100}};

	VkDescriptorPoolCreateInfo descriptorPoolInfo{};
	descriptorPoolInfo.sType		 = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	descriptorPoolInfo.pPoolSizes	 = poolSizes.data();
	descriptorPoolInfo.maxSets		 = 100;
	descriptorPoolInfo.flags		 = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

	if (vkCreateDescriptorPool(gpu.device, &descriptorPoolInfo, nullptr, &gpu.descriptorPool) != VK_SUCCESS) {
		return VK_CREATE_DEVICE_FAILED;
	}

	return INIT_DEVICE_SUCCESS;
}

GPU::~GPU() {
	cleanup();
}

void GPU::cleanup() {
}
