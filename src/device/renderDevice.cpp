#include "renderDevice.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
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

bool renderApi::device::queueSupportsPresentation(VkPhysicalDevice physicalDevice, uint32_t familyIndex, VkSurfaceKHR surface) {
	VkBool32 supported = VK_FALSE;
	vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex, surface, &supported);
	return supported == VK_TRUE;
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

	if (devices.empty()) return VK_NULL_HANDLE;

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

	if (vkCreateCommandPool(gpu.device, &poolInfo, nullptr, &gpu.commandPool) != VK_SUCCESS) return VK_CREATE_DEVICE_FAILED;

	return INIT_DEVICE_SUCCESS;
}

GPU::~GPU() {
	running = false;
	
	if (finishCode.valid()) {
		// Attendre maximum 5 secondes pour que le thread se termine (il vérifie running toutes les 1ms)
		auto status = finishCode.wait_for(std::chrono::seconds(5));
		if (status == std::future_status::timeout) {
			std::cerr << "Warning: GPU thread '" << name << "' did not finish in time, forcing cleanup" << std::endl;
		}
	}
	cleanup();
}

void GPU::cleanup() {
	if (device) {
		vkDeviceWaitIdle(device);

		if (commandPool) {
			vkDestroyCommandPool(device, commandPool, nullptr);
			commandPool = VK_NULL_HANDLE;
		}

		vkDestroyDevice(device, nullptr);
		device = VK_NULL_HANDLE;
	}

	graphicsQueues.clear();
	computeQueues.clear();
	transferQueues.clear();
	presentQueues.clear();
	physicalDevice = VK_NULL_HANDLE;
	instance	   = VK_NULL_HANDLE;
}

VkCommandBuffer GPU::beginOneTimeCommands() {
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool		 = commandPool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}

void GPU::endOneTimeCommands(VkCommandBuffer commandBuffer) {
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers	  = &commandBuffer;

	VkQueue queue = !graphicsQueues.empty()	  ? graphicsQueues[0]
					: !computeQueues.empty()  ? computeQueues[0]
					: !transferQueues.empty() ? transferQueues[0]
											  : nullptr;
	if (queue) {
		std::lock_guard<std::mutex> lock(queueMutex);
		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(queue);
	}

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

VkQueue GPU::getPresentQueue() {
	if (!presentQueues.empty()) {
		return presentQueues[0];
	}
	if (!graphicsQueues.empty()) {
		return graphicsQueues[0];
	}
	return VK_NULL_HANDLE;
}
