#include "renderDevice.hpp"

#include "renderInstance.hpp"
#include "../compute/computeTask.hpp"
#include "../graphics/graphicsTask.hpp"
#include "../context/gpuContext.hpp"
#include "../window/renderWindow.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
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

	// Initialize GPUContext
	gpu.context = std::make_unique<renderApi::GPUContext>();
	if (!gpu.context->initialize(&gpu)) {
		std::cerr << "[GPU] Failed to initialize GPUContext!" << std::endl;
		return VK_CREATE_DEVICE_FAILED;
	}

	return INIT_DEVICE_SUCCESS;
}

GPU::~GPU() {
	cleanup();
}

void GPU::cleanup() {
	// Clear all tasks first
	clearAllTasks();

	// Shutdown context
	if (context) {
		context->shutdown();
		context.reset();
	}

	if (device != VK_NULL_HANDLE) {
		if (descriptorPool != VK_NULL_HANDLE) {
			vkDestroyDescriptorPool(device, descriptorPool, nullptr);
			descriptorPool = VK_NULL_HANDLE;
		}
		if (commandPool != VK_NULL_HANDLE) {
			vkDestroyCommandPool(device, commandPool, nullptr);
			commandPool = VK_NULL_HANDLE;
		}
		vkDestroyDevice(device, nullptr);
		device = VK_NULL_HANDLE;
	}
}

renderApi::ComputeTask* GPU::createComputeTask(const std::vector<uint32_t>& spirvCode, const std::string& name) {
	if (device == VK_NULL_HANDLE || !context) {
		std::cerr << "[GPU] Device not initialized!" << std::endl;
		return nullptr;
	}

	auto task = std::make_unique<renderApi::ComputeTask>();
	
	if (!task->create(*context, name)) {
		std::cerr << "[GPU] Failed to create compute task!" << std::endl;
		return nullptr;
	}

	task->setShader(spirvCode, name);

	std::lock_guard<std::mutex> lock(tasksMutex);
	auto* ptr = task.get();
	computeTasks.push_back(std::move(task));

	return ptr;
}

renderApi::GraphicsTask* GPU::createGraphicsTask(renderApi::RenderWindow* window, const std::string& name) {
	if (device == VK_NULL_HANDLE || !context) {
		std::cerr << "[GPU] Device not initialized!" << std::endl;
		return nullptr;
	}

	if (!window) {
		std::cerr << "[GPU] RenderWindow is null!" << std::endl;
		return nullptr;
	}

	auto task = std::make_unique<renderApi::GraphicsTask>();
	
	if (!task->create(*context, *window, name)) {
		std::cerr << "[GPU] Failed to create graphics task!" << std::endl;
		return nullptr;
	}

	std::lock_guard<std::mutex> lock(tasksMutex);
	auto* ptr = task.get();
	graphicsTasks.push_back(std::move(task));

	return ptr;
}

void GPU::removeComputeTask(const std::string& name) {
	std::lock_guard<std::mutex> lock(tasksMutex);

	auto it = std::remove_if(computeTasks.begin(), computeTasks.end(),
		[&name](const std::unique_ptr<renderApi::ComputeTask>& task) {
			return task->getName() == name;
		});

	computeTasks.erase(it, computeTasks.end());
}

void GPU::removeGraphicsTask(const std::string& name) {
	std::lock_guard<std::mutex> lock(tasksMutex);

	auto it = std::remove_if(graphicsTasks.begin(), graphicsTasks.end(),
		[&name](const std::unique_ptr<renderApi::GraphicsTask>& task) {
			return task->getName() == name;
		});

	graphicsTasks.erase(it, graphicsTasks.end());
}

renderApi::ComputeTask* GPU::getComputeTask(const std::string& name) {
	std::lock_guard<std::mutex> lock(tasksMutex);

	for (auto& task : computeTasks) {
		if (task->getName() == name) {
			return task.get();
		}
	}

	return nullptr;
}

renderApi::GraphicsTask* GPU::getGraphicsTask(const std::string& name) {
	std::lock_guard<std::mutex> lock(tasksMutex);

	for (auto& task : graphicsTasks) {
		if (task->getName() == name) {
			return task.get();
		}
	}

	return nullptr;
}

void GPU::executeAllComputeTasks() {
	// This is now handled by the GPU thread loop
	// Users can also manually execute specific tasks
}

void GPU::clearAllTasks() {
	std::lock_guard<std::mutex> lock(tasksMutex);
	
	computeTasks.clear();
	graphicsTasks.clear();
}

InitDeviceResult renderApi::device::addNewDevice(const Config& config) {
	if (!config.renderInstance || !config.vkInstance) return !config.renderInstance ? RENDER_INSTANCE_NULL : VK_INSTANCE_NULL;

	auto gpu = std::make_unique<GPU>();

	gpu->instance		= config.vkInstance;
	gpu->physicalDevice = config.physicalDevice;
	if (gpu->physicalDevice == VK_NULL_HANDLE) {
		gpu->physicalDevice = selectBestPhysicalDevice(config.vkInstance);
		if (gpu->physicalDevice == VK_NULL_HANDLE)
			return NO_PHYSICAL_DEVICE_FOUND;
	}

	auto families	   = findQueueFamilies(gpu->physicalDevice);
	gpu->queueFamilies = families;

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::vector<int>					 uniqueQueueFamilies;

	if (families.graphicsFamily >= 0)
		uniqueQueueFamilies.push_back(families.graphicsFamily);
	if (families.computeFamily >= 0 &&
		std::find(uniqueQueueFamilies.begin(), uniqueQueueFamilies.end(), families.computeFamily) == uniqueQueueFamilies.end())
		uniqueQueueFamilies.push_back(families.computeFamily);
	if (families.transferFamily >= 0 &&
		std::find(uniqueQueueFamilies.begin(), uniqueQueueFamilies.end(), families.transferFamily) == uniqueQueueFamilies.end())
		uniqueQueueFamilies.push_back(families.transferFamily);

	float queuePriority = 1.0f;
	for (int queueFamily : uniqueQueueFamilies) {
		VkDeviceQueueCreateInfo queueInfo{};
		queueInfo.sType			   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueInfo.queueFamilyIndex = queueFamily;
		queueInfo.queueCount	   = 1;
		queueInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueInfo);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};

	VkDeviceCreateInfo deviceCreateInfo{};
	deviceCreateInfo.sType				  = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos	  = queueCreateInfos.data();
	deviceCreateInfo.pEnabledFeatures	  = &deviceFeatures;

	VkResult result = vkCreateDevice(gpu->physicalDevice, &deviceCreateInfo, nullptr, &gpu->device);
	if (result != VK_SUCCESS) {
		return VK_CREATE_DEVICE_FAILED;
	}
	if (families.graphicsFamily >= 0)
		vkGetDeviceQueue(gpu->device, families.graphicsFamily, 0, &gpu->graphicsQueue);
	if (families.computeFamily >= 0)
		vkGetDeviceQueue(gpu->device, families.computeFamily, 0, &gpu->computeQueue);
	if (families.transferFamily >= 0)
		vkGetDeviceQueue(gpu->device, families.transferFamily, 0, &gpu->transferQueue);

	auto initResult = finishDeviceInitialization(*gpu);
	if (initResult != INIT_DEVICE_SUCCESS)
		return initResult;

	try {
		gpu->finishCode = std::async(std::launch::async, gpuThreadLoop, std::ref(*gpu));
	} catch (const std::exception& e) {
		std::cerr << "Failed to initialize thread: " << e.what() << std::endl;
		return THREAD_INIT_FAILED;
	}

	config.renderInstance->addGPU(std::move(gpu));
	return INIT_DEVICE_SUCCESS;
}
