#include "renderInstance.hpp"

#include "renderDevice.hpp"

#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#endif
#include <vulkan/vulkan_core.h>

using namespace renderApi::instance;
using namespace renderApi::device;

RenderInstance::RenderInstance(const Config& config) : instance_(nullptr), config_(config) {
	VkApplicationInfo	 appInfo{};
	VkInstanceCreateInfo createInfo{};

	appInfo.sType			   = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName   = config_.appName.c_str();
	appInfo.applicationVersion = config_.appVersion;
	appInfo.pEngineName		   = config_.engineName.c_str();
	appInfo.engineVersion	   = config_.engineVersion;
	appInfo.apiVersion		   = config_.apiVersion;

	createInfo.sType			= VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	std::vector<const char*> instanceExtensions = config_.extensions;
	bool					 hasSurface			= false;
	for (const auto& ext : instanceExtensions) {
		if (std::string(ext) == VK_KHR_SURFACE_EXTENSION_NAME) {
			hasSurface = true;
			break;
		}
	}
	if (!hasSurface) {
		instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
	}

	bool hasXlibSurface = false;
	for (const auto& ext : instanceExtensions) {
		if (std::string(ext) == VK_KHR_XLIB_SURFACE_EXTENSION_NAME) {
			hasXlibSurface = true;
			break;
		}
	}
	if (!hasXlibSurface) {
		instanceExtensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
	}

	createInfo.enabledExtensionCount   = static_cast<uint32_t>(instanceExtensions.size());
	createInfo.ppEnabledExtensionNames = instanceExtensions.data();
	createInfo.enabledLayerCount	   = static_cast<uint32_t>(config_.layers.size());
	createInfo.ppEnabledLayerNames	   = config_.layers.data();
	createInfo.flags				   = config_.flags;

	if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) throw std::runtime_error("Failed to create Vulkan instance!");
}

RenderInstance::~RenderInstance() {
	// Les destructeurs des GPUs vont gérer l'arrêt de leurs threads
	gpus_.clear();

	if (instance_) {
		vkDestroyInstance(instance_, nullptr);
		instance_ = nullptr;
	}
}

QueueFamilies RenderInstance::findQueueFamilies(VkPhysicalDevice device) {
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

InitDeviceResult RenderInstance::finishDeviceInitialization(GPU& gpu) {
	if (!gpu.device) return VK_CREATE_DEVICE_FAILED;

	gpu.queueFamilies = findQueueFamilies(gpu.physicalDevice);

	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType			  = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = gpu.queueFamilies.graphicsFamily >= 0 ? gpu.queueFamilies.graphicsFamily : gpu.queueFamilies.computeFamily;
	poolInfo.flags			  = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (vkCreateCommandPool(gpu.device, &poolInfo, nullptr, &gpu.commandPool) != VK_SUCCESS) return VK_CREATE_DEVICE_FAILED;

	return INIT_DEVICE_SUCCESS;
}

InitDeviceResult RenderInstance::addGPU(const device::Config& config) {
	if (!instance_) return RENDER_INSTANCE_NULL;

	auto gpu = std::make_unique<GPU>();

	gpu->instance		= instance_;
	gpu->physicalDevice = config.physicalDevice;
	gpu->name			= config.name;

	if (gpu->physicalDevice == VK_NULL_HANDLE) {
		gpu->physicalDevice = renderApi::device::selectBestPhysicalDevice(instance_);
		if (gpu->physicalDevice == VK_NULL_HANDLE) return NO_PHYSICAL_DEVICE_FOUND;
	}

	auto families	   = findQueueFamilies(gpu->physicalDevice);
	gpu->queueFamilies = families;

	// Get queue family properties to check available queue count
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(gpu->physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(gpu->physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

	// Determine queue counts per family (limited by available queues)
	std::map<int, uint32_t> familyQueueCount;
	std::map<int, uint32_t> familyQueueOffset;
	std::map<int, uint32_t> requestedGraphicsCount;
	std::map<int, uint32_t> requestedComputeCount;
	std::map<int, uint32_t> requestedTransferCount;

	if (families.graphicsFamily >= 0 && config.graphics > 0) {
		requestedGraphicsCount[families.graphicsFamily] = config.graphics;
		familyQueueCount[families.graphicsFamily] += config.graphics;
	}
	if (families.computeFamily >= 0 && config.compute > 0) {
		requestedComputeCount[families.computeFamily] = config.compute;
		familyQueueCount[families.computeFamily] += config.compute;
	}
	if (families.transferFamily >= 0 && config.transfer > 0) {
		requestedTransferCount[families.transferFamily] = config.transfer;
		familyQueueCount[families.transferFamily] += config.transfer;
	}

	// Limit each family to available queues and calculate actual counts
	std::map<int, uint32_t> actualGraphicsCount;
	std::map<int, uint32_t> actualComputeCount;
	std::map<int, uint32_t> actualTransferCount;

	for (auto& [familyIndex, totalRequested] : familyQueueCount) {
		uint32_t maxQueues = queueFamilyProperties[familyIndex].queueCount;
		if (totalRequested > maxQueues) {
			std::cerr << "Warning: Requested " << totalRequested << " queues for family " << familyIndex << " but only " << maxQueues
					  << " available. Limiting to " << maxQueues << std::endl;
		}
		uint32_t actualTotal = std::min(totalRequested, maxQueues);

		// Calculate ratios and distribute actual queues
		uint32_t graphicsReq = requestedGraphicsCount[familyIndex];
		uint32_t computeReq	 = requestedComputeCount[familyIndex];
		uint32_t transferReq = requestedTransferCount[familyIndex];
		uint32_t totalReq	 = graphicsReq + computeReq + transferReq;

		if (totalReq > 0) {
			// Proportional distribution, ensuring at least 1 if requested
			actualGraphicsCount[familyIndex] = graphicsReq > 0 ? std::max(1u, (graphicsReq * actualTotal) / totalReq) : 0;
			actualComputeCount[familyIndex]	 = computeReq > 0 ? std::max(1u, (computeReq * actualTotal) / totalReq) : 0;
			actualTransferCount[familyIndex] = transferReq > 0 ? std::max(1u, (transferReq * actualTotal) / totalReq) : 0;

			// Adjust if we exceeded the total due to rounding
			uint32_t sum = actualGraphicsCount[familyIndex] + actualComputeCount[familyIndex] + actualTransferCount[familyIndex];
			if (sum > actualTotal) {
				// Reduce the largest one
				if (actualGraphicsCount[familyIndex] >= actualComputeCount[familyIndex] &&
					actualGraphicsCount[familyIndex] >= actualTransferCount[familyIndex]) {
					actualGraphicsCount[familyIndex] -= (sum - actualTotal);
				} else if (actualComputeCount[familyIndex] >= actualTransferCount[familyIndex]) {
					actualComputeCount[familyIndex] -= (sum - actualTotal);
				} else {
					actualTransferCount[familyIndex] -= (sum - actualTotal);
				}
			}
		}

		familyQueueCount[familyIndex] = actualTotal;
	}

	// Create queue create infos
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::vector<std::vector<float>>		 queuePriorities;

	for (const auto& [familyIndex, count] : familyQueueCount) {
		familyQueueOffset[familyIndex] = queuePriorities.size();
		queuePriorities.emplace_back(count, 1.0f);

		VkDeviceQueueCreateInfo queueInfo{};
		queueInfo.sType			   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueInfo.queueFamilyIndex = familyIndex;
		queueInfo.queueCount	   = count;
		queueInfo.pQueuePriorities = queuePriorities.back().data();
		queueCreateInfos.push_back(queueInfo);
	}

	// Query available Vulkan 1.2 features
	VkPhysicalDeviceVulkan12Features vulkan12Features{};
	vulkan12Features.sType				 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vulkan12Features.bufferDeviceAddress = VK_TRUE;
	vulkan12Features.descriptorIndexing	 = VK_TRUE;

	VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{};
	meshShaderFeatures.sType	  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
	meshShaderFeatures.meshShader = VK_FALSE;
	meshShaderFeatures.taskShader = VK_FALSE;

	bool	 meshShaderSupported = false;
	uint32_t availableExtCount	 = 0;
	vkEnumerateDeviceExtensionProperties(gpu->physicalDevice, nullptr, &availableExtCount, nullptr);
	std::vector<VkExtensionProperties> availableExts(availableExtCount);
	vkEnumerateDeviceExtensionProperties(gpu->physicalDevice, nullptr, &availableExtCount, availableExts.data());

	std::vector<const char*> deviceExtensions;
	for (const auto& ext : availableExts) {
		if (std::string(ext.extensionName) == VK_KHR_SWAPCHAIN_EXTENSION_NAME) {
			deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		}
		if (std::string(ext.extensionName) == VK_EXT_MESH_SHADER_EXTENSION_NAME) {
			deviceExtensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
			meshShaderSupported = true;
		}
	}

	vulkan12Features.pNext = &meshShaderFeatures;

	VkPhysicalDeviceFeatures  deviceFeatures{};
	VkPhysicalDeviceFeatures2 deviceFeatures2{};
	deviceFeatures2.sType	 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	deviceFeatures2.pNext	 = &vulkan12Features;
	deviceFeatures2.features = deviceFeatures;

	VkDeviceCreateInfo deviceCreateInfo{};
	deviceCreateInfo.sType					 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.pNext					 = &deviceFeatures2;
	deviceCreateInfo.queueCreateInfoCount	 = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos		 = queueCreateInfos.data();
	deviceCreateInfo.pEnabledFeatures		 = nullptr; // Using pNext chain instead
	deviceCreateInfo.enabledExtensionCount	 = static_cast<uint32_t>(deviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();

	VkResult result = vkCreateDevice(gpu->physicalDevice, &deviceCreateInfo, nullptr, &gpu->device);
	if (result != VK_SUCCESS) {
		return VK_CREATE_DEVICE_FAILED;
	}

	gpu->meshShaderSupported = meshShaderSupported && meshShaderFeatures.meshShader;
	if (meshShaderSupported) {
		std::cout << "  Mesh Shader: " << (gpu->meshShaderSupported ? "supported" : "not supported by device") << std::endl;
		if (!gpu->meshShaderSupported && meshShaderFeatures.taskShader) {
			std::cout << "    (Task shader only - no mesh shader)" << std::endl;
		}
	}

	// Retrieve queues
	std::map<int, uint32_t> familyCurrentIndex;

	if (families.graphicsFamily >= 0 && actualGraphicsCount.count(families.graphicsFamily)) {
		uint32_t count = actualGraphicsCount[families.graphicsFamily];
		for (uint32_t i = 0; i < count; i++) {
			VkQueue queue;
			vkGetDeviceQueue(gpu->device, families.graphicsFamily, familyCurrentIndex[families.graphicsFamily]++, &queue);
			gpu->graphicsQueues.push_back(queue);
		}
	}
	if (families.computeFamily >= 0 && actualComputeCount.count(families.computeFamily)) {
		uint32_t count = actualComputeCount[families.computeFamily];
		for (uint32_t i = 0; i < count; i++) {
			VkQueue queue;
			vkGetDeviceQueue(gpu->device, families.computeFamily, familyCurrentIndex[families.computeFamily]++, &queue);
			gpu->computeQueues.push_back(queue);
		}
	}
	if (families.transferFamily >= 0 && actualTransferCount.count(families.transferFamily)) {
		uint32_t count = actualTransferCount[families.transferFamily];
		for (uint32_t i = 0; i < count; i++) {
			VkQueue queue;
			vkGetDeviceQueue(gpu->device, families.transferFamily, familyCurrentIndex[families.transferFamily]++, &queue);
			gpu->transferQueues.push_back(queue);
		}
	}

	auto initResult = finishDeviceInitialization(*gpu);
	if (initResult != INIT_DEVICE_SUCCESS) return initResult;

	std::cout << "GPU created: " << gpu->name << std::endl;
	std::cout << "  Graphics queues: " << gpu->graphicsQueues.size() << " (requested: " << config.graphics << ")" << std::endl;
	std::cout << "  Compute queues: " << gpu->computeQueues.size() << " (requested: " << config.compute << ")" << std::endl;
	std::cout << "  Transfer queues: " << gpu->transferQueues.size() << " (requested: " << config.transfer << ")" << std::endl;

	try {
		gpu->running	= true;
		gpu->finishCode = std::async(std::launch::async, renderApi::device::gpuThreadLoop, std::ref(*gpu));
	} catch (const std::exception& e) {
		std::cerr << "Failed to initialize thread: " << e.what() << std::endl;
		return THREAD_INIT_FAILED;
	}

	gpus_.push_back(std::move(gpu));
	return INIT_DEVICE_SUCCESS;
}
