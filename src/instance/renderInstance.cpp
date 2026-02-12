#include "renderInstance.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vulkan/vulkan_core.h>

using namespace renderApi::instance;

RenderInstance::RenderInstance(const Config& config) : instance_(nullptr), config_(config) {
	VkApplicationInfo		appInfo{};
	VkInstanceCreateInfo	createInfo{};

	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = config_.appName.c_str();
	appInfo.applicationVersion = config_.appVersion;
	appInfo.pEngineName = config_.engineName.c_str();
	appInfo.engineVersion = config_.engineVersion;
	appInfo.apiVersion = config_.apiVersion;

	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = static_cast<uint32_t>(config_.extensions.size());
	createInfo.ppEnabledExtensionNames = config_.extensions.data();
	createInfo.enabledLayerCount = static_cast<uint32_t>(config_.layers.size());
	createInfo.ppEnabledLayerNames = config_.layers.data();
	createInfo.flags = config_.flags;

	if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS)
		throw std::runtime_error("Failed to create Vulkan instance!");
}

RenderInstance::~RenderInstance()
{
	if (instance_)
	{
		vkDestroyInstance(instance_, nullptr);
	}
}
