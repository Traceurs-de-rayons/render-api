#include "renderInstance.hpp"
#include "renderDevice.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
	for (auto& gpu : gpus_)
	{
		if (gpu && gpu->running)
		{
			gpu->running = false;
			if (gpu->finishCode.valid())
				gpu->finishCode.wait();
		}
	}
	gpus_.clear();
	if (instance_)
	{
		vkDestroyInstance(instance_, nullptr);
		instance_ = nullptr;
	}
}

bool RenderInstance::addGPU(std::unique_ptr<device::GPU> gpu)
{
	gpus_.push_back(std::move(gpu));
	return true;
}
