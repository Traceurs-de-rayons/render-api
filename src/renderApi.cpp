#include "renderApi.hpp"

#include "renderDevice.hpp"
#include "renderInstance.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_core.h>

using namespace renderApi;
using namespace renderApi::instance;

InitInstanceResult renderApi::initNewInstance(const Config& config) {
	try {
		for (const char* extention : config.extensions)
			if (!isInstanceExtensionAvailable(extention)) return EXTENTIONS_NOT_AVAILABLE;
	} catch (const std::exception& e) {
		std::cerr << "Error initializing render instance: " << e.what() << std::endl;
		return VK_GET_EXTENTION_FAILED;
	}

	try {
		detail::instances_.emplace_back(config);
	} catch (const std::exception& e) {
		std::cerr << "Error initializing render instance: " << e.what() << std::endl;
		return VK_CREATE_INSTANCE_FAILED;
	}
	return INIT_VK_INSTANCESUCCESS;
}

std::vector<instance::RenderInstance>& renderApi::getInstances() {
	return detail::instances_;
}

instance::RenderInstance* renderApi::getInstance() {
	return detail::instances_.empty() ? nullptr : &detail::instances_[0];
}

device::InitDeviceResult renderApi::addDevice(const device::Config& config) {
	return device::addNewDevice(config);
}

std::vector<device::PhysicalDeviceInfo> renderApi::enumerateDevices(VkInstance instance) {
	return device::enumeratePhysicalDevices(instance);
}

VkPhysicalDevice renderApi::selectBestDevice(VkInstance instance) {
	return device::selectBestPhysicalDevice(instance);
}

device::GPU* renderApi::getGPU(size_t index) {
	auto* inst = getInstance();
	if (!inst)
		return nullptr;
	const auto& gpus = inst->getGPUs();
	return (index < gpus.size()) ? gpus[index].get() : nullptr;
}

size_t renderApi::getGPUCount() {
	auto* inst = getInstance();
	return inst ? inst->getGPUs().size() : 0;
}
