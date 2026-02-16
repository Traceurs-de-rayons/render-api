#include "renderApi.hpp"

#include "buffer.hpp"
#include "computeTask.hpp"
#include "graphicsTask.hpp"
#include "renderDevice.hpp"
#include "renderInstance.hpp"

#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
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

InitResult renderApi::quickInit(const std::string& appName, bool enableValidation, const std::vector<const char*>& windowExtensions) {
	InitResult result = {};

	instance::Config instanceConfig;
	if (enableValidation) {
		instanceConfig = instance::Config::DebugDefault(appName);
	} else {
		instanceConfig = instance::Config::ReleaseDefault(appName);
	}

	for (auto ext : windowExtensions) {
		instanceConfig.extensions.push_back(ext);
	}

	result.instanceResult = renderApi::initNewInstance(instanceConfig);
	if (result.instanceResult != instance::INIT_VK_INSTANCESUCCESS) {
		result.success = false;
		return result;
	}

	result.instance = getInstance();
	if (!result.instance) {
		result.success = false;
		return result;
	}

	VkPhysicalDevice physicalDevice = selectBestDevice(result.instance->getInstance());
	if (physicalDevice == VK_NULL_HANDLE) {
		result.deviceResult = device::NO_PHYSICAL_DEVICE_FOUND;
		result.success		= false;
		return result;
	}

	device::Config deviceConfig;
	deviceConfig.renderInstance = result.instance;
	deviceConfig.vkInstance		= result.instance->getInstance();
	deviceConfig.physicalDevice = physicalDevice;

	result.deviceResult = addDevice(deviceConfig);
	if (result.deviceResult != device::INIT_DEVICE_SUCCESS) {
		result.success = false;
		return result;
	}

	const auto& gpus = result.instance->getGPUs();
	if (!gpus.empty()) {
		result.gpu = gpus[0].get();
	}

	result.success = true;
	return result;
}

GPUContext renderApi::createContext(device::GPU* gpu) {
	GPUContext context;
	if (gpu) context.initialize(gpu);
	return context;
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

ComputeTask* renderApi::createComputeTask(const std::vector<uint32_t>& spirvCode, const std::string& name, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	return gpu ? gpu->createComputeTask(spirvCode, name) : nullptr;
}

GraphicsTask* renderApi::createGraphicsTask(RenderWindow* window, const std::string& name, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	return gpu ? gpu->createGraphicsTask(window, name) : nullptr;
}

ComputeTask* renderApi::getComputeTask(const std::string& name, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	return gpu ? gpu->getComputeTask(name) : nullptr;
}

GraphicsTask* renderApi::getGraphicsTask(const std::string& name, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	return gpu ? gpu->getGraphicsTask(name) : nullptr;
}

void renderApi::removeComputeTask(const std::string& name, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (gpu) gpu->removeComputeTask(name);
}

void renderApi::removeGraphicsTask(const std::string& name, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (gpu) gpu->removeGraphicsTask(name);
}

void renderApi::clearAllTasks(size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (gpu)
		gpu->clearAllTasks();
}

std::vector<uint32_t> renderApi::loadSPIRV(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file.is_open()) return {};
	size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
	file.close();
	return buffer;
}
