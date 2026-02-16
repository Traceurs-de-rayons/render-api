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

bool renderApi::addShader(const std::string& taskName, ShaderStage stage, 
						   const std::vector<uint32_t>& spirvCode,
						   const std::string& name,
						   const std::string& entryPoint,
						   size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->addShader(stage, spirvCode, name, entryPoint);
	return true;
}

bool renderApi::removeShader(const std::string& taskName, ShaderStage stage, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->removeShader(stage);
	return true;
}

bool renderApi::updateShader(const std::string& taskName, ShaderStage stage,
							  const std::vector<uint32_t>& spirvCode, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->updateShader(stage, spirvCode);
	return true;
}

bool renderApi::hasShader(const std::string& taskName, ShaderStage stage, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	return task ? task->hasShader(stage) : false;
}

bool renderApi::enableShader(const std::string& taskName, ShaderStage stage, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->enableShader(stage);
	return true;
}

bool renderApi::disableShader(const std::string& taskName, ShaderStage stage, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->disableShader(stage);
	return true;
}

bool renderApi::isShaderEnabled(const std::string& taskName, ShaderStage stage, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	return task ? task->isShaderEnabled(stage) : false;
}

void renderApi::clearShaders(const std::string& taskName, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (task) task->clearShaders();
}

bool renderApi::bindComputeBuffer(const std::string& taskName, uint32_t binding, Buffer& buffer, size_t gpuIndex) {
	auto* task = getComputeTask(taskName, gpuIndex);
	if (!task) return false;
	task->bindBuffer(binding, buffer);
	return true;
}

bool renderApi::setComputeDispatchSize(const std::string& taskName, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ, size_t gpuIndex) {
	auto* task = getComputeTask(taskName, gpuIndex);
	if (!task) return false;
	task->setDispatchSize(groupsX, groupsY, groupsZ);
	return true;
}

bool renderApi::bindVertexBuffer(const std::string& taskName, uint32_t binding, Buffer& buffer, uint32_t stride, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->bindVertexBuffer(binding, buffer, stride);
	return true;
}

bool renderApi::bindIndexBuffer(const std::string& taskName, Buffer& buffer, VkIndexType indexType, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->bindIndexBuffer(buffer, indexType);
	return true;
}

bool renderApi::bindUniformBuffer(const std::string& taskName, uint32_t binding, Buffer& buffer, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->bindUniformBuffer(binding, buffer);
	return true;
}

bool renderApi::setViewport(const std::string& taskName, float x, float y, float width, float height, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->setViewport(x, y, width, height);
	return true;
}

bool renderApi::setScissor(const std::string& taskName, int32_t x, int32_t y, uint32_t width, uint32_t height, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->setScissor(x, y, width, height);
	return true;
}

bool renderApi::resetViewport(const std::string& taskName, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->resetViewport();
	return true;
}

bool renderApi::resetScissor(const std::string& taskName, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->resetScissor();
	return true;
}

bool renderApi::buildComputeTask(const std::string& taskName, size_t gpuIndex) {
	auto* task = getComputeTask(taskName, gpuIndex);
	return task ? task->build() : false;
}

bool renderApi::rebuildComputeTask(const std::string& taskName, size_t gpuIndex) {
	auto* task = getComputeTask(taskName, gpuIndex);
	return task ? task->rebuild() : false;
}

bool renderApi::setComputeTaskEnabled(const std::string& taskName, bool enabled, size_t gpuIndex) {
	auto* task = getComputeTask(taskName, gpuIndex);
	if (!task) return false;
	task->setEnabled(enabled);
	return true;
}

bool renderApi::isComputeTaskEnabled(const std::string& taskName, size_t gpuIndex) {
	auto* task = getComputeTask(taskName, gpuIndex);
	return task ? task->isEnabled() : false;
}

bool renderApi::isComputeTaskBuilt(const std::string& taskName, size_t gpuIndex) {
	auto* task = getComputeTask(taskName, gpuIndex);
	return task ? task->isBuilt() : false;
}

bool renderApi::isComputeTaskValid(const std::string& taskName, size_t gpuIndex) {
	auto* task = getComputeTask(taskName, gpuIndex);
	return task ? task->isValid() : false;
}

bool renderApi::buildGraphicsTask(const std::string& taskName, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	return task ? task->build() : false;
}

bool renderApi::rebuildGraphicsTask(const std::string& taskName, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	return task ? task->rebuild() : false;
}

bool renderApi::setGraphicsTaskEnabled(const std::string& taskName, bool enabled, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	if (!task) return false;
	task->setEnabled(enabled);
	return true;
}

bool renderApi::isGraphicsTaskEnabled(const std::string& taskName, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	return task ? task->isEnabled() : false;
}

bool renderApi::isGraphicsTaskBuilt(const std::string& taskName, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	return task ? task->isBuilt() : false;
}

bool renderApi::isGraphicsTaskValid(const std::string& taskName, size_t gpuIndex) {
	auto* task = getGraphicsTask(taskName, gpuIndex);
	return task ? task->isValid() : false;
}

Buffer renderApi::createBuffer(size_t size, BufferType type, BufferUsage usage, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (!gpu || !gpu->context) return Buffer();
	return gpu->context->createBuffer(size, type, usage);
}

Buffer renderApi::createVertexBuffer(size_t size, BufferUsage usage, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (!gpu || !gpu->context) return Buffer();
	return gpu->context->createVertexBuffer(size);
}

Buffer renderApi::createIndexBuffer(size_t size, BufferUsage usage, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (!gpu || !gpu->context) return Buffer();
	return gpu->context->createIndexBuffer(size);
}

Buffer renderApi::createUniformBuffer(size_t size, BufferUsage usage, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (!gpu || !gpu->context) return Buffer();
	return gpu->context->createUniformBuffer(size);
}

Buffer renderApi::createStorageBuffer(size_t size, BufferUsage usage, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (!gpu || !gpu->context) return Buffer();
	return gpu->context->createStorageBuffer(size, usage);
}

Buffer renderApi::createStagingBuffer(size_t size, size_t gpuIndex) {
	auto* gpu = getGPU(gpuIndex);
	if (!gpu || !gpu->context) return Buffer();
	return gpu->context->createStagingBuffer(size);
}
