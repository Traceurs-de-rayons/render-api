#ifndef RENDER_API_HPP
#define RENDER_API_HPP

#include "buffer.hpp"
#include "renderDevice.hpp"
#include "renderInstance.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	namespace detail {
		inline std::vector<instance::RenderInstance> instances_;
	}

	// vk instance
	instance::InitInstanceResult			initNewInstance(const instance::Config& config);
	std::vector<instance::RenderInstance>&  getInstances();
	// add un clean


	// vk device
	device::InitDeviceResult				addDevice(const device::Config& config);
	std::vector<device::PhysicalDeviceInfo> enumerateDevices(VkInstance instance);
	VkPhysicalDevice						selectBestDevice(VkInstance instance);
	// add un clean

	// Buffer creation helpers
	Buffer									createBuffer(size_t size, BufferType type, BufferUsage usage = BufferUsage::STATIC, size_t gpuIndex = 0);
	Buffer									createVertexBuffer(size_t size, BufferUsage usage = BufferUsage::STATIC, size_t gpuIndex = 0);
	Buffer									createIndexBuffer(size_t size, BufferUsage usage = BufferUsage::STATIC, size_t gpuIndex = 0);
	Buffer									createUniformBuffer(size_t size, BufferUsage usage = BufferUsage::DYNAMIC, size_t gpuIndex = 0);
	Buffer									createStorageBuffer(size_t size, BufferUsage usage = BufferUsage::STATIC, size_t gpuIndex = 0);
	Buffer									createStagingBuffer(size_t size, size_t gpuIndex = 0);

	// GPU
	device::GPU*							getGPU(size_t index = 0);
	size_t									getGPUCount();

	std::vector<uint32_t>					loadSPIRV(const std::string& filename);

	struct InitResult {
		bool						 success;
		instance::InitInstanceResult instanceResult;
		device::InitDeviceResult	 deviceResult;
		instance::RenderInstance*	 instance;
		device::GPU*				 gpu;

		operator bool() const { return success; }
	};

	InitResult quickInit(const std::string& appName = "RenderApp", bool enableValidation = true,
						 const std::vector<const char*>& windowExtensions = {});
	instance::RenderInstance* getInstance();
}
#endif
