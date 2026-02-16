#ifndef RENDER_API_HPP
#define RENDER_API_HPP

#include "buffer.hpp"
#include "computeManager.hpp"
#include "computeTask.hpp"
#include "descriptors.hpp"
#include "gpuContext.hpp"
#include "graphicsTask.hpp"
#include "renderDevice.hpp"
#include "renderInstance.hpp"
#include "renderWindow.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	namespace detail {
		inline std::vector<instance::RenderInstance> instances_;
	}

	instance::InitInstanceResult			initNewInstance(const instance::Config& config);
	std::vector<instance::RenderInstance>&  getInstances();

	device::InitDeviceResult				addDevice(const device::Config& config);
	std::vector<device::PhysicalDeviceInfo> enumerateDevices(VkInstance instance);
	VkPhysicalDevice						selectBestDevice(VkInstance instance);


	ComputeTask* getComputeTask(const std::string& name, size_t gpuIndex = 0);
	GraphicsTask* getGraphicsTask(const std::string& name, size_t gpuIndex = 0);
	void removeComputeTask(const std::string& name, size_t gpuIndex = 0);
	void removeGraphicsTask(const std::string& name, size_t gpuIndex = 0);
	std::vector<uint32_t> loadSPIRV(const std::string& filename);


	// pipeline
	ComputeTask* createComputeTask(const std::vector<uint32_t>& spirvCode, const std::string& name = "", size_t gpuIndex = 0);
	GraphicsTask* createGraphicsTask(RenderWindow* window, const std::string& name = "", size_t gpuIndex = 0);

		void clearAllTasks(size_t gpuIndex = 0);

	// create renderDevice
	GPUContext	createContext(device::GPU* gpu);

	// GPU
	device::GPU* getGPU(size_t index = 0);
	size_t getGPUCount();

	struct InitResult {
		bool						 success;
		instance::InitInstanceResult instanceResult;
		device::InitDeviceResult	 deviceResult;
		instance::RenderInstance*	 instance;
		device::GPU*				 gpu;

		operator bool() const { return success; }
	};
}
#endif
