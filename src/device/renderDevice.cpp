#include "renderDevice.hpp"
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <vulkan/vulkan_core.h>

using namespace renderApi::device;

InitDeviceResult	addNewDevice(const Config& config) {
	GPU gpu;

	// get queue / device
	gpu.physicalDevice = VK_NULL_HANDLE;
	gpu.device = VK_NULL_HANDLE;
	gpu.graphicsQueue = VK_NULL_HANDLE;
	gpu.computeQueue = VK_NULL_HANDLE;
	gpu.transferQueue = VK_NULL_HANDLE;

	try {
		gpu.finishCode = std::async(std::launch::async, gpuThreadLoop, std::ref(gpu));
	} catch (const std::exception& e) {
		std::cerr << "Failed to initialize thread: " << e.what() << std::endl;
		return THREAD_INIT_FAILED;
	}

	return INIT_DEVICE_SUCCESS;
}
