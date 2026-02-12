/** @file renderDevice.hpp
 * @brief Render device header file
 * @details Defines the RenderDevice class and QueueCount structure for Vulkan device management
 */

#ifndef RENDER_DEVICE_HPP
#define RENDER_DEVICE_HPP

#include <atomic>
#include <cstdint>
#include <future>
#include <vulkan/vulkan_core.h>

namespace renderApi::device {

	enum gpuLoopThreadResult {
		THEARD_LOOP_SUCCESS = 0
	};

	enum InitDeviceResult {
		INIT_DEVICE_SUCCESS = 0,
		EXTENTIONS_NOT_AVAILABLE = 1,
		VK_GET_EXTENTION_FAILED = 2,
		VK_CREATE_DEVICE_FAILED = 3,
		THREAD_INIT_FAILED = 4
	};

	struct Config {
		VkInstance			instance = nullptr;
		VkPhysicalDevice	physicalDevice = nullptr;
		uint32_t			graphics = 0;
		uint32_t			compute = 0;
		uint32_t			transfer = 0;
	};

	struct GPU {
		VkPhysicalDevice	physicalDevice = VK_NULL_HANDLE;
		VkDevice			device = VK_NULL_HANDLE;
		VkQueue				graphicsQueue = VK_NULL_HANDLE;
		VkQueue				computeQueue = VK_NULL_HANDLE;
		VkQueue				transferQueue = VK_NULL_HANDLE;
		std::atomic<bool>	running = false;
		std::future<gpuLoopThreadResult>	finishCode;
	};

	InitDeviceResult	addNewDevice(const device::Config& config);
	gpuLoopThreadResult	gpuThreadLoop(renderApi::device::GPU& gpu);

	// VkPhysicalDevice		getPhysicalDevice(); // auto (the most performant)
	// VkPhysicalDevice		getPhysicalDevice(uint32_t index); // index chosen
	// VkPhysicalDevice		getPhysicalDevice(const std::vector<const char*>& extensions, const std::vector<const char*>& queues); // per property
};

#endif
