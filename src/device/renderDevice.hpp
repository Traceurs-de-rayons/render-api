#ifndef RENDER_DEVICE_HPP
#define RENDER_DEVICE_HPP

#include "gpuTask.hpp"
#include "utils.hpp"

#include <atomic>
#include <cstdint>
#include <future>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi::instance {
	class RenderInstance;
}

namespace renderApi::device {

	enum gpuLoopThreadResult { THEARD_LOOP_SUCCESS = 0 };

	enum InitDeviceResult : int {
		INIT_DEVICE_SUCCESS		 = 0,
		EXTENTIONS_NOT_AVAILABLE = 1,
		VK_GET_EXTENTION_FAILED	 = 2,
		VK_CREATE_DEVICE_FAILED	 = 3,
		THREAD_INIT_FAILED		 = 4,
		VK_INSTANCE_NULL		 = 5,
		RENDER_INSTANCE_NULL	 = 6,
		NO_PHYSICAL_DEVICE_FOUND = 7
	};

	struct Config {
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		uint32_t		 graphics		= 0;
		uint32_t		 compute		= 0;
		uint32_t		 transfer		= 0;
		std::string		 name			= generateRandomString();
	};

	struct PhysicalDeviceInfo {
		VkPhysicalDevice		   device;
		VkPhysicalDeviceProperties properties;
		VkPhysicalDeviceFeatures   features;
		std::string				   name;
		uint32_t				   memoryMB;
		bool					   discreteGPU;
	};

	struct QueueFamilies {
		int graphicsFamily = -1;
		int computeFamily  = -1;
		int transferFamily = -1;
		int presentFamily  = -1;
	};

	struct GPU {
		VkInstance								  instance		 = VK_NULL_HANDLE;
		VkPhysicalDevice						  physicalDevice = VK_NULL_HANDLE;
		VkDevice								  device		 = VK_NULL_HANDLE;
		std::vector<VkQueue>					  graphicsQueues;
		std::vector<VkQueue>					  computeQueues;
		std::vector<VkQueue>					  transferQueues;
		QueueFamilies							  queueFamilies;
		VkCommandPool							  commandPool = VK_NULL_HANDLE;
		std::atomic<bool>						  running	  = false;
		std::future<gpuLoopThreadResult>		  finishCode;
		std::vector<renderApi::gpuTask::GpuTask*> GpuTasks;
		std::string								  name;

		~GPU();
		void cleanup();
		VkCommandBuffer beginOneTimeCommands();
		void endOneTimeCommands(VkCommandBuffer commandBuffer);
	};

	gpuLoopThreadResult				gpuThreadLoop(renderApi::device::GPU& gpu);
	std::vector<PhysicalDeviceInfo> enumeratePhysicalDevices(VkInstance instance);
	VkPhysicalDevice				selectBestPhysicalDevice(VkInstance instance);
	InitDeviceResult				finishDeviceInitialization(GPU& gpu);
	QueueFamilies					findQueueFamilies(VkPhysicalDevice device);
	uint32_t						findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
}; // namespace renderApi::device

#endif
