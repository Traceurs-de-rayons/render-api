#ifndef RENDER_DEVICE_HPP
# define RENDER_DEVICE_HPP

# include <atomic>
# include <cstdint>
# include <future>
# include <memory>
# include <mutex>
# include <string>
# include <vector>
# include <vulkan/vulkan_core.h>

namespace renderApi::instance {
	class RenderInstance;
}

namespace renderApi {
	class ComputeTask;
	class GraphicsTask;
	class GPUContext;
	class RenderWindow;
}

namespace renderApi::device {

	enum gpuLoopThreadResult { THEARD_LOOP_SUCCESS = 0 };

	enum InitDeviceResult {
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
		instance::RenderInstance* renderInstance = nullptr;
		VkInstance				  vkInstance	 = nullptr;
		VkPhysicalDevice		  physicalDevice = VK_NULL_HANDLE;
		uint32_t				  graphics		 = 0;
		uint32_t				  compute		 = 0;
		uint32_t				  transfer		 = 0;
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
		VkInstance						 instance		= VK_NULL_HANDLE;
		VkPhysicalDevice				 physicalDevice = VK_NULL_HANDLE;
		VkDevice						 device			= VK_NULL_HANDLE;
		VkQueue							 graphicsQueue	= VK_NULL_HANDLE;
		VkQueue							 computeQueue	= VK_NULL_HANDLE;
		VkQueue							 transferQueue	= VK_NULL_HANDLE;
		QueueFamilies					 queueFamilies;
		VkCommandPool					 commandPool	= VK_NULL_HANDLE;
		VkDescriptorPool				 descriptorPool = VK_NULL_HANDLE;
		std::atomic<bool>				 running		= false;
		std::future<gpuLoopThreadResult> finishCode;

		// Context wrapper
		std::unique_ptr<renderApi::GPUContext> context;

		// Task management
		std::vector<std::unique_ptr<renderApi::ComputeTask>>  computeTasks;
		std::vector<std::unique_ptr<renderApi::GraphicsTask>> graphicsTasks;
		std::mutex											   tasksMutex;

		// Task management methods
		renderApi::ComputeTask*	 createComputeTask(const std::vector<uint32_t>& spirvCode, const std::string& name = "");
		renderApi::GraphicsTask* createGraphicsTask(renderApi::RenderWindow* window, const std::string& name = "");
		void					 removeComputeTask(const std::string& name);
		void					 removeGraphicsTask(const std::string& name);
		renderApi::ComputeTask*	 getComputeTask(const std::string& name);
		renderApi::GraphicsTask* getGraphicsTask(const std::string& name);
		void					 executeAllComputeTasks();
		void					 clearAllTasks();

		~GPU();
		void cleanup();
	};

	InitDeviceResult	addNewDevice(const device::Config& config);
	gpuLoopThreadResult gpuThreadLoop(renderApi::device::GPU& gpu);
	std::vector<PhysicalDeviceInfo> enumeratePhysicalDevices(VkInstance instance);
	VkPhysicalDevice				selectBestPhysicalDevice(VkInstance instance);
	InitDeviceResult				finishDeviceInitialization(GPU& gpu);
	QueueFamilies findQueueFamilies(VkPhysicalDevice device);
	uint32_t	  findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

#endif
