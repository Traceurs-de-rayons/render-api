#ifndef RENDER_INSTANCE_HPP
#define RENDER_INSTANCE_HPP

#include "../device/renderDevice.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <sys/types.h>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi::instance {

	enum InitInstanceResult { INIT_VK_INSTANCESUCCESS = 0, EXTENTIONS_NOT_AVAILABLE = 1, VK_GET_EXTENTION_FAILED = 2, VK_CREATE_INSTANCE_FAILED = 3 };

	struct Config {
		std::string appName		  = "Default";
		uint32_t	appVersion	  = VK_MAKE_VERSION(0, 0, 0);
		std::string engineName	  = "Default";
		uint32_t	engineVersion = VK_MAKE_VERSION(0, 0, 0);
		uint32_t	apiVersion	  = VK_API_VERSION_1_3;

		std::vector<const char*> extensions;
		std::vector<const char*> layers;

		VkInstanceCreateFlags flags = 0;

		static Config DebugDefault(std::string appName) {
			Config c;
			c.appName = appName + "Debug";
			c.layers  = {"VK_LAYER_KHRONOS_validation"};
			return (c);
		}

		static Config ReleaseDefault(std::string appName) {
			Config c;
			c.appName = appName;
			return (c);
		}
	};

	class RenderInstance {
	  private:
		VkInstance								  instance_;
		Config									  config_;
		std::vector<std::unique_ptr<device::GPU>> gpus_;

	  public:
		RenderInstance(const Config& config);
		~RenderInstance();

		RenderInstance(const RenderInstance&)			 = delete;
		RenderInstance& operator=(const RenderInstance&) = delete;

		RenderInstance(RenderInstance&&)			= default;
		RenderInstance& operator=(RenderInstance&&) = default;

		bool addGPU(std::unique_ptr<device::GPU> gpu);

		VkInstance										 getInstance() const { return instance_; }
		const Config&									 getConfig() const { return config_; }
		const std::vector<std::unique_ptr<device::GPU>>& getGPUs() const { return gpus_; }
	};

	bool isInstanceExtensionAvailable(const char* extensionName);
};

#endif
