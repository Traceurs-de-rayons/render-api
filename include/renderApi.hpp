#ifndef RENDER_API_HPP
#define RENDER_API_HPP

#include "renderDevice.hpp"
#include "renderInstance.hpp"
#include "image/image.hpp"
#include "query/queryPool.hpp"
#include "descriptor/descriptorSetManager.hpp"

#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

#define INIT_RENDER_API renderApi::Api _renderApiInstance

namespace renderApi {

	struct InitResult {
		bool						 success;
		instance::InitInstanceResult instanceResult;
		device::InitDeviceResult	 deviceResult;
		instance::RenderInstance*	 instance;
		device::GPU*				 gpu;

		operator bool() const { return success; }
	};

	class Api {
		private:
			void	cleanup();
		public:
			Api() = default;
			~Api() { cleanup(); }

			Api(const Api&)			   = delete;
			Api& operator=(const Api&) = delete;
			Api(Api&&)				   = delete;
			Api& operator=(Api&&)	   = delete;
	};

	namespace {
		inline std::vector<instance::RenderInstance>& getInstancesVector() {
			static std::vector<instance::RenderInstance> instances_;
			return instances_;
		}
	}

	// vk instance
	instance::InitInstanceResult 			initNewInstance(const instance::Config& config);
	std::vector<instance::RenderInstance>&	getInstances();
	instance::RenderInstance*				getInstance(int index);
	instance::RenderInstance*				getInstance(std::string name);

}

#endif
