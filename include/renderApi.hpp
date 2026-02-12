/**
 * @file renderApi.hpp
 * @brief Main Render API header file
 * @details Provides the main interface for the render API including instance management
 */

#ifndef RENDER_API_HPP
#define RENDER_API_HPP

#include "renderInstance.hpp"
#include "renderDevice.hpp"

#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	namespace {
		std::vector<instance::RenderInstance> instances_;
	}

	// -- add function
	instance::InitInstanceResult	initNewInstance(const instance::Config& config);
	device::InitDeviceResult		addNewDevice(const device::Config& config);

	// -- getter
	inline std::vector<instance::RenderInstance>& getInstances() { return instances_; }
}

#endif
