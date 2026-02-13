/**
 * @file renderApi.hpp
 * @brief Main Render API header file
 * @details Single include for all render API functionality
 */

#ifndef RENDER_API_HPP
#define RENDER_API_HPP

// Core components - adjust paths based on your include setup
// If including from project root: #include "src/buffer/buffer.hpp"
// If including from include directory:
#include "../src/buffer/buffer.hpp"
#include "../src/context/gpuContext.hpp"
#include "../src/descriptors/descriptors.hpp"
#include "../src/device/renderDevice.hpp"
#include "../src/instance/renderInstance.hpp"
#include "../src/pipeline/pipeline.hpp"

#include <memory>
#include <vector>

namespace renderApi {

	// Global instances storage (internal use)
	namespace detail {
		inline std::vector<instance::RenderInstance> instances_;
	}

	// ============================================================================
	// Instance Management
	// ============================================================================

	/**
	 * @brief Initialize a new render instance
	 * @param config Instance configuration
	 * @return Result code
	 */
	inline instance::InitInstanceResult init(const instance::Config& config) {
		try {
			for (const char* extension : config.extensions) {
				if (!instance::isInstanceExtensionAvailable(extension)) return instance::EXTENTIONS_NOT_AVAILABLE;
			}
		} catch (...) {
			return instance::VK_GET_EXTENTION_FAILED;
		}

		try {
			detail::instances_.emplace_back(config);
		} catch (...) {
			return instance::VK_CREATE_INSTANCE_FAILED;
		}
		return instance::INIT_VK_INSTANCESUCCESS;
	}

	/**
	 * @brief Get all instances
	 * @return Reference to instances vector
	 */
	inline std::vector<instance::RenderInstance>& getInstances() { return detail::instances_; }

	/**
	 * @brief Get the first instance (convenience)
	 * @return Pointer to first instance, or nullptr if none
	 */
	inline instance::RenderInstance* getInstance() { return detail::instances_.empty() ? nullptr : &detail::instances_[0]; }

	// ============================================================================
	// Device Management
	// ============================================================================

	/**
	 * @brief Add a new device to an instance
	 * @param config Device configuration
	 * @return Result code
	 */
	inline device::InitDeviceResult addDevice(const device::Config& config) { return device::addNewDevice(config); }

	/**
	 * @brief Enumerate all physical devices available on an instance
	 * @param instance Vulkan instance
	 * @return Vector of device information
	 */
	inline std::vector<device::PhysicalDeviceInfo> enumerateDevices(VkInstance instance) { return device::enumeratePhysicalDevices(instance); }

	/**
	 * @brief Automatically select the best physical device
	 * @param instance Vulkan instance
	 * @return Best device handle, or VK_NULL_HANDLE if none found
	 */
	inline VkPhysicalDevice selectBestDevice(VkInstance instance) { return device::selectBestPhysicalDevice(instance); }

	// ============================================================================
	// Quick Initialization (One-liner setup)
	// ============================================================================

	/**
	 * @struct InitResult
	 * @brief Result of quick initialization
	 */
	struct InitResult {
		bool						 success;
		instance::InitInstanceResult instanceResult;
		device::InitDeviceResult	 deviceResult;
		instance::RenderInstance*	 instance;
		device::GPU*				 gpu;

		operator bool() const { return success; }
	};

	/**
	 * @brief Quick initialization - creates instance, device, and context
	 * @param appName Application name
	 * @param enableValidation Enable validation layers
	 * @param windowExtensions Additional instance extensions (e.g., for windowing)
	 * @return Initialization result with pointers to created objects
	 */
	inline InitResult
	quickInit(const std::string& appName = "RenderApp", bool enableValidation = true, const std::vector<const char*>& windowExtensions = {}) {
		InitResult result = {};

		// Create instance config
		instance::Config instanceConfig;
		if (enableValidation) {
			instanceConfig = instance::Config::DebugDefault(appName);
		} else {
			instanceConfig = instance::Config::ReleaseDefault(appName);
		}

		// Add window extensions
		for (auto ext : windowExtensions) {
			instanceConfig.extensions.push_back(ext);
		}

		// Create instance
		result.instanceResult = init(instanceConfig);
		if (result.instanceResult != instance::INIT_VK_INSTANCESUCCESS) {
			result.success = false;
			return result;
		}

		result.instance = getInstance();
		if (!result.instance) {
			result.success = false;
			return result;
		}

		// Select best device
		VkPhysicalDevice physicalDevice = selectBestDevice(result.instance->getInstance());
		if (physicalDevice == VK_NULL_HANDLE) {
			result.deviceResult = device::NO_PHYSICAL_DEVICE_FOUND;
			result.success		= false;
			return result;
		}

		// Create device
		device::Config deviceConfig;
		deviceConfig.renderInstance = result.instance;
		deviceConfig.vkInstance		= result.instance->getInstance();
		deviceConfig.physicalDevice = physicalDevice;

		result.deviceResult = addDevice(deviceConfig);
		if (result.deviceResult != device::INIT_DEVICE_SUCCESS) {
			result.success = false;
			return result;
		}

		// Get GPU pointer
		const auto& gpus = result.instance->getGPUs();
		if (!gpus.empty()) {
			result.gpu = gpus[0].get();
		}

		result.success = true;
		return result;
	}

	// ============================================================================
	// Convenience Functions
	// ============================================================================

	/**
	 * @brief Create a GPU context from a GPU
	 * @param gpu GPU pointer
	 * @return Initialized GPUContext
	 */
	inline GPUContext createContext(device::GPU* gpu) {
		GPUContext context;
		if (gpu) {
			context.initialize(gpu);
		}
		return context;
	}

} // namespace renderApi

#endif
