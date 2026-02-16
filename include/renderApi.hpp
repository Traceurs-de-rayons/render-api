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

	// vk instance
	instance::InitInstanceResult			initNewInstance(const instance::Config& config);
	std::vector<instance::RenderInstance>&  getInstances();
	// add un clean


	// vk device
	device::InitDeviceResult				addDevice(const device::Config& config);
	std::vector<device::PhysicalDeviceInfo> enumerateDevices(VkInstance instance);
	VkPhysicalDevice						selectBestDevice(VkInstance instance);
	// add un clean

	// pipeline
	ComputeTask*							createComputeTask(const std::vector<uint32_t>& spirvCode, const std::string& name = "", size_t gpuIndex = 0);
	GraphicsTask*							createGraphicsTask(RenderWindow* window, const std::string& name = "", size_t gpuIndex = 0);
	ComputeTask*							getComputeTask(const std::string& name, size_t gpuIndex = 0);
	GraphicsTask*							getGraphicsTask(const std::string& name, size_t gpuIndex = 0);

	bool									addShader(const std::string& taskName, ShaderStage stage,
														const std::vector<uint32_t>& spirvCode,
														const std::string& name = "",
														const std::string& entryPoint = "main",
														size_t gpuIndex = 0);
	bool									removeShader(const std::string& taskName, ShaderStage stage, size_t gpuIndex = 0);
	bool									updateShader(const std::string& taskName, ShaderStage stage,
														const std::vector<uint32_t>& spirvCode, size_t gpuIndex = 0);
	bool									hasShader(const std::string& taskName, ShaderStage stage, size_t gpuIndex = 0);

	void									removeComputeTask(const std::string& name, size_t gpuIndex = 0);
	void									removeGraphicsTask(const std::string& name, size_t gpuIndex = 0);
	void									clearAllTasks(size_t gpuIndex = 0);


	// Shader enable/disable for graphics tasks
	bool									enableShader(const std::string& taskName, ShaderStage stage, size_t gpuIndex = 0);
	bool									disableShader(const std::string& taskName, ShaderStage stage, size_t gpuIndex = 0);
	bool									isShaderEnabled(const std::string& taskName, ShaderStage stage, size_t gpuIndex = 0);
	void									clearShaders(const std::string& taskName, size_t gpuIndex = 0);

	// Buffer binding for compute tasks
	bool									bindComputeBuffer(const std::string& taskName, uint32_t binding, Buffer& buffer, size_t gpuIndex = 0);
	bool									setComputeDispatchSize(const std::string& taskName, uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1, size_t gpuIndex = 0);

	// Buffer binding for graphics tasks
	bool									bindVertexBuffer(const std::string& taskName, uint32_t binding, Buffer& buffer, uint32_t stride, size_t gpuIndex = 0);
	bool									bindIndexBuffer(const std::string& taskName, Buffer& buffer, VkIndexType indexType = VK_INDEX_TYPE_UINT32, size_t gpuIndex = 0);
	bool									bindUniformBuffer(const std::string& taskName, uint32_t binding, Buffer& buffer, size_t gpuIndex = 0);

	// Viewport & Scissor for graphics tasks
	bool									setViewport(const std::string& taskName, float x, float y, float width, float height, size_t gpuIndex = 0);
	bool									setScissor(const std::string& taskName, int32_t x, int32_t y, uint32_t width, uint32_t height, size_t gpuIndex = 0);
	bool									resetViewport(const std::string& taskName, size_t gpuIndex = 0);
	bool									resetScissor(const std::string& taskName, size_t gpuIndex = 0);

	// Buffer creation helpers
	Buffer									createBuffer(size_t size, BufferType type, BufferUsage usage = BufferUsage::STATIC, size_t gpuIndex = 0);
	Buffer									createVertexBuffer(size_t size, BufferUsage usage = BufferUsage::STATIC, size_t gpuIndex = 0);
	Buffer									createIndexBuffer(size_t size, BufferUsage usage = BufferUsage::STATIC, size_t gpuIndex = 0);
	Buffer									createUniformBuffer(size_t size, BufferUsage usage = BufferUsage::DYNAMIC, size_t gpuIndex = 0);
	Buffer									createStorageBuffer(size_t size, BufferUsage usage = BufferUsage::STATIC, size_t gpuIndex = 0);
	Buffer									createStagingBuffer(size_t size, size_t gpuIndex = 0);

	// Task control - compute
	bool									buildComputeTask(const std::string& taskName, size_t gpuIndex = 0);
	bool									rebuildComputeTask(const std::string& taskName, size_t gpuIndex = 0);
	bool									setComputeTaskEnabled(const std::string& taskName, bool enabled, size_t gpuIndex = 0);
	bool									isComputeTaskEnabled(const std::string& taskName, size_t gpuIndex = 0);
	bool									isComputeTaskBuilt(const std::string& taskName, size_t gpuIndex = 0);
	bool									isComputeTaskValid(const std::string& taskName, size_t gpuIndex = 0);

	// Task control - graphics
	bool									buildGraphicsTask(const std::string& taskName, size_t gpuIndex = 0);
	bool									rebuildGraphicsTask(const std::string& taskName, size_t gpuIndex = 0);
	bool									setGraphicsTaskEnabled(const std::string& taskName, bool enabled, size_t gpuIndex = 0);
	bool									isGraphicsTaskEnabled(const std::string& taskName, size_t gpuIndex = 0);
	bool									isGraphicsTaskBuilt(const std::string& taskName, size_t gpuIndex = 0);
	bool									isGraphicsTaskValid(const std::string& taskName, size_t gpuIndex = 0);

	// create renderDevice
	GPUContext								createContext(device::GPU* gpu);

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
