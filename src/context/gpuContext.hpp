#ifndef GPU_CONTEXT_HPP
#define GPU_CONTEXT_HPP

#include "buffer.hpp"
#include "descriptors.hpp"
#include "renderDevice.hpp"
#include "pipeline.hpp"

#include <cstddef>
#include <vulkan/vulkan_core.h>
#include <vector>

namespace renderApi {

	class GPUContext {
	  public:
		GPUContext();
		~GPUContext();

		GPUContext(const GPUContext&)			 = delete;
		GPUContext& operator=(const GPUContext&) = delete;
		GPUContext(GPUContext&& other) noexcept;
		GPUContext& operator=(GPUContext&& other) noexcept;

		bool initialize(device::GPU* gpu);
		void shutdown();
		bool isInitialized() const { return gpu_ != nullptr; }

		device::GPU&	   getGPU() { return *gpu_; }
		const device::GPU& getGPU() const { return *gpu_; }
		VkDevice		   getDevice() const { return gpu_ ? gpu_->device : VK_NULL_HANDLE; }
		VkQueue			   getGraphicsQueue() const { return gpu_ ? gpu_->graphicsQueue : VK_NULL_HANDLE; }
		VkQueue			   getComputeQueue() const { return gpu_ ? gpu_->computeQueue : VK_NULL_HANDLE; }
		VkQueue			   getTransferQueue() const { return gpu_ ? gpu_->transferQueue : VK_NULL_HANDLE; }
		VkCommandPool	   getCommandPool() const { return gpu_ ? gpu_->commandPool : VK_NULL_HANDLE; }
		VkDescriptorPool   getDescriptorPool() const { return gpu_ ? gpu_->descriptorPool : VK_NULL_HANDLE; }
		VkCommandBuffer beginOneTimeCommands();


		void endOneTimeCommands(VkCommandBuffer cmd);
		void waitIdle() const;
		bool submitGraphics(VkCommandBuffer					cmd,
							const std::vector<VkSemaphore>& waitSemaphores	 = {},
							const std::vector<VkSemaphore>& signalSemaphores = {},
							VkFence							fence			 = VK_NULL_HANDLE);

		bool submitCompute(VkCommandBuffer				   cmd,
						   const std::vector<VkSemaphore>& waitSemaphores	= {},
						   const std::vector<VkSemaphore>& signalSemaphores = {},
						   VkFence						   fence			= VK_NULL_HANDLE);

		Buffer createBuffer(size_t size, BufferType type, BufferUsage usage = BufferUsage::STATIC);
		Buffer createVertexBuffer(size_t size);
		Buffer createIndexBuffer(size_t size);
		Buffer createUniformBuffer(size_t size);
		Buffer createStorageBuffer(size_t size, BufferUsage usage = BufferUsage::STATIC);
		Buffer createStagingBuffer(size_t size);

		GraphicsPipeline createGraphicsPipeline(const GraphicsPipelineConfig& config);
		ComputePipeline	 createComputePipeline(const ComputePipelineConfig& config);

	  private:
		device::GPU* gpu_;
		VkFence		 oneTimeFence_;
	};

}

#endif
