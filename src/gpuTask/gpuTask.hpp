#ifndef GPUTASK_HPP
#define GPUTASK_HPP

#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {
	class Buffer;
	enum class BufferType;
} // namespace renderApi

namespace renderApi::device {
	struct GPU;
}

namespace renderApi::gpuTask {

	class ComputePipeline;
	class GraphicsPipeline;

	class GpuTask {
		friend class ComputePipeline;

	  private:
		std::string	 name_;
		device::GPU* gpu_;

		VkDescriptorPool	  descriptorPool_	   = VK_NULL_HANDLE;
		VkDescriptorSet		  descriptorSet_	   = VK_NULL_HANDLE;
		VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;

		VkCommandPool	commandPool_   = VK_NULL_HANDLE;
		VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
		VkFence			fence_		   = VK_NULL_HANDLE;

		std::vector<Buffer*>			buffers_;
		std::vector<VkShaderStageFlags> bufferStages_;

		std::vector<Buffer*> vertexBuffers_;

		std::vector<std::unique_ptr<ComputePipeline>>  pipelines_;
		std::vector<std::unique_ptr<GraphicsPipeline>> graphicsPipelines_;

		bool isBuilt_ = false;
		bool enabled_ = true;
		bool autoExecute_ = false;

	  public:
		GpuTask(const std::string& name, device::GPU* gpu);
		~GpuTask();

		GpuTask(const GpuTask&)			   = delete;
		GpuTask& operator=(const GpuTask&) = delete;

		void addBuffer(Buffer* buffer, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT);
		void addVertexBuffer(Buffer* buffer);
		void removeBuffer(Buffer* buffer);
		void clearBuffers();

		ComputePipeline*  createComputePipeline(const std::string& name);
		GraphicsPipeline* createGraphicsPipeline(const std::string& name);

		bool build(uint32_t renderWidth = 0, uint32_t renderHeight = 0);
		void destroy();

		void execute();
		void wait();

		bool				  isBuilt() const { return isBuilt_; }
		VkDescriptorSet		  getDescriptorSet() const { return descriptorSet_; }
		VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout_; }
		VkCommandBuffer		  getCommandBuffer() const { return commandBuffer_; }
		const std::string&	  getName() const { return name_; }
		device::GPU*		  getGPU() const { return gpu_; }

		void setEnabled(bool enabled) { enabled_ = enabled; }
		bool isEnabled() const { return enabled_; }

		void setAutoExecute(bool autoExecute) { autoExecute_ = autoExecute; }
		bool isAutoExecute() const { return autoExecute_; }

		void registerWithGPU();
		void unregisterFromGPU();
	};

} // namespace renderApi::gpuTask

#endif
