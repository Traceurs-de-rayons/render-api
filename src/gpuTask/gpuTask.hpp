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

namespace renderApi::descriptor {
	class DescriptorSetManager;
}

namespace renderApi::query {
	class QueryPool;
}

namespace renderApi::gpuTask {

	class ComputePipeline;
	class GraphicsPipeline;

	class GpuTask {
		friend class ComputePipeline;

	  private:
		std::string	 name_;
		device::GPU* gpu_;

		// Legacy descriptor support (for backward compatibility)
		VkDescriptorPool	  descriptorPool_	   = VK_NULL_HANDLE;
		VkDescriptorSet		  descriptorSet_	   = VK_NULL_HANDLE;
		VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;

		// New descriptor manager for multiple sets
		std::unique_ptr<descriptor::DescriptorSetManager> descriptorManager_;
		bool useDescriptorManager_ = false;

		// Query pool for profiling
		std::unique_ptr<query::QueryPool> queryPool_;

		VkCommandPool	commandPool_   = VK_NULL_HANDLE;
		VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
		VkFence			fence_		   = VK_NULL_HANDLE;

		std::vector<Buffer*>			buffers_;
		std::vector<VkShaderStageFlags> bufferStages_;

		std::vector<Buffer*> vertexBuffers_;
		Buffer*				 indexBuffer_	  = nullptr;
		VkIndexType			 indexType_		  = VK_INDEX_TYPE_UINT32;
		uint32_t			 indexCount_	  = 0;
		uint32_t			 vertexCount_	  = 3;
		uint32_t			 instanceCount_	  = 1;
		uint32_t			 firstVertex_	  = 0;
		uint32_t			 firstIndex_	  = 0;
		uint32_t			 vertexOffset_	  = 0;
		uint32_t			 firstInstance_	  = 0;

		std::vector<std::unique_ptr<ComputePipeline>>  pipelines_;
		std::vector<std::unique_ptr<GraphicsPipeline>> graphicsPipelines_;

		// Push constants storage
		struct PushConstantData {
			VkShaderStageFlags stageFlags;
			uint32_t offset;
			uint32_t size;
			std::vector<uint8_t> data;
		};
		std::vector<PushConstantData> pushConstants_;

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
		void setIndexBuffer(Buffer* buffer, VkIndexType indexType = VK_INDEX_TYPE_UINT32);
		void setDrawParams(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0, uint32_t firstInstance = 0);
		void setIndexedDrawParams(uint32_t indexCount, uint32_t instanceCount = 1, uint32_t firstIndex = 0, int32_t vertexOffset = 0, uint32_t firstInstance = 0);
		void removeBuffer(Buffer* buffer);
		void clearBuffers();
		
		// Push constants
		void pushConstants(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* data);

		// Descriptor set manager (new system)
		descriptor::DescriptorSetManager* getDescriptorManager();
		void enableDescriptorManager(bool enable = true);

		// Query pool support
		query::QueryPool* createQueryPool(uint32_t queryCount = 64);
		query::QueryPool* getQueryPool() const { return queryPool_.get(); }

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
