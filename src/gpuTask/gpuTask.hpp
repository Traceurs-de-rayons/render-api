#ifndef GPUTASK_HPP
#define GPUTASK_HPP

#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {
	class Buffer;
}

namespace renderApi::device {
	struct GPU;
}

namespace renderApi::gpuTask {

	class GpuTask {
	  private:
		std::string	 name_;
		device::GPU* gpu_;

		// Descriptor resources
		VkDescriptorPool	  descriptorPool_	   = VK_NULL_HANDLE;
		VkDescriptorSet		  descriptorSet_	   = VK_NULL_HANDLE;
		VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;

		// Buffers
		std::vector<Buffer*>			buffers_;
		std::vector<VkShaderStageFlags> bufferStages_;

		bool isBuilt_ = false;

	  public:
		GpuTask(const std::string& name, device::GPU* gpu);
		~GpuTask();

		// Buffer management
		void addBuffer(Buffer* buffer, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT);
		void removeBuffer(Buffer* buffer);
		void clearBuffers();

		// Build descriptor resources
		bool build();
		void destroy();

		// Execute the task
		void executeTask();

		// Getters
		bool				  isBuilt() const { return isBuilt_; }
		VkDescriptorSet		  getDescriptorSet() const { return descriptorSet_; }
		VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout_; }
		const std::string&	  getName() const { return name_; }
	};

} // namespace renderApi::gpuTask

#endif
