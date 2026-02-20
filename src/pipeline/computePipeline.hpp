#ifndef COMPUTE_PIPELINE_HPP
#define COMPUTE_PIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi::device {
	struct GPU;
}

namespace renderApi::gpuTask {

	class GpuTask;

	class ComputePipeline {
	  public:
		ComputePipeline(device::GPU* gpu, const std::string& name);

	  private:
		device::GPU* gpu_ = nullptr;
		std::string	 name_;

		VkShaderModule	 shaderModule_	 = VK_NULL_HANDLE;
		VkPipeline		 pipeline_		 = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;

		bool enabled_ = true;

		uint32_t workgroupSizeX_ = 1;
		uint32_t workgroupSizeY_ = 1;
		uint32_t workgroupSizeZ_ = 1;

		VkPipelineShaderStageCreateInfo shaderStage_{};

		friend class GpuTask;

	  public:
		~ComputePipeline();

		ComputePipeline(const ComputePipeline&)			   = delete;
		ComputePipeline& operator=(const ComputePipeline&) = delete;
		ComputePipeline(ComputePipeline&& other) noexcept;
		ComputePipeline& operator=(ComputePipeline&& other) noexcept;

		void setShader(const std::vector<uint32_t>& spvCode);
		void setWorkgroupSize(uint32_t x, uint32_t y = 1, uint32_t z = 1);

		const std::string& getName() const { return name_; }
		VkPipeline		   getPipeline() const { return pipeline_; }
		VkPipelineLayout   getLayout() const { return pipelineLayout_; }

		void setEnabled(bool enabled) { enabled_ = enabled; }
		bool isEnabled() const { return enabled_; }

		void destroy();

		bool build(VkDescriptorSetLayout descriptorSetLayout);
	};

} // namespace renderApi::gpuTask

#endif
