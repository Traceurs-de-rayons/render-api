#include "pipeline/computePipeline.hpp"

#include "device/renderDevice.hpp"

#include <iostream>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace renderApi::gpuTask {

	ComputePipeline::ComputePipeline(device::GPU* gpu, const std::string& name) : gpu_(gpu), name_(name) {}

	ComputePipeline::~ComputePipeline() { destroy(); }

	ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
		: gpu_(other.gpu_), name_(std::move(other.name_)), shaderModule_(other.shaderModule_), pipeline_(other.pipeline_),
		  pipelineLayout_(other.pipelineLayout_), workgroupSizeX_(other.workgroupSizeX_), workgroupSizeY_(other.workgroupSizeY_),
		  workgroupSizeZ_(other.workgroupSizeZ_), shaderStage_(other.shaderStage_) {
		other.shaderModule_	  = VK_NULL_HANDLE;
		other.pipeline_		  = VK_NULL_HANDLE;
		other.pipelineLayout_ = VK_NULL_HANDLE;
	}

	ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
		if (this != &other) {
			destroy();
			gpu_				  = other.gpu_;
			name_				  = std::move(other.name_);
			shaderModule_		  = other.shaderModule_;
			pipeline_			  = other.pipeline_;
			pipelineLayout_		  = other.pipelineLayout_;
			workgroupSizeX_		  = other.workgroupSizeX_;
			workgroupSizeY_		  = other.workgroupSizeY_;
			workgroupSizeZ_		  = other.workgroupSizeZ_;
			shaderStage_		  = other.shaderStage_;
			other.shaderModule_	  = VK_NULL_HANDLE;
			other.pipeline_		  = VK_NULL_HANDLE;
			other.pipelineLayout_ = VK_NULL_HANDLE;
		}
		return *this;
	}

	void ComputePipeline::setShader(const std::vector<uint32_t>& spvCode) {
		if (!gpu_) {
			std::cerr << "ComputePipeline: GPU not initialized" << std::endl;
			return;
		}

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = spvCode.size() * sizeof(uint32_t);
		createInfo.pCode	= spvCode.data();

		if (vkCreateShaderModule(gpu_->device, &createInfo, nullptr, &shaderModule_) != VK_SUCCESS) {
			std::cerr << "ComputePipeline: Failed to create shader module" << std::endl;
			return;
		}

		shaderStage_.sType	= VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage_.stage	= VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStage_.module = shaderModule_;
		shaderStage_.pName	= "main";
	}

	void ComputePipeline::setWorkgroupSize(uint32_t x, uint32_t y, uint32_t z) {
		workgroupSizeX_ = x;
		workgroupSizeY_ = y;
		workgroupSizeZ_ = z;
	}

	bool ComputePipeline::build(VkDescriptorSetLayout descriptorSetLayout) {
		if (!gpu_ || !gpu_->device) {
			std::cerr << "ComputePipeline: GPU not initialized" << std::endl;
			return false;
		}

		if (shaderModule_ == VK_NULL_HANDLE) {
			std::cerr << "ComputePipeline: No shader set" << std::endl;
			return false;
		}

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType		  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts	  = &descriptorSetLayout;

		if (vkCreatePipelineLayout(gpu_->device, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
			std::cerr << "ComputePipeline: Failed to create pipeline layout" << std::endl;
			return false;
		}

		VkComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType				= VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.stage				= shaderStage_;
		pipelineInfo.layout				= pipelineLayout_;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateComputePipelines(gpu_->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_) != VK_SUCCESS) {
			std::cerr << "ComputePipeline: Failed to create compute pipeline" << std::endl;
			vkDestroyPipelineLayout(gpu_->device, pipelineLayout_, nullptr);
			pipelineLayout_ = VK_NULL_HANDLE;
			return false;
		}

		return true;
	}

	void ComputePipeline::destroy() {
		if (!gpu_ || !gpu_->device) {
			return;
		}

		if (pipeline_ != VK_NULL_HANDLE) {
			vkDestroyPipeline(gpu_->device, pipeline_, nullptr);
			pipeline_ = VK_NULL_HANDLE;
		}

		if (pipelineLayout_ != VK_NULL_HANDLE) {
			vkDestroyPipelineLayout(gpu_->device, pipelineLayout_, nullptr);
			pipelineLayout_ = VK_NULL_HANDLE;
		}

		if (shaderModule_ != VK_NULL_HANDLE) {
			vkDestroyShaderModule(gpu_->device, shaderModule_, nullptr);
			shaderModule_ = VK_NULL_HANDLE;
		}
	}

} // namespace renderApi::gpuTask
