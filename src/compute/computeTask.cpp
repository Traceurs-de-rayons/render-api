#include "computeTask.hpp"

#include "gpuContext.hpp"

#include <iostream>
#include <utility>
#include <vulkan/vulkan_core.h>

using namespace renderApi;

ComputeTask::ComputeTask()
	: context_(nullptr), hasShader_(false), pipeline_(VK_NULL_HANDLE), pipelineLayout_(VK_NULL_HANDLE), built_(false), groupsX_(1),
	  groupsY_(1), groupsZ_(1), enabled_(true) {}

ComputeTask::~ComputeTask() { destroy(); }

ComputeTask::ComputeTask(ComputeTask&& other) noexcept
	: context_(other.context_), name_(std::move(other.name_)), shader_(std::move(other.shader_)), hasShader_(other.hasShader_),
	  pipeline_(other.pipeline_), pipelineLayout_(other.pipelineLayout_), descriptorLayout_(std::move(other.descriptorLayout_)),
	  descriptorSet_(std::move(other.descriptorSet_)), bufferBindings_(std::move(other.bufferBindings_)), built_(other.built_),
	  groupsX_(other.groupsX_), groupsY_(other.groupsY_), groupsZ_(other.groupsZ_), enabled_(other.enabled_.load()) {
	other.pipeline_		  = VK_NULL_HANDLE;
	other.pipelineLayout_ = VK_NULL_HANDLE;
	other.hasShader_	  = false;
	other.built_		  = false;
}

ComputeTask& ComputeTask::operator=(ComputeTask&& other) noexcept {
	if (this != &other) {
		destroy();

		context_		  = other.context_;
		name_			  = std::move(other.name_);
		shader_			  = std::move(other.shader_);
		hasShader_		  = other.hasShader_;
		pipeline_		  = other.pipeline_;
		pipelineLayout_	  = other.pipelineLayout_;
		descriptorLayout_ = std::move(other.descriptorLayout_);
		descriptorSet_	  = std::move(other.descriptorSet_);
		bufferBindings_	  = std::move(other.bufferBindings_);
		built_			  = other.built_;
		groupsX_		  = other.groupsX_;
		groupsY_		  = other.groupsY_;
		groupsZ_		  = other.groupsZ_;
		enabled_.store(other.enabled_.load());

		other.pipeline_		  = VK_NULL_HANDLE;
		other.pipelineLayout_ = VK_NULL_HANDLE;
		other.hasShader_	  = false;
		other.built_		  = false;
	}
	return *this;
}

bool ComputeTask::create(GPUContext& context, const std::string& name) {
	context_ = &context;
	name_	 = name.empty() ? "ComputeTask" : name;

	return true;
}

void ComputeTask::destroy() {
	if (!context_) return;

	if (pipeline_ != VK_NULL_HANDLE) {
		vkDestroyPipeline(context_->getDevice(), pipeline_, nullptr);
		pipeline_ = VK_NULL_HANDLE;
	}

	if (pipelineLayout_ != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context_->getDevice(), pipelineLayout_, nullptr);
		pipelineLayout_ = VK_NULL_HANDLE;
	}

	destroyShaderModule();

	descriptorSet_.free();
	descriptorLayout_.destroy();

	bufferBindings_.clear();
	hasShader_ = false;
	built_	   = false;
	context_   = nullptr;
}

ComputeTask& ComputeTask::setShader(const std::vector<uint32_t>& spirvCode, const std::string& name,
									const std::string& entryPoint) {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot set shader after build()! Call rebuild() instead." << std::endl;
		return *this;
	}

	if (spirvCode.empty()) {
		std::cerr << "[" << name_ << "] Shader SPIR-V code is empty!" << std::endl;
		return *this;
	}

	// Destroy old shader if exists
	destroyShaderModule();

	shader_	   = ComputeShaderModule(spirvCode, name, entryPoint);
	hasShader_ = true;

	std::cout << "[" << name_ << "] Shader set successfully." << std::endl;

	return *this;
}

ComputeTask& ComputeTask::updateShader(const std::vector<uint32_t>& spirvCode) {
	if (!hasShader_) {
		std::cerr << "[" << name_ << "] No shader to update!" << std::endl;
		return *this;
	}

	if (spirvCode.empty()) {
		std::cerr << "[" << name_ << "] Shader SPIR-V code is empty!" << std::endl;
		return *this;
	}

	// Destroy old module
	destroyShaderModule();

	// Update code
	shader_.spirvCode = spirvCode;
	shader_.module	  = VK_NULL_HANDLE;

	// If already built, need to rebuild
	if (built_) {
		std::cout << "[" << name_ << "] Shader updated, rebuilding pipeline..." << std::endl;
		rebuild();
	}

	return *this;
}

void ComputeTask::clearShader() {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot clear shader after build()!" << std::endl;
		return;
	}

	destroyShaderModule();
	hasShader_ = false;
}

ComputeTask& ComputeTask::bindBuffer(uint32_t binding, Buffer& buffer) {
	if (built_) {
		std::cerr << "[" << name_ << "] Cannot bind buffer after build()!" << std::endl;
		return *this;
	}

	for (auto& bb : bufferBindings_) {
		if (bb.binding == binding) {
			bb.buffer = &buffer;
			return *this;
		}
	}

	bufferBindings_.push_back({binding, &buffer});
	return *this;
}

bool ComputeTask::build() {
	if (built_) {
		std::cerr << "[" << name_ << "] Already built!" << std::endl;
		return true;
	}

	if (!context_) {
		std::cerr << "[" << name_ << "] Context is null!" << std::endl;
		return false;
	}

	if (!hasShader_) {
		std::cerr << "[" << name_ << "] No shader set!" << std::endl;
		return false;
	}

	std::cout << "[" << name_ << "] Building compute task..." << std::endl;

	if (!createShaderModule())
		return false;

	for (const auto& bb : bufferBindings_)
		descriptorLayout_.addStorageBuffer(bb.binding, VK_SHADER_STAGE_COMPUTE_BIT);

	if (!bufferBindings_.empty()) {
		if (!descriptorLayout_.build(*context_)) {
			std::cerr << "[" << name_ << "] Failed to build descriptor layout!" << std::endl;
			return false;
		}
	}

	if (!createPipeline())
		return false;

	if (!bufferBindings_.empty()) {
		if (!createDescriptors())
			return false;
	}

	built_ = true;
	std::cout << "[" << name_ << "] Build successful!" << std::endl;
	return true;
}

bool ComputeTask::rebuild() {
	if (!context_) {
		std::cerr << "[" << name_ << "] Context is null!" << std::endl;
		return false;
	}

	std::cout << "[" << name_ << "] Rebuilding compute task..." << std::endl;

	// Destroy old pipeline
	if (pipeline_ != VK_NULL_HANDLE) {
		vkDestroyPipeline(context_->getDevice(), pipeline_, nullptr);
		pipeline_ = VK_NULL_HANDLE;
	}

	if (pipelineLayout_ != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context_->getDevice(), pipelineLayout_, nullptr);
		pipelineLayout_ = VK_NULL_HANDLE;
	}

	// Destroy and recreate shader module
	destroyShaderModule();

	built_ = false;

	return build();
}

void ComputeTask::setDispatchSize(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) {
	groupsX_ = groupsX;
	groupsY_ = groupsY;
	groupsZ_ = groupsZ;
}

void ComputeTask::execute(VkCommandBuffer cmd) {
	if (!isEnabled() || !built_ || pipeline_ == VK_NULL_HANDLE) {
		return;
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

	if (!bufferBindings_.empty()) {
		VkDescriptorSet setHandle = descriptorSet_.getHandle();
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &setHandle, 0, nullptr);
	}

	vkCmdDispatch(cmd, groupsX_, groupsY_, groupsZ_);
}

bool ComputeTask::createShaderModule() {
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = shader_.spirvCode.size() * sizeof(uint32_t);
	createInfo.pCode	= shader_.spirvCode.data();

	if (vkCreateShaderModule(context_->getDevice(), &createInfo, nullptr, &shader_.module) != VK_SUCCESS) {
		std::cerr << "[" << name_ << "] Failed to create shader module!" << std::endl;
		return false;
	}

	return true;
}

void ComputeTask::destroyShaderModule() {
	if (shader_.module != VK_NULL_HANDLE && context_) {
		vkDestroyShaderModule(context_->getDevice(), shader_.module, nullptr);
		shader_.module = VK_NULL_HANDLE;
	}
}

bool ComputeTask::createPipeline() {
	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

	if (!bufferBindings_.empty()) {
		layoutInfo.setLayoutCount		   = 1;
		VkDescriptorSetLayout layoutHandle = descriptorLayout_.getHandle();
		layoutInfo.pSetLayouts			   = &layoutHandle;
	} else {
		layoutInfo.setLayoutCount = 0;
		layoutInfo.pSetLayouts	  = nullptr;
	}

	if (vkCreatePipelineLayout(context_->getDevice(), &layoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
		std::cerr << "[" << name_ << "] Failed to create pipeline layout!" << std::endl;
		return false;
	}

	VkPipelineShaderStageCreateInfo stageInfo{};
	stageInfo.sType	 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stageInfo.stage	 = VK_SHADER_STAGE_COMPUTE_BIT;
	stageInfo.module = shader_.module;
	stageInfo.pName	 = shader_.entryPoint.c_str();

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType	= VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage	= stageInfo;
	pipelineInfo.layout = pipelineLayout_;

	if (vkCreateComputePipelines(context_->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_) != VK_SUCCESS) {
		std::cerr << "[" << name_ << "] Failed to create compute pipeline!" << std::endl;
		return false;
	}

	return true;
}

bool ComputeTask::createDescriptors() {
	if (!descriptorSet_.allocate(*context_, descriptorLayout_)) {
		std::cerr << "[" << name_ << "] Failed to allocate descriptor set!" << std::endl;
		return false;
	}

	for (const auto& bb : bufferBindings_) {
		if (bb.buffer && bb.buffer->isValid()) {
			descriptorSet_.updateStorageBuffer(bb.binding, bb.buffer->getHandle());
		}
	}

	return true;
}