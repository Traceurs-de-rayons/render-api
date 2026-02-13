#include "compute.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace renderApi::compute;

Buffer::~Buffer() { destroy(); }

Buffer::Buffer(Buffer&& other) noexcept
	: context_(other.context_), buffer_(other.buffer_), memory_(other.memory_), size_(other.size_), type_(other.type_), usage_(other.usage_),
	  mappedData_(other.mappedData_) {
	other.buffer_	  = VK_NULL_HANDLE;
	other.memory_	  = VK_NULL_HANDLE;
	other.size_		  = 0;
	other.mappedData_ = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
	if (this != &other) {
		destroy();
		context_		  = other.context_;
		buffer_			  = other.buffer_;
		memory_			  = other.memory_;
		size_			  = other.size_;
		type_			  = other.type_;
		usage_			  = other.usage_;
		mappedData_		  = other.mappedData_;
		other.buffer_	  = VK_NULL_HANDLE;
		other.memory_	  = VK_NULL_HANDLE;
		other.size_		  = 0;
		other.mappedData_ = nullptr;
	}
	return *this;
}

bool Buffer::create(ComputeContext& context, size_t size, BufferType type, BufferUsage usage) {
	destroy();

	context_ = &context;
	size_	 = size;
	type_	 = type;
	usage_	 = usage;

	if (!context_->isInitialized()) {
		std::cerr << "ComputeContext not initialized!" << std::endl;
		return false;
	}

	VkDevice	 device = context_->getDevice();
	device::GPU& gpu	= context_->getGPU();

	VkBufferUsageFlags usageFlags = 0;
	switch (type) {
	case BufferType::STORAGE:
		usageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		break;
	case BufferType::UNIFORM:
		usageFlags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		break;
	case BufferType::STAGING:
		usageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		break;
	}

	VkMemoryPropertyFlags memProperties = 0;
	switch (usage) {
	case BufferUsage::GPU_ONLY:
		memProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		break;
	case BufferUsage::CPU_TO_GPU:
		memProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		break;
	case BufferUsage::GPU_TO_CPU:
		memProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		break;
	case BufferUsage::CPU_GPU_BOTH:
		memProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		break;
	}

	if (type == BufferType::STAGING) {
		memProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	}

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType	   = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size		   = size;
	bufferInfo.usage	   = usageFlags;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer_) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer!" << std::endl;
		return false;
	}

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer_, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType			  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize  = memRequirements.size;
	allocInfo.memoryTypeIndex = device::findMemoryType(gpu.physicalDevice, memRequirements.memoryTypeBits, memProperties);

	if (vkAllocateMemory(device, &allocInfo, nullptr, &memory_) != VK_SUCCESS) {
		std::cerr << "Failed to allocate buffer memory!" << std::endl;
		vkDestroyBuffer(device, buffer_, nullptr);
		buffer_ = VK_NULL_HANDLE;
		return false;
	}

	vkBindBufferMemory(device, buffer_, memory_, 0);

	return true;
}

void Buffer::destroy() {
	if (!context_ || !context_->isInitialized()) return;

	VkDevice device = context_->getDevice();

	if (mappedData_) {
		unmap();
	}

	if (memory_ != VK_NULL_HANDLE) {
		vkFreeMemory(device, memory_, nullptr);
		memory_ = VK_NULL_HANDLE;
	}

	if (buffer_ != VK_NULL_HANDLE) {
		vkDestroyBuffer(device, buffer_, nullptr);
		buffer_ = VK_NULL_HANDLE;
	}

	size_ = 0;
}

bool Buffer::resize(size_t newSize) {
	if (!isValid()) return false;

	BufferType		oldType	   = type_;
	BufferUsage		oldUsage   = usage_;
	ComputeContext* oldContext = context_;

	destroy();

	return create(*oldContext, newSize, oldType, oldUsage);
}

bool Buffer::upload(const void* data, size_t size, size_t offset) {
	if (!isValid() || !data) return false;

	VkDevice device = context_->getDevice();

	if (usage_ == BufferUsage::CPU_TO_GPU || usage_ == BufferUsage::CPU_GPU_BOTH || type_ == BufferType::STAGING) {
		void* mapped = map();
		if (!mapped) return false;
		memcpy(static_cast<char*>(mapped) + offset, data, size);
		unmap();
		return true;
	}

	Buffer staging;
	if (!staging.create(*context_, size, BufferType::STAGING, BufferUsage::CPU_TO_GPU)) {
		return false;
	}

	void* mapped = staging.map();
	memcpy(mapped, data, size);
	staging.unmap();

	VkCommandBuffer cmd = context_->beginOneTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = offset;
	copyRegion.size		 = size;
	vkCmdCopyBuffer(cmd, staging.getHandle(), buffer_, 1, &copyRegion);

	context_->endOneTimeCommands(cmd);

	return true;
}

bool Buffer::download(void* data, size_t size, size_t offset) {
	if (!isValid() || !data) return false;

	VkDevice device = context_->getDevice();

	if (usage_ == BufferUsage::GPU_TO_CPU || usage_ == BufferUsage::CPU_GPU_BOTH || type_ == BufferType::STAGING) {
		void* mapped = map();
		if (!mapped) return false;
		memcpy(data, static_cast<char*>(mapped) + offset, size);
		unmap();
		return true;
	}

	Buffer staging;
	if (!staging.create(*context_, size, BufferType::STAGING, BufferUsage::GPU_TO_CPU)) {
		return false;
	}

	VkCommandBuffer cmd = context_->beginOneTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = offset;
	copyRegion.dstOffset = 0;
	copyRegion.size		 = size;
	vkCmdCopyBuffer(cmd, buffer_, staging.getHandle(), 1, &copyRegion);

	context_->endOneTimeCommands(cmd);

	void* mapped = staging.map();
	memcpy(data, mapped, size);
	staging.unmap();

	return true;
}

void* Buffer::map() {
	if (!isValid() || mappedData_) return mappedData_;

	VkDevice device = context_->getDevice();
	if (vkMapMemory(device, memory_, 0, size_, 0, &mappedData_) != VK_SUCCESS) {
		return nullptr;
	}
	return mappedData_;
}

void Buffer::unmap() {
	if (!isValid() || !mappedData_) return;

	VkDevice device = context_->getDevice();
	vkUnmapMemory(device, memory_);
	mappedData_ = nullptr;
}

Pipeline::~Pipeline() { destroy(); }

Pipeline::Pipeline(Pipeline&& other) noexcept
	: context_(other.context_), shaderModule_(other.shaderModule_), pipeline_(other.pipeline_), layout_(other.layout_),
	  descriptorSetLayout_(other.descriptorSetLayout_), descriptorSet_(other.descriptorSet_), boundBuffers_(std::move(other.boundBuffers_)),
	  descriptorsDirty_(other.descriptorsDirty_) {
	other.shaderModule_		   = VK_NULL_HANDLE;
	other.pipeline_			   = VK_NULL_HANDLE;
	other.layout_			   = VK_NULL_HANDLE;
	other.descriptorSetLayout_ = VK_NULL_HANDLE;
	other.descriptorSet_	   = VK_NULL_HANDLE;
}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
	if (this != &other) {
		destroy();
		context_				   = other.context_;
		shaderModule_			   = other.shaderModule_;
		pipeline_				   = other.pipeline_;
		layout_					   = other.layout_;
		descriptorSetLayout_	   = other.descriptorSetLayout_;
		descriptorSet_			   = other.descriptorSet_;
		boundBuffers_			   = std::move(other.boundBuffers_);
		descriptorsDirty_		   = other.descriptorsDirty_;
		other.shaderModule_		   = VK_NULL_HANDLE;
		other.pipeline_			   = VK_NULL_HANDLE;
		other.layout_			   = VK_NULL_HANDLE;
		other.descriptorSetLayout_ = VK_NULL_HANDLE;
		other.descriptorSet_	   = VK_NULL_HANDLE;
	}
	return *this;
}

void Pipeline::destroy() {
	if (!context_ || !context_->isInitialized()) return;

	VkDevice	 device = context_->getDevice();
	device::GPU& gpu	= context_->getGPU();

	if (pipeline_ != VK_NULL_HANDLE) {
		vkDestroyPipeline(device, pipeline_, nullptr);
		pipeline_ = VK_NULL_HANDLE;
	}

	if (layout_ != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(device, layout_, nullptr);
		layout_ = VK_NULL_HANDLE;
	}

	if (descriptorSet_ != VK_NULL_HANDLE) {
		vkFreeDescriptorSets(device, gpu.descriptorPool, 1, &descriptorSet_);
		descriptorSet_ = VK_NULL_HANDLE;
	}

	if (descriptorSetLayout_ != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout_, nullptr);
		descriptorSetLayout_ = VK_NULL_HANDLE;
	}

	if (shaderModule_ != VK_NULL_HANDLE) {
		vkDestroyShaderModule(device, shaderModule_, nullptr);
		shaderModule_ = VK_NULL_HANDLE;
	}

	boundBuffers_.clear();
}

bool Pipeline::create(ComputeContext& context, const std::vector<uint32_t>& spirvCode) {
	destroy();
	context_ = &context;

	if (!context_->isInitialized()) {
		std::cerr << "ComputeContext not initialized!" << std::endl;
		return false;
	}

	VkDevice device = context_->getDevice();

	VkShaderModuleCreateInfo shaderInfo{};
	shaderInfo.sType	= VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
	shaderInfo.pCode	= spirvCode.data();

	if (vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule_) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module!" << std::endl;
		return false;
	}

	if (!createDescriptorSetLayout()) {
		return false;
	}

	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType		  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = 1;
	layoutInfo.pSetLayouts	  = &descriptorSetLayout_;

	if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout_) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout!" << std::endl;
		return false;
	}

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType		  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	pipelineInfo.stage.module = shaderModule_;
	pipelineInfo.stage.pName  = "main";
	pipelineInfo.layout		  = layout_;

	if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_) != VK_SUCCESS) {
		std::cerr << "Failed to create compute pipeline!" << std::endl;
		return false;
	}

	return true;
}

bool Pipeline::createFromGLSL(ComputeContext& context, const std::string& glslCode) {
	std::cerr << "GLSL compilation not implemented. Use SPIR-V instead." << std::endl;
	return false;
}

bool Pipeline::createDescriptorSetLayout() {
	VkDevice device = context_->getDevice();

	std::vector<VkDescriptorSetLayoutBinding> bindings;

	for (uint32_t i = 0; i < 16; i++) {
		VkDescriptorSetLayoutBinding storageBinding{};
		storageBinding.binding		   = i;
		storageBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		storageBinding.descriptorCount = 1;
		storageBinding.stageFlags	   = VK_SHADER_STAGE_COMPUTE_BIT;
		bindings.push_back(storageBinding);
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType		= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings	= bindings.data();

	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout!" << std::endl;
		return false;
	}

	return true;
}

bool Pipeline::createDescriptorSet() {
	VkDevice	 device = context_->getDevice();
	device::GPU& gpu	= context_->getGPU();

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType				 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool	 = gpu.descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts		 = &descriptorSetLayout_;

	if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet_) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set!" << std::endl;
		return false;
	}

	return true;
}

bool Pipeline::bindBuffer(uint32_t binding, const Buffer& buffer) {
	if (binding >= 16) {
		std::cerr << "Binding index too high!" << std::endl;
		return false;
	}

	bool found = false;
	for (auto& bound : boundBuffers_) {
		// Check if binding already exists in boundBuffers_
		// This is a placeholder - you'd need to track binding indices
	}

	boundBuffers_.push_back({&buffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER});
	descriptorsDirty_ = true;
	return true;
}

bool Pipeline::bindUniformBuffer(uint32_t binding, const Buffer& buffer) {
	if (binding >= 16) {
		std::cerr << "Binding index too high!" << std::endl;
		return false;
	}

	boundBuffers_.push_back({&buffer, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER});
	descriptorsDirty_ = true;
	return true;
}

bool Pipeline::updateDescriptorSet() {
	if (!descriptorsDirty_ || boundBuffers_.empty()) return true;

	VkDevice device = context_->getDevice();

	// Allocate descriptor set if not already allocated
	if (descriptorSet_ == VK_NULL_HANDLE) {
		if (!createDescriptorSet()) return false;
	}

	std::vector<VkWriteDescriptorSet>	descriptorWrites;
	std::vector<VkDescriptorBufferInfo> bufferInfos;
	bufferInfos.reserve(boundBuffers_.size());

	for (size_t i = 0; i < boundBuffers_.size(); i++) {
		const auto& bound = boundBuffers_[i];

		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = bound.buffer->getHandle();
		bufferInfo.offset = 0;
		bufferInfo.range  = bound.buffer->getSize();
		bufferInfos.push_back(bufferInfo);

		VkWriteDescriptorSet write{};
		write.sType			  = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.dstSet		  = descriptorSet_;
		write.dstBinding	  = static_cast<uint32_t>(i);
		write.dstArrayElement = 0;
		write.descriptorType  = bound.type;
		write.descriptorCount = 1;
		write.pBufferInfo	  = &bufferInfos.back();
		descriptorWrites.push_back(write);
	}

	vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

	descriptorsDirty_ = false;
	return true;
}

bool Pipeline::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
	if (!isValid()) return false;

	// Update descriptors if needed
	if (!updateDescriptorSet()) return false;

	VkCommandBuffer cmd = context_->beginOneTimeCommands();

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &descriptorSet_, 0, nullptr);
	vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);

	context_->endOneTimeCommands(cmd);

	return true;
}

bool Pipeline::dispatchIndirect(const Buffer& indirectBuffer, size_t offset) {
	if (!isValid()) return false;

	// Update descriptors if needed
	if (!updateDescriptorSet()) return false;

	VkCommandBuffer cmd = context_->beginOneTimeCommands();

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &descriptorSet_, 0, nullptr);
	vkCmdDispatchIndirect(cmd, indirectBuffer.getHandle(), offset);

	context_->endOneTimeCommands(cmd);

	return true;
}

// ============================================================================
// ComputeContext Implementation
// ============================================================================

ComputeContext::~ComputeContext() { shutdown(); }

bool ComputeContext::initialize(device::GPU& gpu) {
	if (initialized_) return true;

	gpu_ = &gpu;

	VkDevice device = gpu.device;

	// Create compute fence
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	if (vkCreateFence(device, &fenceInfo, nullptr, &computeFence_) != VK_SUCCESS) {
		std::cerr << "Failed to create compute fence!" << std::endl;
		return false;
	}

	initialized_ = true;
	return true;
}

void ComputeContext::shutdown() {
	if (!initialized_) return;

	waitIdle();

	VkDevice device = gpu_->device;

	if (computeFence_ != VK_NULL_HANDLE) {
		vkDestroyFence(device, computeFence_, nullptr);
		computeFence_ = VK_NULL_HANDLE;
	}

	gpu_		 = nullptr;
	initialized_ = false;
}

VkCommandBuffer ComputeContext::beginOneTimeCommands() {
	VkDevice device = gpu_->device;

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool		 = gpu_->commandPool;
	allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(device, &allocInfo, &cmd);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(cmd, &beginInfo);

	return cmd;
}

void ComputeContext::endOneTimeCommands(VkCommandBuffer cmd) {
	VkDevice device = gpu_->device;

	vkEndCommandBuffer(cmd);

	VkSubmitInfo submitInfo{};
	submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers	  = &cmd;

	vkQueueSubmit(gpu_->computeQueue, 1, &submitInfo, computeFence_);
	vkWaitForFences(device, 1, &computeFence_, VK_TRUE, UINT64_MAX);
	vkResetFences(device, 1, &computeFence_);

	vkFreeCommandBuffers(device, gpu_->commandPool, 1, &cmd);
}

void ComputeContext::waitIdle() {
	if (!initialized_) return;
	vkDeviceWaitIdle(gpu_->device);
}

Buffer ComputeContext::createBuffer(size_t size, BufferType type, BufferUsage usage) {
	Buffer buffer;
	buffer.create(*this, size, type, usage);
	return buffer;
}

Buffer ComputeContext::createStagingBuffer(size_t size) { return createBuffer(size, BufferType::STAGING, BufferUsage::CPU_TO_GPU); }

Buffer ComputeContext::createStorageBuffer(size_t size) { return createBuffer(size, BufferType::STORAGE, BufferUsage::GPU_ONLY); }

Buffer ComputeContext::createUniformBuffer(size_t size) { return createBuffer(size, BufferType::UNIFORM, BufferUsage::CPU_TO_GPU); }

Pipeline ComputeContext::createPipeline(const std::vector<uint32_t>& spirvCode) {
	Pipeline pipeline;
	pipeline.create(*this, spirvCode);
	return pipeline;
}

Pipeline ComputeContext::createPipelineFromGLSL(const std::string& glslCode) {
	Pipeline pipeline;
	pipeline.createFromGLSL(*this, glslCode);
	return pipeline;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::vector<uint32_t> renderApi::compute::compileGLSLToSPIRV(const std::string& glslCode, const std::string& entryPoint) {
	// Placeholder - in a real implementation, use shaderc or glslang
	std::cerr << "GLSL compilation not implemented. Please compile shaders offline." << std::endl;
	return {};
}
