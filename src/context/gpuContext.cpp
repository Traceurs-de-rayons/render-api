#include "gpuContext.hpp"

#include <cstring>
#include <iostream>

using namespace renderApi;

GPUContext::GPUContext() : gpu_(nullptr), oneTimeFence_(VK_NULL_HANDLE) {}

GPUContext::~GPUContext() { shutdown(); }

GPUContext::GPUContext(GPUContext&& other) noexcept : gpu_(other.gpu_), oneTimeFence_(other.oneTimeFence_) {
	other.gpu_			= nullptr;
	other.oneTimeFence_ = VK_NULL_HANDLE;
}

GPUContext& GPUContext::operator=(GPUContext&& other) noexcept {
	if (this != &other) {
		shutdown();
		gpu_				= other.gpu_;
		oneTimeFence_		= other.oneTimeFence_;
		other.gpu_			= nullptr;
		other.oneTimeFence_ = VK_NULL_HANDLE;
	}
	return *this;
}

bool GPUContext::initialize(device::GPU* gpu) {
	if (!gpu || !gpu->device) {
		std::cerr << "GPUContext: Invalid GPU" << std::endl;
		return false;
	}

	gpu_ = gpu;

	// Create fence for one-time commands
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	if (vkCreateFence(gpu_->device, &fenceInfo, nullptr, &oneTimeFence_) != VK_SUCCESS) {
		std::cerr << "GPUContext: Failed to create fence" << std::endl;
		return false;
	}

	return true;
}

void GPUContext::shutdown() {
	if (!gpu_) return;

	waitIdle();

	if (oneTimeFence_ != VK_NULL_HANDLE) {
		vkDestroyFence(gpu_->device, oneTimeFence_, nullptr);
		oneTimeFence_ = VK_NULL_HANDLE;
	}

	gpu_ = nullptr;
}

VkCommandBuffer GPUContext::beginOneTimeCommands() {
	if (!gpu_) return VK_NULL_HANDLE;

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool		 = gpu_->commandPool;
	allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(gpu_->device, &allocInfo, &cmd);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(cmd, &beginInfo);
	return cmd;
}

void GPUContext::endOneTimeCommands(VkCommandBuffer cmd) {
	if (!gpu_ || cmd == VK_NULL_HANDLE) return;

	vkEndCommandBuffer(cmd);

	VkSubmitInfo submitInfo{};
	submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers	  = &cmd;

	vkQueueSubmit(gpu_->graphicsQueue, 1, &submitInfo, oneTimeFence_);
	vkWaitForFences(gpu_->device, 1, &oneTimeFence_, VK_TRUE, UINT64_MAX);
	vkResetFences(gpu_->device, 1, &oneTimeFence_);

	vkFreeCommandBuffers(gpu_->device, gpu_->commandPool, 1, &cmd);
}

void GPUContext::waitIdle() const {
	if (gpu_ && gpu_->device) {
		vkDeviceWaitIdle(gpu_->device);
	}
}

bool GPUContext::submitGraphics(VkCommandBuffer					cmd,
								const std::vector<VkSemaphore>& waitSemaphores,
								const std::vector<VkSemaphore>& signalSemaphores,
								VkFence							fence) {
	if (!gpu_ || cmd == VK_NULL_HANDLE) return false;

	std::vector<VkPipelineStageFlags> waitStages(waitSemaphores.size(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	VkSubmitInfo submitInfo{};
	submitInfo.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount	= static_cast<uint32_t>(waitSemaphores.size());
	submitInfo.pWaitSemaphores		= waitSemaphores.empty() ? nullptr : waitSemaphores.data();
	submitInfo.pWaitDstStageMask	= waitStages.empty() ? nullptr : waitStages.data();
	submitInfo.commandBufferCount	= 1;
	submitInfo.pCommandBuffers		= &cmd;
	submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
	submitInfo.pSignalSemaphores	= signalSemaphores.empty() ? nullptr : signalSemaphores.data();

	return vkQueueSubmit(gpu_->graphicsQueue, 1, &submitInfo, fence) == VK_SUCCESS;
}

bool GPUContext::submitCompute(VkCommandBuffer				   cmd,
							   const std::vector<VkSemaphore>& waitSemaphores,
							   const std::vector<VkSemaphore>& signalSemaphores,
							   VkFence						   fence) {
	if (!gpu_ || cmd == VK_NULL_HANDLE) return false;

	std::vector<VkPipelineStageFlags> waitStages(waitSemaphores.size(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	VkSubmitInfo submitInfo{};
	submitInfo.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount	= static_cast<uint32_t>(waitSemaphores.size());
	submitInfo.pWaitSemaphores		= waitSemaphores.empty() ? nullptr : waitSemaphores.data();
	submitInfo.pWaitDstStageMask	= waitStages.empty() ? nullptr : waitStages.data();
	submitInfo.commandBufferCount	= 1;
	submitInfo.pCommandBuffers		= &cmd;
	submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
	submitInfo.pSignalSemaphores	= signalSemaphores.empty() ? nullptr : signalSemaphores.data();

	return vkQueueSubmit(gpu_->computeQueue, 1, &submitInfo, fence) == VK_SUCCESS;
}

// Convenience buffer creators
Buffer GPUContext::createBuffer(size_t size, BufferType type, BufferUsage usage) {
	Buffer buffer;
	buffer.create(*this, size, type, usage);
	return buffer;
}

Buffer GPUContext::createVertexBuffer(size_t size) { return createBuffer(size, BufferType::VERTEX, BufferUsage::STATIC); }

Buffer GPUContext::createIndexBuffer(size_t size) { return createBuffer(size, BufferType::INDEX, BufferUsage::STATIC); }

Buffer GPUContext::createUniformBuffer(size_t size) { return createBuffer(size, BufferType::UNIFORM, BufferUsage::DYNAMIC); }

Buffer GPUContext::createStorageBuffer(size_t size, BufferUsage usage) { return createBuffer(size, BufferType::STORAGE, usage); }

Buffer GPUContext::createStagingBuffer(size_t size) { return createBuffer(size, BufferType::STAGING, BufferUsage::STREAM); }

GraphicsPipeline GPUContext::createGraphicsPipeline(const GraphicsPipelineConfig& config) {
	GraphicsPipeline pipeline;
	pipeline.create(*this, config);
	return pipeline;
}

ComputePipeline GPUContext::createComputePipeline(const ComputePipelineConfig& config) {
	ComputePipeline pipeline;
	pipeline.create(*this, config);
	return pipeline;
}