#include "gpuTask.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "query/queryPool.hpp"
#include "renderDevice.hpp"

#include <cstdint>
#include <iostream>
#include <mutex>
#include <vulkan/vulkan.h>

using namespace renderApi::gpuTask;

// One frame of the task: acquireFrame -> recordCommands -> submitFrame.
// See executeFrame.cpp, recordGraphics.cpp and recordCompute.cpp.
void GpuTask::execute() {
	if (!isBuilt_ || !gpu_ || !gpu_->device) {
		std::cerr << "GpuTask not built" << std::endl;
		return;
	}

	std::lock_guard<std::mutex> lock(executionMutex_);

	const bool usesSwapchain = !graphicsPipelines_.empty() && graphicsPipelines_[0]->getSwapchain() != VK_NULL_HANDLE;
	uint32_t   imageIndex	 = 0;

	if (!acquireFrame(usesSwapchain, imageIndex))
		return;

	VkCommandBuffer commandBuffer = commandBuffers_[currentFrame_];

	vkResetCommandBuffer(commandBuffer, 0);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		std::cerr << "Failed to begin command buffer" << std::endl;
		return;
	}

	if (queryPool_ && queryPool_->isValid()) {
		queryPool_->reset(commandBuffer);
	}

	recordCommands(commandBuffer, imageIndex, usesSwapchain);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		std::cerr << "Failed to end command buffer" << std::endl;
		return;
	}

	if (!submitFrame(commandBuffer, imageIndex, usesSwapchain))
		return;

	currentFrame_ = (currentFrame_ + 1) % maxFramesInFlight_;
}

// Picks how the frame is recorded: user callbacks, default graphics draw (inline or through
// secondary command buffers) or compute dispatch.
void GpuTask::recordCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex, bool usesSwapchain) {
	if (useCustomRecording_ && !recordingCallbacks_.empty()) {
		for (const auto& callback : recordingCallbacks_) {
			callback(commandBuffer, currentFrame_, imageIndex);
		}
	} else if (!graphicsPipelines_.empty() && secondaryCommandBuffers_.empty()) {
		recordGraphicsInline(commandBuffer, imageIndex, usesSwapchain);
	} else if (!graphicsPipelines_.empty() && !secondaryCommandBuffers_.empty()) {
		recordGraphicsSecondary(commandBuffer, imageIndex, usesSwapchain);
	} else if (!pipelines_.empty() && !useCustomRecording_) {
		recordCompute(commandBuffer);
	}
}
