#include "buffer/buffer.hpp"
#include "createDescriptorSetLayout.hpp"
#include "descriptor/descriptorSetManager.hpp"
#include "gpuTask.hpp"
#include "pipeline/computePipeline.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "query/queryPool.hpp"
#include "renderDevice.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

using namespace renderApi::gpuTask;

VkCommandBuffer GpuTask::createSecondaryCommandBuffer(const std::string& name) {
	if (!gpu_ || !gpu_->device || commandPool_ == VK_NULL_HANDLE) {
		std::cerr << "GpuTask: Cannot create secondary command buffer, task not built" << std::endl;
		return VK_NULL_HANDLE;
	}

	for (const auto& scb : secondaryCommandBuffers_) {
		if (scb.name == name) {
			std::cerr << "GpuTask: Secondary command buffer '" << name << "' already exists" << std::endl;
			return scb.buffer;
		}
	}

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool		 = commandPool_;
	allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer secondaryBuffer;
	if (vkAllocateCommandBuffers(gpu_->device, &allocInfo, &secondaryBuffer) != VK_SUCCESS) {
		std::cerr << "GpuTask: Failed to allocate secondary command buffer '" << name << "'" << std::endl;
		return VK_NULL_HANDLE;
	}

	SecondaryCommandBuffer scb;
	scb.name	= name;
	scb.buffer	= secondaryBuffer;
	scb.enabled = true;

	secondaryCommandBuffers_.push_back(scb);

	std::cout << "GpuTask: Created secondary command buffer '" << name << "'" << std::endl;
	return secondaryBuffer;
}

void GpuTask::recordSecondaryCommandBuffer(const std::string& name, RecordingCallback callback) {
	VkCommandBuffer secondaryBuffer = VK_NULL_HANDLE;
	for (const auto& scb : secondaryCommandBuffers_) {
		if (scb.name == name) {
			secondaryBuffer = scb.buffer;
			break;
		}
	}

	if (secondaryBuffer == VK_NULL_HANDLE) {
		std::cerr << "GpuTask: Secondary command buffer '" << name << "' not found" << std::endl;
		return;
	}

	if (!graphicsPipelines_.empty()) {
		VkCommandBufferInheritanceInfo inheritanceInfo{};
		inheritanceInfo.sType		= VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		inheritanceInfo.renderPass	= graphicsPipelines_[0]->getRenderPass();
		inheritanceInfo.subpass		= 0;
		inheritanceInfo.framebuffer = VK_NULL_HANDLE;

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType			   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags			   = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = &inheritanceInfo;

		if (vkBeginCommandBuffer(secondaryBuffer, &beginInfo) != VK_SUCCESS) {
			std::cerr << "GpuTask: Failed to begin secondary command buffer '" << name << "'" << std::endl;
			return;
		}

		callback(secondaryBuffer, currentFrame_, 0);

		if (vkEndCommandBuffer(secondaryBuffer) != VK_SUCCESS) {
			std::cerr << "GpuTask: Failed to end secondary command buffer '" << name << "'" << std::endl;
			return;
		}

		std::cout << "GpuTask: Recorded secondary command buffer '" << name << "'" << std::endl;
	}
}

void GpuTask::executeSecondaryCommandBuffers(VkCommandBuffer primaryCmd) {
	std::vector<VkCommandBuffer> secondariesToExecute;

	for (const auto& scb : secondaryCommandBuffers_) {
		if (scb.enabled && scb.buffer != VK_NULL_HANDLE) {
			secondariesToExecute.push_back(scb.buffer);
		}
	}

	if (!secondariesToExecute.empty()) {
		vkCmdExecuteCommands(primaryCmd, static_cast<uint32_t>(secondariesToExecute.size()), secondariesToExecute.data());
		std::cout << "GpuTask: Executed " << secondariesToExecute.size() << " secondary command buffers" << std::endl;
	}
}

void GpuTask::enableSecondaryCommandBuffer(const std::string& name, bool enable) {
	for (auto& scb : secondaryCommandBuffers_) {
		if (scb.name == name) {
			scb.enabled = enable;
			std::cout << "GpuTask: Secondary command buffer '" << name << "' " << (enable ? "enabled" : "disabled") << std::endl;
			return;
		}
	}
	std::cerr << "GpuTask: Secondary command buffer '" << name << "' not found" << std::endl;
}

void GpuTask::destroySecondaryCommandBuffer(const std::string& name) {
	for (auto it = secondaryCommandBuffers_.begin(); it != secondaryCommandBuffers_.end(); ++it) {
		if (it->name == name) {
			if (it->buffer != VK_NULL_HANDLE && commandPool_ != VK_NULL_HANDLE && gpu_ && gpu_->device) {
				vkFreeCommandBuffers(gpu_->device, commandPool_, 1, &it->buffer);
			}
			secondaryCommandBuffers_.erase(it);
			std::cout << "GpuTask: Destroyed secondary command buffer '" << name << "'" << std::endl;
			return;
		}
	}
	std::cerr << "GpuTask: Secondary command buffer '" << name << "' not found" << std::endl;
}
