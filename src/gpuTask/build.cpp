#include "buffer/buffer.hpp"
#include "createDescriptorSetLayout.hpp"
#include "descriptor/descriptorSetManager.hpp"
#include "gpuTask.hpp"
#include "pipeline/computePipeline.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "query/queryPool.hpp"
#include "renderDevice.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

using namespace renderApi::gpuTask;

bool GpuTask::build(uint32_t renderWidth, uint32_t renderHeight) {
	if (isBuilt_) {
		return true;
	}

	if (!gpu_ || !gpu_->device) {
		std::cerr << "GPU not initialized" << std::endl;
		return false;
	}

	if (useDescriptorManager_ && descriptorManager_) {
		if (!descriptorManager_->build(gpu_)) {
			std::cerr << "Failed to build descriptor manager" << std::endl;
			return false;
		}
	} else if (!buffers_.empty()) {
		try {
			descriptorSetLayout_ = createDescriptorSetLayoutFromBuffers(gpu_->device, buffers_, bufferStages_);
		} catch (const std::exception& e) {
			std::cerr << "Failed to create descriptor set layout: " << e.what() << std::endl;
			return false;
		}

		std::vector<VkDescriptorPoolSize> poolSizes;

		uint32_t storageBufferCount = 0;
		uint32_t uniformBufferCount = 0;

		for (auto* buffer : buffers_) {
			if (buffer->getType() == BufferType::STORAGE) {
				storageBufferCount++;
			} else if (buffer->getType() == BufferType::UNIFORM) {
				uniformBufferCount++;
			}
		}

		if (storageBufferCount > 0) {
			poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storageBufferCount});
		}
		if (uniformBufferCount > 0) {
			poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBufferCount});
		}

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType		   = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes	   = poolSizes.data();
		poolInfo.maxSets	   = 1;
		poolInfo.flags		   = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

		if (vkCreateDescriptorPool(gpu_->device, &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
			std::cerr << "Failed to create descriptor pool" << std::endl;
			destroyDescriptorSetLayout(gpu_->device, descriptorSetLayout_);
			descriptorSetLayout_ = VK_NULL_HANDLE;
			return false;
		}

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType				 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool	 = descriptorPool_;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts		 = &descriptorSetLayout_;

		if (vkAllocateDescriptorSets(gpu_->device, &allocInfo, &descriptorSet_) != VK_SUCCESS) {
			std::cerr << "Failed to allocate descriptor set" << std::endl;
			destroy();
			return false;
		}

		std::vector<VkWriteDescriptorSet>	descriptorWrites;
		std::vector<VkDescriptorBufferInfo> bufferInfos;
		bufferInfos.reserve(buffers_.size());

		for (size_t i = 0; i < buffers_.size(); ++i) {
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = buffers_[i]->getHandle();
			bufferInfo.offset = 0;
			bufferInfo.range  = buffers_[i]->getSize();
			bufferInfos.push_back(bufferInfo);

			VkWriteDescriptorSet descriptorWrite{};
			descriptorWrite.sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet			= descriptorSet_;
			descriptorWrite.dstBinding		= static_cast<uint32_t>(i);
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType =
					(buffers_[i]->getType() == BufferType::STORAGE) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pBufferInfo		= &bufferInfos.back();

			descriptorWrites.push_back(descriptorWrite);
		}

		vkUpdateDescriptorSets(gpu_->device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}

	VkCommandPoolCreateInfo cmdPoolInfo{};
	cmdPoolInfo.sType			 = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex = !graphicsPipelines_.empty() ? gpu_->queueFamilies.graphicsFamily : gpu_->queueFamilies.computeFamily;
	cmdPoolInfo.flags			 = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (vkCreateCommandPool(gpu_->device, &cmdPoolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
		std::cerr << "Failed to create command pool" << std::endl;
		destroy();
		return false;
	}

	commandBuffers_.resize(maxFramesInFlight_);

	VkCommandBufferAllocateInfo cmdAllocInfo{};
	cmdAllocInfo.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdAllocInfo.commandPool		= commandPool_;
	cmdAllocInfo.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandBufferCount = maxFramesInFlight_;

	if (vkAllocateCommandBuffers(gpu_->device, &cmdAllocInfo, commandBuffers_.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate command buffers" << std::endl;
		destroy();
		return false;
	}

	std::cout << "GpuTask: Allocated " << maxFramesInFlight_ << " primary command buffers" << std::endl;

	if (graphicsPipelines_.size() > 1 && !useCustomRecording_) {
		std::cout << "GpuTask: Multiple pipelines detected (" << graphicsPipelines_.size() << "), creating secondary command buffers automatically..."
				  << std::endl;

		for (size_t i = 0; i < graphicsPipelines_.size(); ++i) {
			std::string		bufferName		= "pipeline_" + std::to_string(i);
			VkCommandBuffer secondaryBuffer = createSecondaryCommandBuffer(bufferName);
			if (secondaryBuffer == VK_NULL_HANDLE) {
				std::cerr << "Failed to create secondary command buffer for pipeline " << i << std::endl;
				destroy();
				return false;
			}
		}

		std::cout << "GpuTask: Created " << graphicsPipelines_.size() << " secondary command buffers (one per pipeline)" << std::endl;
	}

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (vkCreateFence(gpu_->device, &fenceInfo, nullptr, &fence_) != VK_SUCCESS) {
		std::cerr << "Failed to create fence" << std::endl;
		destroy();
		return false;
	}

	VkDescriptorSetLayout layout = VK_NULL_HANDLE;
	if (useDescriptorManager_ && descriptorManager_) {
		auto layouts = descriptorManager_->getLayouts();
		if (!layouts.empty()) {
			layout = layouts[0];
		}
	} else {
		layout = descriptorSetLayout_;
	}

	for (auto& pipeline : pipelines_) {
		if (!pipeline->build(layout)) {
			std::cerr << "Failed to build pipeline: " << pipeline->getName() << std::endl;
			destroy();
			return false;
		}
	}

	if (!graphicsPipelines_.empty()) {
		if (renderWidth == 0 || renderHeight == 0) {
			std::cerr << "Graphics pipeline requires render width and height" << std::endl;
			destroy();
			return false;
		}
		for (auto& pipeline : graphicsPipelines_) {
			if (!pipeline->build(layout, renderWidth, renderHeight)) {
				std::cerr << "Failed to build graphics pipeline: " << pipeline->getName() << std::endl;
				destroy();
				return false;
			}
		}
	}

	isBuilt_ = true;
	return true;
}

void GpuTask::destroy() {
	if (!gpu_ || !gpu_->device) {
		return;
	}

	pipelines_.clear();
	graphicsPipelines_.clear();

	if (descriptorManager_) {
		descriptorManager_->destroy();
	}

	if (queryPool_) {
		queryPool_->destroy();
	}

	if (fence_ != VK_NULL_HANDLE) {
		vkDestroyFence(gpu_->device, fence_, nullptr);
		fence_ = VK_NULL_HANDLE;
	}

	if (!secondaryCommandBuffers_.empty() && commandPool_ != VK_NULL_HANDLE) {
		std::vector<VkCommandBuffer> buffersToFree;
		for (const auto& scb : secondaryCommandBuffers_) {
			if (scb.buffer != VK_NULL_HANDLE) {
				buffersToFree.push_back(scb.buffer);
			}
		}
		if (!buffersToFree.empty()) {
			vkFreeCommandBuffers(gpu_->device, commandPool_, static_cast<uint32_t>(buffersToFree.size()), buffersToFree.data());
		}
		secondaryCommandBuffers_.clear();
	}

	if (!commandBuffers_.empty() && commandPool_ != VK_NULL_HANDLE) {
		vkFreeCommandBuffers(gpu_->device, commandPool_, static_cast<uint32_t>(commandBuffers_.size()), commandBuffers_.data());
		commandBuffers_.clear();
	}

	if (commandPool_ != VK_NULL_HANDLE) {
		vkDestroyCommandPool(gpu_->device, commandPool_, nullptr);
		commandPool_ = VK_NULL_HANDLE;
	}

	if (descriptorSet_ != VK_NULL_HANDLE && descriptorPool_ != VK_NULL_HANDLE) {
		vkFreeDescriptorSets(gpu_->device, descriptorPool_, 1, &descriptorSet_);
		descriptorSet_ = VK_NULL_HANDLE;
	}

	if (descriptorPool_ != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(gpu_->device, descriptorPool_, nullptr);
		descriptorPool_ = VK_NULL_HANDLE;
	}

	if (descriptorSetLayout_ != VK_NULL_HANDLE) {
		destroyDescriptorSetLayout(gpu_->device, descriptorSetLayout_);
		descriptorSetLayout_ = VK_NULL_HANDLE;
	}

	isBuilt_ = false;
}
