#include "gpuTask.hpp"

#include "buffer/buffer.hpp"
#include "renderDevice.hpp"
#include "createDescriptorSetLayout.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace renderApi::gpuTask {

	GpuTask::GpuTask(const std::string& name, device::GPU* gpu)
		: name_(name), gpu_(gpu), descriptorPool_(VK_NULL_HANDLE), descriptorSet_(VK_NULL_HANDLE), descriptorSetLayout_(VK_NULL_HANDLE),
		  isBuilt_(false) {}

	GpuTask::~GpuTask() { destroy(); }

	void GpuTask::addBuffer(Buffer* buffer, VkShaderStageFlags stageFlags) {
		if (isBuilt_) {
			std::cerr << "Cannot add buffer to built GPU task. Call destroy() first." << std::endl;
			return;
		}
		buffers_.push_back(buffer);
		bufferStages_.push_back(stageFlags);
	}

	void GpuTask::removeBuffer(Buffer* buffer) {
		if (isBuilt_) {
			std::cerr << "Cannot remove buffer from built GPU task. Call destroy() first." << std::endl;
			return;
		}
		for (size_t i = 0; i < buffers_.size(); ++i) {
			if (buffers_[i] == buffer) {
				buffers_.erase(buffers_.begin() + i);
				bufferStages_.erase(bufferStages_.begin() + i);
				break;
			}
		}
	}

	void GpuTask::clearBuffers() {
		if (isBuilt_) {
			std::cerr << "Cannot clear buffers from built GPU task. Call destroy() first." << std::endl;
			return;
		}
		buffers_.clear();
		bufferStages_.clear();
	}

	bool GpuTask::build() {
		if (isBuilt_) {
			return true;
		}

		if (!gpu_ || !gpu_->device) {
			std::cerr << "GPU not initialized" << std::endl;
			return false;
		}

		if (buffers_.empty()) {
			isBuilt_ = true;
			return true;
		}

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

		isBuilt_ = true;
		return true;
	}

	void GpuTask::destroy() {
		if (!gpu_ || !gpu_->device) {
			return;
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

	void GpuTask::executeTask() {
	}

}
