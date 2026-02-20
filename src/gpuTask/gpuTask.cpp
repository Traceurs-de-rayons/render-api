#include "gpuTask.hpp"

#include "buffer/buffer.hpp"
#include "createDescriptorSetLayout.hpp"
#include "pipeline/computePipeline.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "renderDevice.hpp"

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
		  commandPool_(VK_NULL_HANDLE), commandBuffer_(VK_NULL_HANDLE), fence_(VK_NULL_HANDLE), isBuilt_(false) {}

	GpuTask::~GpuTask() { destroy(); }

	void GpuTask::addBuffer(Buffer* buffer, VkShaderStageFlags stageFlags) {
		if (isBuilt_) {
			std::cerr << "Cannot add buffer to built GPU task. Call destroy() first." << std::endl;
			return;
		}
		buffers_.push_back(buffer);
		bufferStages_.push_back(stageFlags);
	}

	void GpuTask::addVertexBuffer(Buffer* buffer) {
		if (isBuilt_) {
			std::cerr << "Cannot add vertex buffer to built GPU task. Call destroy() first." << std::endl;
			return;
		}
		vertexBuffers_.push_back(buffer);
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

	ComputePipeline* GpuTask::createComputePipeline(const std::string& name) {
		auto  pipeline = std::make_unique<ComputePipeline>(gpu_, name);
		auto* ptr	   = pipeline.get();
		pipelines_.push_back(std::move(pipeline));
		return ptr;
	}

	GraphicsPipeline* GpuTask::createGraphicsPipeline(const std::string& name) {
		auto  pipeline = std::make_unique<GraphicsPipeline>(gpu_, name);
		auto* ptr	   = pipeline.get();
		graphicsPipelines_.push_back(std::move(pipeline));
		return ptr;
	}

	bool GpuTask::build(uint32_t renderWidth, uint32_t renderHeight) {
		if (isBuilt_) {
			return true;
		}

		if (!gpu_ || !gpu_->device) {
			std::cerr << "GPU not initialized" << std::endl;
			return false;
		}

		if (!buffers_.empty()) {
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

		VkCommandBufferAllocateInfo cmdAllocInfo{};
		cmdAllocInfo.sType				= VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdAllocInfo.commandPool		= commandPool_;
		cmdAllocInfo.level				= VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdAllocInfo.commandBufferCount = 1;

		if (vkAllocateCommandBuffers(gpu_->device, &cmdAllocInfo, &commandBuffer_) != VK_SUCCESS) {
			std::cerr << "Failed to allocate command buffer" << std::endl;
			destroy();
			return false;
		}

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		if (vkCreateFence(gpu_->device, &fenceInfo, nullptr, &fence_) != VK_SUCCESS) {
			std::cerr << "Failed to create fence" << std::endl;
			destroy();
			return false;
		}

		for (auto& pipeline : pipelines_) {
			if (!pipeline->build(descriptorSetLayout_)) {
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
				if (!pipeline->build(descriptorSetLayout_, renderWidth, renderHeight)) {
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

		if (fence_ != VK_NULL_HANDLE) {
			vkDestroyFence(gpu_->device, fence_, nullptr);
			fence_ = VK_NULL_HANDLE;
		}

		if (commandBuffer_ != VK_NULL_HANDLE && commandPool_ != VK_NULL_HANDLE) {
			vkFreeCommandBuffers(gpu_->device, commandPool_, 1, &commandBuffer_);
			commandBuffer_ = VK_NULL_HANDLE;
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

	void GpuTask::execute() {
		if (!isBuilt_ || !gpu_ || !gpu_->device) {
			std::cerr << "GpuTask not built" << std::endl;
			return;
		}

		vkWaitForFences(gpu_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
		vkResetFences(gpu_->device, 1, &fence_);

		vkResetCommandBuffer(commandBuffer_, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		if (vkBeginCommandBuffer(commandBuffer_, &beginInfo) != VK_SUCCESS) {
			std::cerr << "Failed to begin command buffer" << std::endl;
			return;
		}

		if (!graphicsPipelines_.empty()) {
			std::vector<VkClearValue> clearValues(2);
			clearValues[0].color		= {{0.0f, 0.0f, 0.0f, 1.0f}};
			clearValues[1].depthStencil = {1.0f, 0};

			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType			 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass		 = graphicsPipelines_[0]->getRenderPass();
			renderPassInfo.framebuffer		 = graphicsPipelines_[0]->getFramebuffer();
			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = {graphicsPipelines_[0]->getWidth(), graphicsPipelines_[0]->getHeight()};
			renderPassInfo.clearValueCount	 = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues		 = clearValues.data();

			if (!buffers_.empty()) {
				vkCmdBindDescriptorSets(
						commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelines_[0]->getLayout(), 0, 1, &descriptorSet_, 0, nullptr);
			}

			vkCmdBeginRenderPass(commandBuffer_, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			if (!vertexBuffers_.empty()) {
				std::vector<VkBuffer>	  vkBuffers(vertexBuffers_.size());
				std::vector<VkDeviceSize> offsets(vertexBuffers_.size(), 0);
				for (size_t i = 0; i < vertexBuffers_.size(); ++i) {
					vkBuffers[i] = vertexBuffers_[i]->getHandle();
				}
				vkCmdBindVertexBuffers(commandBuffer_, 0, static_cast<uint32_t>(vkBuffers.size()), vkBuffers.data(), offsets.data());
			}

			for (auto& pipeline : graphicsPipelines_) {
				vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getPipeline());
				vkCmdDraw(commandBuffer_, 3, 1, 0, 0);
			}

			vkCmdEndRenderPass(commandBuffer_);
		} else if (!pipelines_.empty()) {
			if (!buffers_.empty()) {
				vkCmdBindDescriptorSets(commandBuffer_,
										VK_PIPELINE_BIND_POINT_COMPUTE,
										pipelines_.empty() ? VK_NULL_HANDLE : pipelines_[0]->getLayout(),
										0,
										1,
										&descriptorSet_,
										0,
										nullptr);
			}

			for (auto& pipeline : pipelines_) {
				vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipeline());
				vkCmdDispatch(commandBuffer_, 1, 1, 1);
			}
		}

		if (vkEndCommandBuffer(commandBuffer_) != VK_SUCCESS) {
			std::cerr << "Failed to end command buffer" << std::endl;
			return;
		}

		VkSubmitInfo submitInfo{};
		submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers	  = &commandBuffer_;

		VkQueue queue;
		if (!graphicsPipelines_.empty() && !gpu_->graphicsQueues.empty()) {
			queue = gpu_->graphicsQueues[0];
		} else {
			queue = gpu_->computeQueues[0];
		}

		if (vkQueueSubmit(queue, 1, &submitInfo, fence_) != VK_SUCCESS) {
			std::cerr << "Failed to submit queue" << std::endl;
			return;
		}
	}

	void GpuTask::wait() {
		if (gpu_ && gpu_->device && fence_ != VK_NULL_HANDLE) {
			vkWaitForFences(gpu_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
		}
	}

} // namespace renderApi::gpuTask
