#include "gpuTask.hpp"

#include "buffer/buffer.hpp"
#include "createDescriptorSetLayout.hpp"
#include "descriptor/descriptorSetManager.hpp"
#include "pipeline/computePipeline.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "query/queryPool.hpp"
#include "renderDevice.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
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

	void GpuTask::setIndexBuffer(Buffer* buffer, VkIndexType indexType) {
		if (isBuilt_) {
			std::cerr << "Cannot set index buffer to built GPU task. Call destroy() first." << std::endl;
			return;
		}
		indexBuffer_ = buffer;
		indexType_	 = indexType;
	}

	void GpuTask::setDrawParams(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
		vertexCount_   = vertexCount;
		instanceCount_ = instanceCount;
		firstVertex_   = firstVertex;
		firstInstance_ = firstInstance;
	}

	void
	GpuTask::setIndexedDrawParams(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
		indexCount_	   = indexCount;
		instanceCount_ = instanceCount;
		firstIndex_	   = firstIndex;
		vertexOffset_  = vertexOffset;
		firstInstance_ = firstInstance;
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

	void GpuTask::pushConstants(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* data) {
		// Store push constants to be applied during command buffer recording in execute()
		PushConstantData pcData;
		pcData.stageFlags = stageFlags;
		pcData.offset = offset;
		pcData.size = size;
		pcData.data.resize(size);
		memcpy(pcData.data.data(), data, size);

		// Clear previous push constants and store the new one
		// (In a more advanced implementation, you could merge multiple ranges)
		pushConstants_.clear();
		pushConstants_.push_back(pcData);
	}

	descriptor::DescriptorSetManager* GpuTask::getDescriptorManager() {
		if (!descriptorManager_) {
			descriptorManager_ = std::make_unique<descriptor::DescriptorSetManager>();
		}
		return descriptorManager_.get();
	}

	void GpuTask::enableDescriptorManager(bool enable) {
		useDescriptorManager_ = enable;
		if (enable && !descriptorManager_) {
			descriptorManager_ = std::make_unique<descriptor::DescriptorSetManager>();
		}
	}

	query::QueryPool* GpuTask::createQueryPool(uint32_t queryCount) {
		if (!queryPool_) {
			queryPool_ = std::make_unique<query::QueryPool>();
			if (!queryPool_->create(gpu_, query::QueryType::TIMESTAMP, queryCount)) {
				std::cerr << "Failed to create query pool" << std::endl;
				queryPool_.reset();
				return nullptr;
			}
		}
		return queryPool_.get();
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

		// Use new descriptor manager if enabled
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

		// Get descriptor set layout
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (useDescriptorManager_ && descriptorManager_) {
			auto layouts = descriptorManager_->getLayouts();
			if (!layouts.empty()) {
				layout = layouts[0]; // Use first set for now
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

		// Reset query pool if present
		if (queryPool_ && queryPool_->isValid()) {
			// Will be reset in command buffer
		}

		uint32_t imageIndex	   = 0;
		bool	 usesSwapchain = false;

		if (!graphicsPipelines_.empty() && graphicsPipelines_[0]->getSwapchain() != VK_NULL_HANDLE) {
			usesSwapchain		  = true;
			VkSemaphore semaphore = graphicsPipelines_[0]->getImageAvailableSemaphore();
			VkResult	result =
					vkAcquireNextImageKHR(gpu_->device, graphicsPipelines_[0]->getSwapchain(), UINT64_MAX, semaphore, VK_NULL_HANDLE, &imageIndex);

			if (result == VK_ERROR_OUT_OF_DATE_KHR) {
				graphicsPipelines_[0]->recreateSwapchain();
				return;
			} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
				std::cerr << "Failed to acquire swapchain image" << std::endl;
				return;
			}
		}

		vkResetCommandBuffer(commandBuffer_, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		if (vkBeginCommandBuffer(commandBuffer_, &beginInfo) != VK_SUCCESS) {
			std::cerr << "Failed to begin command buffer" << std::endl;
			return;
		}

		// Reset query pool at the start of command buffer
		if (queryPool_ && queryPool_->isValid()) {
			queryPool_->reset(commandBuffer_);
		}

		if (!graphicsPipelines_.empty()) {
			std::vector<VkClearValue> clearValues(2);
			clearValues[0].color		= {{0.2f, 0.2f, 0.2f, 1.0f}};  // Gray background for better visibility
			clearValues[1].depthStencil = {1.0f, 0};

			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType	  = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = graphicsPipelines_[0]->getRenderPass();
			if (usesSwapchain) {
				renderPassInfo.framebuffer = graphicsPipelines_[0]->getSwapchainFramebuffer(imageIndex);
			} else {
				renderPassInfo.framebuffer = graphicsPipelines_[0]->getFramebuffer();
			}
			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = {graphicsPipelines_[0]->getWidth(), graphicsPipelines_[0]->getHeight()};
			renderPassInfo.clearValueCount	 = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues		 = clearValues.data();



			// Bind descriptor sets
			if (useDescriptorManager_ && descriptorManager_) {
				auto descriptorSets = descriptorManager_->getDescriptorSets();
				auto layouts		= descriptorManager_->getLayouts();
				if (!descriptorSets.empty() && !graphicsPipelines_.empty()) {
					vkCmdBindDescriptorSets(commandBuffer_,
											VK_PIPELINE_BIND_POINT_GRAPHICS,
											graphicsPipelines_[0]->getLayout(),
											0,
											static_cast<uint32_t>(descriptorSets.size()),
											descriptorSets.data(),
											0,
											nullptr);
				}
			} else if (!buffers_.empty()) {
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

			if (indexBuffer_ != nullptr) {
				vkCmdBindIndexBuffer(commandBuffer_, indexBuffer_->getHandle(), 0, indexType_);
			}

			for (auto& pipeline : graphicsPipelines_) {
				if (pipeline->isEnabled()) {
					vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getPipeline());

					// Apply push constants if any
					for (const auto& pc : pushConstants_) {
						vkCmdPushConstants(commandBuffer_, pipeline->getLayout(), pc.stageFlags, pc.offset, pc.size, pc.data.data());
					}

					if (indexBuffer_ != nullptr) {
						vkCmdDrawIndexed(commandBuffer_, indexCount_, instanceCount_, firstIndex_, vertexOffset_, firstInstance_);
					} else {
						vkCmdDraw(commandBuffer_, vertexCount_, instanceCount_, firstVertex_, firstInstance_);
					}
				}
			}

			vkCmdEndRenderPass(commandBuffer_);
		} else if (!pipelines_.empty()) {
			// Bind descriptor sets for compute
			if (useDescriptorManager_ && descriptorManager_) {
				auto descriptorSets = descriptorManager_->getDescriptorSets();
				if (!descriptorSets.empty() && !pipelines_.empty()) {
					vkCmdBindDescriptorSets(commandBuffer_,
											VK_PIPELINE_BIND_POINT_COMPUTE,
											pipelines_[0]->getLayout(),
											0,
											static_cast<uint32_t>(descriptorSets.size()),
											descriptorSets.data(),
											0,
											nullptr);
				}
			} else if (!buffers_.empty()) {
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
				if (pipeline->isEnabled()) {
					vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipeline());

					// Apply push constants if any
					for (const auto& pc : pushConstants_) {
						vkCmdPushConstants(commandBuffer_, pipeline->getLayout(), pc.stageFlags, pc.offset, pc.size, pc.data.data());
					}

					vkCmdDispatch(commandBuffer_, 1, 1, 1);
				}
			}
		}

		// No need for manual layout transition - renderpass finalLayout already handles it

		if (vkEndCommandBuffer(commandBuffer_) != VK_SUCCESS) {
			std::cerr << "Failed to end command buffer" << std::endl;
			return;
		}

		VkSubmitInfo submitInfo{};
		submitInfo.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount	= 1;
		submitInfo.pCommandBuffers		= &commandBuffer_;
		submitInfo.waitSemaphoreCount	= 0;
		submitInfo.pWaitSemaphores		= nullptr;
		submitInfo.pWaitDstStageMask	= nullptr;
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores	= nullptr;

		VkQueue queue;
		if (!graphicsPipelines_.empty() && !gpu_->graphicsQueues.empty()) {
			queue = gpu_->graphicsQueues[0];
		} else {
			queue = gpu_->computeQueues[0];
		}

		{
			std::lock_guard<std::mutex> lock(gpu_->queueMutex);

			VkSemaphore			 waitSemaphore	 = VK_NULL_HANDLE;
			VkSemaphore			 signalSemaphore = VK_NULL_HANDLE;
			VkPipelineStageFlags waitStage		 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

			if (usesSwapchain) {
				waitSemaphore	= graphicsPipelines_[0]->getImageAvailableSemaphore();
				signalSemaphore = graphicsPipelines_[0]->getRenderFinishedSemaphore();

				submitInfo.waitSemaphoreCount = 1;
				submitInfo.pWaitSemaphores	  = &waitSemaphore;
				submitInfo.pWaitDstStageMask  = &waitStage;

				submitInfo.signalSemaphoreCount = 1;
				submitInfo.pSignalSemaphores	= &signalSemaphore;
			} else {
				submitInfo.waitSemaphoreCount	= 0;
				submitInfo.pWaitSemaphores		= nullptr;
				submitInfo.pWaitDstStageMask	= nullptr;
				submitInfo.signalSemaphoreCount = 0;
				submitInfo.pSignalSemaphores	= nullptr;
			}

			if (vkQueueSubmit(queue, 1, &submitInfo, fence_) != VK_SUCCESS) {
				std::cerr << "Failed to submit queue" << std::endl;
				return;
			}

			if (usesSwapchain) {
				VkSwapchainKHR swapchain = graphicsPipelines_[0]->getSwapchain();

				VkPresentInfoKHR presentInfo{};
				presentInfo.sType			   = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
				presentInfo.waitSemaphoreCount = 1;
				presentInfo.pWaitSemaphores	   = &signalSemaphore;
				presentInfo.swapchainCount	   = 1;
				presentInfo.pSwapchains		   = &swapchain;
				presentInfo.pImageIndices	   = &imageIndex;

				VkQueue presentQueue = gpu_->getPresentQueue();
				if (presentQueue != VK_NULL_HANDLE) {
					VkResult presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
					if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
						graphicsPipelines_[0]->recreateSwapchain();
					} else if (presentResult != VK_SUCCESS) {
						std::cerr << "Failed to present swapchain image" << std::endl;
					}
				}
			}
		}

		if (usesSwapchain) {
			vkQueueWaitIdle(queue);
		}
	}

	void GpuTask::wait() {
		if (gpu_ && gpu_->device && fence_ != VK_NULL_HANDLE) {
			vkWaitForFences(gpu_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
		}
	}

	void GpuTask::registerWithGPU() {
		if (!gpu_) {
			std::cerr << "Cannot register GpuTask: GPU is null" << std::endl;
			return;
		}

		std::lock_guard<std::mutex> lock(gpu_->GpuTasksMutex);

		for (auto* task : gpu_->GpuTasks) {
			if (task == this) {
				return;
			}
		}

		gpu_->GpuTasks.push_back(this);
	}

	void GpuTask::unregisterFromGPU() {
		if (!gpu_) {
			return;
		}

		std::lock_guard<std::mutex> lock(gpu_->GpuTasksMutex);

		auto it = std::find(gpu_->GpuTasks.begin(), gpu_->GpuTasks.end(), this);
		if (it != gpu_->GpuTasks.end()) {
			gpu_->GpuTasks.erase(it);
		}
	}

} // namespace renderApi::gpuTask
