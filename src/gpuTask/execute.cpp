#include "buffer/buffer.hpp"
#include "createDescriptorSetLayout.hpp"
#include "descriptor/descriptorSetManager.hpp"
#include "gpuTask.hpp"
#include "pipeline/computePipeline.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "query/queryPool.hpp"
#include "renderDevice.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

using namespace renderApi::gpuTask;

void GpuTask::execute() {
	if (!isBuilt_ || !gpu_ || !gpu_->device) {
		std::cerr << "GpuTask not built" << std::endl;
		return;
	}

	uint32_t imageIndex	   = 0;
	bool	 usesSwapchain = false;

	if (!graphicsPipelines_.empty() && graphicsPipelines_[0]->getSwapchain() != VK_NULL_HANDLE) {
		usesSwapchain = true;
	}

	if (!usesSwapchain) {
		vkWaitForFences(gpu_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
		vkResetFences(gpu_->device, 1, &fence_);
	}

	if (usesSwapchain) {
		VkFence inFlightFence = graphicsPipelines_[0]->getInFlightFence();
		if (inFlightFence != VK_NULL_HANDLE) {
			vkWaitForFences(gpu_->device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
		}

		// Get the acquire semaphore for this frame
		VkSemaphore acquireSemaphore = graphicsPipelines_[0]->getImageAvailableSemaphore();
		
		// Acquire next image - this gives us the imageIndex
		VkResult result =
				vkAcquireNextImageKHR(gpu_->device, graphicsPipelines_[0]->getSwapchain(), UINT64_MAX, acquireSemaphore, VK_NULL_HANDLE, &imageIndex);

		if (result == VK_TIMEOUT) {
			std::cerr << "Warning: Acquire image timeout!" << std::endl;
			return;
		} else if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			graphicsPipelines_[0]->recreateSwapchain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			std::cerr << "Failed to acquire swapchain image: " << result << std::endl;
			return;
		}

		// Check if this image is already being used by another frame
		auto& imagesInFlight = graphicsPipelines_[0]->imagesInFlight_;
		if (imageIndex < imagesInFlight.size() && imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
			vkWaitForFences(gpu_->device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
		}
		// Mark this image as now being used by this frame
		imagesInFlight[imageIndex] = inFlightFence;

		if (inFlightFence != VK_NULL_HANDLE) {
			vkResetFences(gpu_->device, 1, &inFlightFence);
		}
	}

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

	if (useCustomRecording_ && !recordingCallbacks_.empty()) {
		for (const auto& callback : recordingCallbacks_) {
			callback(commandBuffer, currentFrame_, imageIndex);
		}
	} else if (!graphicsPipelines_.empty() && secondaryCommandBuffers_.empty()) {
		std::vector<VkClearValue> clearValues(2);
		clearValues[0].color		= {{0.2f, 0.2f, 0.2f, 1.0f}};
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

		if (useDescriptorManager_ && descriptorManager_) {
			auto descriptorSets = descriptorManager_->getDescriptorSets();
			auto layouts		= descriptorManager_->getLayouts();
			if (!descriptorSets.empty() && !graphicsPipelines_.empty()) {
				vkCmdBindDescriptorSets(commandBuffer,
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
					commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelines_[0]->getLayout(), 0, 1, &descriptorSet_, 0, nullptr);
		}

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		if (!vertexBuffers_.empty()) {
			std::vector<VkBuffer>	  vkBuffers(vertexBuffers_.size());
			std::vector<VkDeviceSize> offsets(vertexBuffers_.size(), 0);
			for (size_t i = 0; i < vertexBuffers_.size(); ++i) {
				vkBuffers[i] = vertexBuffers_[i]->getHandle();
			}
			vkCmdBindVertexBuffers(commandBuffer, 0, static_cast<uint32_t>(vkBuffers.size()), vkBuffers.data(), offsets.data());
		}

		if (indexBuffer_ != nullptr) {
			vkCmdBindIndexBuffer(commandBuffer, indexBuffer_->getHandle(), 0, indexType_);
		}

		for (auto& pipeline : graphicsPipelines_) {
			if (pipeline->isEnabled()) {
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getPipeline());

				for (const auto& pc : pushConstants_) {
					vkCmdPushConstants(commandBuffer, pipeline->getLayout(), pc.stageFlags, pc.offset, pc.size, pc.data.data());
				}

				if (indexBuffer_ != nullptr) {
					vkCmdDrawIndexed(commandBuffer, indexCount_, instanceCount_, firstIndex_, vertexOffset_, firstInstance_);
				} else {
					vkCmdDraw(commandBuffer, vertexCount_, instanceCount_, firstVertex_, firstInstance_);
				}
			}
		}

		vkCmdEndRenderPass(commandBuffer);
	} else if (!graphicsPipelines_.empty() && !secondaryCommandBuffers_.empty()) {
		std::vector<VkClearValue> clearValues(2);
		clearValues[0].color		= {{0.2f, 0.2f, 0.2f, 1.0f}};
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

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

		std::vector<VkCommandBuffer> secondariesToExecute;

		for (size_t pipelineIdx = 0; pipelineIdx < graphicsPipelines_.size(); ++pipelineIdx) {
			std::string bufferName = "pipeline_" + std::to_string(pipelineIdx);

			VkCommandBuffer secondaryBuffer = VK_NULL_HANDLE;
			for (const auto& scb : secondaryCommandBuffers_) {
				if (scb.name == bufferName && scb.enabled) {
					secondaryBuffer = scb.buffer;
					break;
				}
			}

			if (secondaryBuffer != VK_NULL_HANDLE) {
				VkCommandBufferInheritanceInfo inheritanceInfo{};
				inheritanceInfo.sType	   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
				inheritanceInfo.renderPass = graphicsPipelines_[0]->getRenderPass();
				inheritanceInfo.subpass	   = 0;
				inheritanceInfo.framebuffer =
						usesSwapchain ? graphicsPipelines_[0]->getSwapchainFramebuffer(imageIndex) : graphicsPipelines_[0]->getFramebuffer();

				VkCommandBufferBeginInfo beginInfo{};
				beginInfo.sType			   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				beginInfo.flags			   = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
				beginInfo.pInheritanceInfo = &inheritanceInfo;

				vkResetCommandBuffer(secondaryBuffer, 0);
				if (vkBeginCommandBuffer(secondaryBuffer, &beginInfo) == VK_SUCCESS) {
					auto* pipeline = graphicsPipelines_[pipelineIdx].get();

					if (pipeline->isEnabled()) {
						vkCmdBindPipeline(secondaryBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getPipeline());

						if (!vertexBuffers_.empty()) {
							std::vector<VkBuffer>	  vkBuffers(vertexBuffers_.size());
							std::vector<VkDeviceSize> offsets(vertexBuffers_.size(), 0);
							for (size_t i = 0; i < vertexBuffers_.size(); ++i) {
								vkBuffers[i] = vertexBuffers_[i]->getHandle();
							}
							vkCmdBindVertexBuffers(secondaryBuffer, 0, static_cast<uint32_t>(vkBuffers.size()), vkBuffers.data(), offsets.data());
						}

						if (indexBuffer_ != nullptr) {
							vkCmdBindIndexBuffer(secondaryBuffer, indexBuffer_->getHandle(), 0, indexType_);
						}

						if (!buffers_.empty()) {
							vkCmdBindDescriptorSets(
									secondaryBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getLayout(), 0, 1, &descriptorSet_, 0, nullptr);
						}

						for (const auto& pc : pushConstants_) {
							vkCmdPushConstants(secondaryBuffer, pipeline->getLayout(), pc.stageFlags, pc.offset, pc.size, pc.data.data());
						}

						if (indexBuffer_ != nullptr) {
							vkCmdDrawIndexed(secondaryBuffer, indexCount_, instanceCount_, firstIndex_, vertexOffset_, firstInstance_);
						} else {
							vkCmdDraw(secondaryBuffer, vertexCount_, instanceCount_, firstVertex_, firstInstance_);
						}
					}

					vkEndCommandBuffer(secondaryBuffer);
					secondariesToExecute.push_back(secondaryBuffer);
				}
			}
		}

		if (!secondariesToExecute.empty()) {
			vkCmdExecuteCommands(commandBuffer, static_cast<uint32_t>(secondariesToExecute.size()), secondariesToExecute.data());
		}

		vkCmdEndRenderPass(commandBuffer);
	} else if (!pipelines_.empty() && !useCustomRecording_) {
		if (useDescriptorManager_ && descriptorManager_) {
			auto descriptorSets = descriptorManager_->getDescriptorSets();
			if (!descriptorSets.empty() && !pipelines_.empty()) {
				vkCmdBindDescriptorSets(commandBuffer,
										VK_PIPELINE_BIND_POINT_COMPUTE,
										pipelines_[0]->getLayout(),
										0,
										static_cast<uint32_t>(descriptorSets.size()),
										descriptorSets.data(),
										0,
										nullptr);
			}
		} else if (!buffers_.empty()) {
			vkCmdBindDescriptorSets(commandBuffer,
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
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipeline());

				for (const auto& pc : pushConstants_) {
					vkCmdPushConstants(commandBuffer, pipeline->getLayout(), pc.stageFlags, pc.offset, pc.size, pc.data.data());
				}

				vkCmdDispatch(commandBuffer, pipeline->workgroupSizeX_, pipeline->workgroupSizeY_, pipeline->workgroupSizeZ_);
			}
		}
	}

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		std::cerr << "Failed to end command buffer" << std::endl;
		return;
	}

	VkSubmitInfo submitInfo{};
	submitInfo.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount	= 1;
	submitInfo.pCommandBuffers		= &commandBuffer;
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
			// Wait on the acquire semaphore (from current frame)
			// Signal the render finished semaphore (specific to this swapchain image)
			waitSemaphore	= graphicsPipelines_[0]->getImageAvailableSemaphore();
			signalSemaphore = graphicsPipelines_[0]->getRenderFinishedSemaphore(imageIndex);

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

		VkFence submitFence = usesSwapchain ? graphicsPipelines_[0]->getInFlightFence() : fence_;

		if (!usesSwapchain && submitFence != VK_NULL_HANDLE) {
			vkResetFences(gpu_->device, 1, &submitFence);
		}

		if (vkQueueSubmit(queue, 1, &submitInfo, submitFence) != VK_SUCCESS) {
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

			graphicsPipelines_[0]->advanceFrame();
		}
	}

	currentFrame_ = (currentFrame_ + 1) % maxFramesInFlight_;
}
