#include "gpuTask.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "renderDevice.hpp"

#include <cstdint>
#include <iostream>
#include <mutex>
#include <vulkan/vulkan.h>

using namespace renderApi::gpuTask;

// Waits for the previous frame and, when presenting, acquires the next swapchain image.
// Returns false when the frame must be skipped.
bool GpuTask::acquireFrame(bool usesSwapchain, uint32_t& imageIndex) {
	if (!usesSwapchain) {
		vkWaitForFences(gpu_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
		vkResetFences(gpu_->device, 1, &fence_);
		return true;
	}

	VkFence inFlightFence = graphicsPipelines_[0]->getInFlightFence();
	if (inFlightFence != VK_NULL_HANDLE) {
		vkWaitForFences(gpu_->device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
	}

	VkSemaphore acquireSemaphore = graphicsPipelines_[0]->getImageAvailableSemaphore();

	VkResult result =
			vkAcquireNextImageKHR(gpu_->device, graphicsPipelines_[0]->getSwapchain(), UINT64_MAX, acquireSemaphore, VK_NULL_HANDLE, &imageIndex);

	if (result == VK_TIMEOUT) {
		std::cerr << "Warning: Acquire image timeout!" << std::endl;
		return false;
	} else if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		graphicsPipelines_[0]->recreateSwapchain();
		return false;
	} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		std::cerr << "Failed to acquire swapchain image: " << result << std::endl;
		return false;
	}

	auto& imagesInFlight = graphicsPipelines_[0]->imagesInFlight_;
	if (imageIndex < imagesInFlight.size() && imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
		vkWaitForFences(gpu_->device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
	}

	imagesInFlight[imageIndex] = inFlightFence;

	if (inFlightFence != VK_NULL_HANDLE) {
		vkResetFences(gpu_->device, 1, &inFlightFence);
	}
	return true;
}

// Submits the command buffer and, when presenting, presents the acquired image.
// Returns false when the submission failed.
bool GpuTask::submitFrame(VkCommandBuffer commandBuffer, uint32_t imageIndex, bool usesSwapchain) {
	VkSubmitInfo submitInfo{};
	submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers	  = &commandBuffer;

	VkQueue queue = VK_NULL_HANDLE;
	if (!graphicsPipelines_.empty()) {
		if (!gpu_->graphicsQueues.empty()) {
			queue = gpu_->graphicsQueues[0];
		}
	} else {
		if (!gpu_->computeQueues.empty()) {
			queue = gpu_->computeQueues[0];
		} else if (!gpu_->graphicsQueues.empty()) {
			queue = gpu_->graphicsQueues[0];
		}
	}

	if (queue == VK_NULL_HANDLE) {
		std::cerr << "Failed to submit queue: no available graphics or compute queue" << std::endl;
		return false;
	}

	std::lock_guard<std::mutex> lock(gpu_->queueMutex);

	VkSemaphore			 waitSemaphore	 = VK_NULL_HANDLE;
	VkSemaphore			 signalSemaphore = VK_NULL_HANDLE;
	VkPipelineStageFlags waitStage		 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	if (usesSwapchain) {
		waitSemaphore	= graphicsPipelines_[0]->getImageAvailableSemaphore();
		signalSemaphore = graphicsPipelines_[0]->getRenderFinishedSemaphore(imageIndex);

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores	  = &waitSemaphore;
		submitInfo.pWaitDstStageMask  = &waitStage;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores	= &signalSemaphore;
	}

	VkFence submitFence = usesSwapchain ? graphicsPipelines_[0]->getInFlightFence() : fence_;

	if (!usesSwapchain && submitFence != VK_NULL_HANDLE) {
		vkResetFences(gpu_->device, 1, &submitFence);
	}

	VkResult submitResult = vkQueueSubmit(queue, 1, &submitInfo, submitFence);
	if (submitResult != VK_SUCCESS) {
		std::cerr << "Failed to submit queue: " << submitResult << std::endl;
		return false;
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
	return true;
}
