#include "renderDevice.hpp"
#include "computeTask.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vulkan/vulkan_core.h>

using namespace renderApi::device;

gpuLoopThreadResult renderApi::device::gpuThreadLoop(GPU& gpu) {
	std::cout << "[GPU Thread] Started" << std::endl;

	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
	if (gpu.commandPool != VK_NULL_HANDLE) {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType				 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool		 = gpu.commandPool;
		allocInfo.level				 = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;

		if (vkAllocateCommandBuffers(gpu.device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
			std::cerr << "[GPU Thread] Failed to allocate command buffer!" << std::endl;
		}
	}

	while (gpu.running) {
		if (commandBuffer != VK_NULL_HANDLE) {
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			if (vkBeginCommandBuffer(commandBuffer, &beginInfo) == VK_SUCCESS) {
				std::lock_guard<std::mutex> lock(gpu.tasksMutex);
				for (auto& task : gpu.computeTasks) {
					if (task && task->isEnabled() && task->isBuilt()) {
						task->execute(commandBuffer);
					}
				}

				vkEndCommandBuffer(commandBuffer);

				if (gpu.computeQueue != VK_NULL_HANDLE) {
					VkSubmitInfo submitInfo{};
					submitInfo.sType			  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
					submitInfo.commandBufferCount = 1;
					submitInfo.pCommandBuffers	  = &commandBuffer;

					vkQueueSubmit(gpu.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
					vkQueueWaitIdle(gpu.computeQueue);
				}
				vkResetCommandBuffer(commandBuffer, 0);
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	if (commandBuffer != VK_NULL_HANDLE && gpu.commandPool != VK_NULL_HANDLE)
		vkFreeCommandBuffers(gpu.device, gpu.commandPool, 1, &commandBuffer);

	std::cout << "[GPU Thread] Stopped" << std::endl;
	return THEARD_LOOP_SUCCESS;
}
