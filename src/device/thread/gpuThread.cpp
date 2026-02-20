#include "../renderDevice.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <vulkan/vulkan_core.h>

using namespace renderApi::device;

gpuLoopThreadResult renderApi::device::gpuThreadLoop(GPU& gpu) {
	// running flag is already set to true before this thread starts
	while (gpu.running) {
		// Process all auto-execute enabled GpuTasks
		bool anyTaskExecuted = false;
		
		{
			std::lock_guard<std::mutex> lock(gpu.GpuTasksMutex);
			
			for (auto* task : gpu.GpuTasks) {
				if (task && task->isBuilt() && task->isEnabled() && task->isAutoExecute()) {
					task->execute();
					anyTaskExecuted = true;
				}
			}
		}
		
		// If no tasks were executed, sleep to avoid busy waiting
		if (!anyTaskExecuted) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	return THEARD_LOOP_SUCCESS;
}
