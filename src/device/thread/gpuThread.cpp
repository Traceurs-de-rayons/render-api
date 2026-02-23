#include "renderDevice.hpp"

#include <chrono>
#include <mutex>
#include <thread>
#include <vulkan/vulkan_core.h>

using namespace renderApi::device;

gpuLoopThreadResult renderApi::device::gpuThreadLoop(GPU& gpu) {
	while (gpu.running) {
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

		if (!anyTaskExecuted) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	return THEARD_LOOP_SUCCESS;
}
