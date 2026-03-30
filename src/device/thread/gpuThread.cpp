#include "renderDevice.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vulkan/vulkan_core.h>

using namespace renderApi::device;

gpuLoopThreadResult renderApi::device::gpuThreadLoop(GPU& gpu) {
	while (gpu.running) {
		std::vector<renderApi::gpuTask::GpuTask*> tasksToWait;

		{
			std::lock_guard<std::mutex> lock(gpu.GpuTasksMutex);

			for (auto* task : gpu.GpuTasks) {
				if (task && task->isBuilt() && task->isEnabled() && task->isAutoExecute()) {
					task->execute();
					tasksToWait.push_back(task);
				}
			}
		}

		if (!tasksToWait.empty()) {
			for (auto* task : tasksToWait) {
				task->wait();
			}
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	return THEARD_LOOP_SUCCESS;
}
