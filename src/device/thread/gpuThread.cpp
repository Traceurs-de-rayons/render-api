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

		// Another thread asked for the lock (GPU::lockTasks()): let it have it first.
		while (gpu.tasksLockRequests.load() > 0 && gpu.running)
			std::this_thread::yield();

		{
			std::lock_guard<std::mutex> lock(gpu.GpuTasksMutex);

			for (auto* task : gpu.GpuTasks) {
				if (task && task->isBuilt() && task->isEnabled() && task->isAutoExecute()) {
					task->execute();
					tasksToWait.push_back(task);
				}
			}

			if (!tasksToWait.empty()) {
				for (auto* task : tasksToWait) {
					task->wait();
				}
			}
		}

		if (tasksToWait.empty()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		} else {
			// Not enough alone for a thread waiting for the lock (see GPU::lockTasks()).
			std::this_thread::yield();
		}
	}
	return THEARD_LOOP_SUCCESS;
}
