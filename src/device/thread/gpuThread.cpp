#include "renderDevice.hpp"

#include <chrono>
#include <thread>
#include <vulkan/vulkan_core.h>

using namespace renderApi::device;

gpuLoopThreadResult renderApi::device::gpuThreadLoop(GPU& gpu) {
	// running flag is already set to true before this thread starts
	while (gpu.running) {
		// Process GPU tasks here
		// For now, just yield to avoid busy waiting
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return THEARD_LOOP_SUCCESS;
}
