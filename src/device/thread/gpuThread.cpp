#include "../renderDevice.hpp"

#include <chrono>
#include <thread>

using namespace renderApi::device;

gpuLoopThreadResult renderApi::device::gpuThreadLoop(GPU& gpu) {
	while (gpu.running) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return THEARD_LOOP_SUCCESS;
}
