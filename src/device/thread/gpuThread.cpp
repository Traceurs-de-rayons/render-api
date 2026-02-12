#include "renderDevice.hpp"

using namespace renderApi::device;

gpuLoopThreadResult gpuThreadLoop(GPU& gpu) {
	while (gpu.running) {
		// Process GPU tasks
	}
	return THEARD_LOOP_SUCCESS;
}
