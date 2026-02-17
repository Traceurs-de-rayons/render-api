#include "renderDevice.hpp"

#include <vulkan/vulkan_core.h>

using namespace renderApi::device;

gpuLoopThreadResult renderApi::device::gpuThreadLoop(GPU& gpu) {
	while (true)
		;
	return THEARD_LOOP_SUCCESS;
}
