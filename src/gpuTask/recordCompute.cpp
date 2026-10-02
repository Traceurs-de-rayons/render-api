#include "descriptor/descriptorSetManager.hpp"
#include "gpuTask.hpp"
#include "pipeline/computePipeline.hpp"

#include <cstdint>
#include <vulkan/vulkan.h>

using namespace renderApi::gpuTask;

// Default compute path: dispatches every enabled compute pipeline.
void GpuTask::recordCompute(VkCommandBuffer commandBuffer) {
	if (useDescriptorManager_ && descriptorManager_) {
		auto descriptorSets = descriptorManager_->getDescriptorSets();
		if (!descriptorSets.empty()) {
			vkCmdBindDescriptorSets(commandBuffer,
									VK_PIPELINE_BIND_POINT_COMPUTE,
									pipelines_[0]->getLayout(),
									0,
									static_cast<uint32_t>(descriptorSets.size()),
									descriptorSets.data(),
									0,
									nullptr);
		}
	} else if (!buffers_.empty()) {
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[0]->getLayout(), 0, 1, &descriptorSet_, 0, nullptr);
	}

	for (auto& pipeline : pipelines_) {
		if (pipeline->isEnabled()) {
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipeline());

			for (const auto& pc : pushConstants_) {
				vkCmdPushConstants(commandBuffer, pipeline->getLayout(), pc.stageFlags, pc.offset, pc.size, pc.data.data());
			}

			vkCmdDispatch(commandBuffer, pipeline->workgroupSizeX_, pipeline->workgroupSizeY_, pipeline->workgroupSizeZ_);
		}
	}
}
