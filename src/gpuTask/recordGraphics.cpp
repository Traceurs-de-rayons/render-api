#include "buffer/buffer.hpp"
#include "descriptor/descriptorSetManager.hpp"
#include "gpuTask.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "renderDevice.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#ifndef VK_EXT_mesh_shader
#define VK_EXT_mesh_shader			 1
#define VK_SHADER_STAGE_TASK_BIT_EXT ((VkShaderStageFlagBits)0x40)
#define VK_SHADER_STAGE_MESH_BIT_EXT ((VkShaderStageFlagBits)0x80)
typedef VkResult(VKAPI_PTR* PFN_vkCmdDrawMeshTasksEXT)(VkCommandBuffer commandBuffer,
													   uint32_t		   groupCountX,
													   uint32_t		   groupCountY,
													   uint32_t		   groupCountZ);
#endif

using namespace renderApi::gpuTask;

static PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT_fn = nullptr;

// Begins the render pass of the first graphics pipeline on its swapchain or offscreen framebuffer.
void GpuTask::beginGraphicsRenderPass(VkCommandBuffer commandBuffer, uint32_t imageIndex, bool usesSwapchain, VkSubpassContents contents) {
	std::vector<VkClearValue> clearValues(2);
	clearValues[0].color		= {{0.2f, 0.2f, 0.2f, 1.0f}};
	clearValues[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType	  = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = graphicsPipelines_[0]->getRenderPass();
	if (usesSwapchain) {
		renderPassInfo.framebuffer = graphicsPipelines_[0]->getSwapchainFramebuffer(imageIndex);
	} else {
		renderPassInfo.framebuffer = graphicsPipelines_[0]->getFramebuffer();
	}
	renderPassInfo.renderArea.offset = {0, 0};
	renderPassInfo.renderArea.extent = {graphicsPipelines_[0]->getWidth(), graphicsPipelines_[0]->getHeight()};
	renderPassInfo.clearValueCount	 = static_cast<uint32_t>(clearValues.size());
	renderPassInfo.pClearValues		 = clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, contents);
}

void GpuTask::drawMeshTasks(VkCommandBuffer commandBuffer) {
	if (!vkCmdDrawMeshTasksEXT_fn && gpu_->meshShaderSupported) {
		vkCmdDrawMeshTasksEXT_fn = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(gpu_->device, "vkCmdDrawMeshTasksEXT");
	}

	if (meshTaskCountX_ > 0 || meshTaskCountY_ > 0 || meshTaskCountZ_ > 0) {
		if (vkCmdDrawMeshTasksEXT_fn) {
			vkCmdDrawMeshTasksEXT_fn(commandBuffer, meshTaskCountX_, meshTaskCountY_, meshTaskCountZ_);
		} else {
			std::cerr << "GpuTask: Mesh shader function not available" << std::endl;
		}
	} else {
		std::cerr << "GpuTask: Mesh shader pipeline used but no task count set. Call setMeshTaskCount() or use classic draw." << std::endl;
	}
}

// Default graphics path: every enabled pipeline is drawn inline in one render pass.
void GpuTask::recordGraphicsInline(VkCommandBuffer commandBuffer, uint32_t imageIndex, bool usesSwapchain) {
	if (useDescriptorManager_ && descriptorManager_) {
		auto descriptorSets = descriptorManager_->getDescriptorSets();
		if (!descriptorSets.empty()) {
			vkCmdBindDescriptorSets(commandBuffer,
									VK_PIPELINE_BIND_POINT_GRAPHICS,
									graphicsPipelines_[0]->getLayout(),
									0,
									static_cast<uint32_t>(descriptorSets.size()),
									descriptorSets.data(),
									0,
									nullptr);
		}
	} else if (!buffers_.empty()) {
		vkCmdBindDescriptorSets(
				commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelines_[0]->getLayout(), 0, 1, &descriptorSet_, 0, nullptr);
	}

	beginGraphicsRenderPass(commandBuffer, imageIndex, usesSwapchain, VK_SUBPASS_CONTENTS_INLINE);

	if (!vertexBuffers_.empty()) {
		std::vector<VkBuffer>	  vkBuffers(vertexBuffers_.size());
		std::vector<VkDeviceSize> offsets(vertexBuffers_.size(), 0);
		for (size_t i = 0; i < vertexBuffers_.size(); ++i) {
			vkBuffers[i] = vertexBuffers_[i] && vertexBuffers_[i]->isValid() ? vertexBuffers_[i]->getHandle() : VK_NULL_HANDLE;
		}
		vkCmdBindVertexBuffers(commandBuffer, 0, static_cast<uint32_t>(vkBuffers.size()), vkBuffers.data(), offsets.data());
	}

	if (indexBuffer_ != nullptr && indexBuffer_->isValid()) {
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer_->getHandle(), 0, indexType_);
	}

	for (auto& pipeline : graphicsPipelines_) {
		if (pipeline->isEnabled()) {
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getPipeline());

			for (const auto& pc : pushConstants_) {
				vkCmdPushConstants(commandBuffer, pipeline->getLayout(), pc.stageFlags, pc.offset, pc.size, pc.data.data());
			}

			if (pipeline->isUsingMeshShader()) {
				drawMeshTasks(commandBuffer);
			} else if (indirectBuffer_ != nullptr && indirectBuffer_->isValid() && indirectDrawCount_ > 0) {
				vkCmdDrawIndexedIndirect(commandBuffer, indirectBuffer_->getHandle(), 0,
										indirectDrawCount_, sizeof(VkDrawIndexedIndirectCommand));
			} else if (indexBuffer_ != nullptr && indexBuffer_->isValid()) {
				vkCmdDrawIndexed(commandBuffer, indexCount_, instanceCount_, firstIndex_, vertexOffset_, firstInstance_);
			} else {
				vkCmdDraw(commandBuffer, vertexCount_, instanceCount_, firstVertex_, firstInstance_);
			}
		}
	}

	for (const auto& callback : renderPassCallbacks_) {
		callback(commandBuffer, currentFrame_, imageIndex);
	}

	vkCmdEndRenderPass(commandBuffer);
}

// Records one pipeline into its secondary command buffer.
void GpuTask::recordPipelineSecondary(VkCommandBuffer secondaryBuffer, GraphicsPipeline* pipeline) {
	if (!pipeline->isEnabled())
		return;

	vkCmdBindPipeline(secondaryBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getPipeline());

	if (!vertexBuffers_.empty()) {
		std::vector<VkBuffer>	  vkBuffers(vertexBuffers_.size());
		std::vector<VkDeviceSize> offsets(vertexBuffers_.size(), 0);
		for (size_t i = 0; i < vertexBuffers_.size(); ++i) {
			vkBuffers[i] = vertexBuffers_[i]->getHandle();
		}
		vkCmdBindVertexBuffers(secondaryBuffer, 0, static_cast<uint32_t>(vkBuffers.size()), vkBuffers.data(), offsets.data());
	}

	if (indexBuffer_ != nullptr) {
		vkCmdBindIndexBuffer(secondaryBuffer, indexBuffer_->getHandle(), 0, indexType_);
	}

	if (!buffers_.empty()) {
		vkCmdBindDescriptorSets(secondaryBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->getLayout(), 0, 1, &descriptorSet_, 0, nullptr);
	}

	for (const auto& pc : pushConstants_) {
		vkCmdPushConstants(secondaryBuffer, pipeline->getLayout(), pc.stageFlags, pc.offset, pc.size, pc.data.data());
	}

	if (pipeline->isUsingMeshShader()) {
		drawMeshTasks(secondaryBuffer);
	} else if (indexBuffer_ != nullptr) {
		vkCmdDrawIndexed(secondaryBuffer, indexCount_, instanceCount_, firstIndex_, vertexOffset_, firstInstance_);
	} else {
		vkCmdDraw(secondaryBuffer, vertexCount_, instanceCount_, firstVertex_, firstInstance_);
	}
}

// Graphics path with secondary command buffers: pipeline N is recorded into the enabled
// secondary buffer named "pipeline_N", then all of them are executed in one render pass.
void GpuTask::recordGraphicsSecondary(VkCommandBuffer commandBuffer, uint32_t imageIndex, bool usesSwapchain) {
	beginGraphicsRenderPass(commandBuffer, imageIndex, usesSwapchain, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

	std::vector<VkCommandBuffer> secondariesToExecute;

	for (size_t pipelineIdx = 0; pipelineIdx < graphicsPipelines_.size(); ++pipelineIdx) {
		std::string bufferName = "pipeline_" + std::to_string(pipelineIdx);

		VkCommandBuffer secondaryBuffer = VK_NULL_HANDLE;
		for (const auto& scb : secondaryCommandBuffers_) {
			if (scb.name == bufferName && scb.enabled) {
				secondaryBuffer = scb.buffer;
				break;
			}
		}
		if (secondaryBuffer == VK_NULL_HANDLE)
			continue;

		VkCommandBufferInheritanceInfo inheritanceInfo{};
		inheritanceInfo.sType	   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
		inheritanceInfo.renderPass = graphicsPipelines_[0]->getRenderPass();
		inheritanceInfo.subpass	   = 0;
		inheritanceInfo.framebuffer =
				usesSwapchain ? graphicsPipelines_[0]->getSwapchainFramebuffer(imageIndex) : graphicsPipelines_[0]->getFramebuffer();

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType			   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags			   = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = &inheritanceInfo;

		vkResetCommandBuffer(secondaryBuffer, 0);
		if (vkBeginCommandBuffer(secondaryBuffer, &beginInfo) == VK_SUCCESS) {
			recordPipelineSecondary(secondaryBuffer, graphicsPipelines_[pipelineIdx].get());
			vkEndCommandBuffer(secondaryBuffer);
			secondariesToExecute.push_back(secondaryBuffer);
		}
	}

	if (!secondariesToExecute.empty()) {
		vkCmdExecuteCommands(commandBuffer, static_cast<uint32_t>(secondariesToExecute.size()), secondariesToExecute.data());
	}

	vkCmdEndRenderPass(commandBuffer);
}
