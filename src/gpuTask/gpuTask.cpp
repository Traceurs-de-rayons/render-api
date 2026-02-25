#include "gpuTask.hpp"

#include "buffer/buffer.hpp"
#include "createDescriptorSetLayout.hpp"
#include "descriptor/descriptorSetManager.hpp"
#include "gpuTask/build.cpp"
#include "gpuTask/execute.cpp"
#include "gpuTask/secondaryCommandBuffer.cpp"
#include "pipeline/computePipeline.hpp"
#include "pipeline/graphicsPipeline.hpp"
#include "query/queryPool.hpp"
#include "renderDevice.hpp"

#include <SDL_timer.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

using namespace renderApi::gpuTask;
using namespace renderApi;

GpuTask::GpuTask(const std::string& name, device::GPU* gpu)
	: name_(name), gpu_(gpu), descriptorPool_(VK_NULL_HANDLE), descriptorSet_(VK_NULL_HANDLE), descriptorSetLayout_(VK_NULL_HANDLE),
	  commandPool_(VK_NULL_HANDLE), fence_(VK_NULL_HANDLE), isBuilt_(false) {}

GpuTask::~GpuTask() { destroy(); }

void GpuTask::addBuffer(Buffer* buffer, VkShaderStageFlags stageFlags) {
	if (isBuilt_) {
		std::cerr << "Cannot add buffer to built GPU task. Call destroy() first." << std::endl;
		return;
	}
	buffers_.push_back(buffer);
	bufferStages_.push_back(stageFlags);
}

void GpuTask::addVertexBuffer(Buffer* buffer) {
	if (isBuilt_) {
		std::cerr << "Cannot add vertex buffer to built GPU task. Call destroy() first." << std::endl;
		return;
	}
	vertexBuffers_.push_back(buffer);
}

void GpuTask::setIndexBuffer(Buffer* buffer, VkIndexType indexType) {
	if (isBuilt_) {
		std::cerr << "Cannot set index buffer to built GPU task. Call destroy() first." << std::endl;
		return;
	}
	indexBuffer_ = buffer;
	indexType_	 = indexType;
}

void GpuTask::setDrawParams(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
	vertexCount_   = vertexCount;
	instanceCount_ = instanceCount;
	firstVertex_   = firstVertex;
	firstInstance_ = firstInstance;
}

void GpuTask::setIndexedDrawParams(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
	indexCount_	   = indexCount;
	instanceCount_ = instanceCount;
	firstIndex_	   = firstIndex;
	vertexOffset_  = vertexOffset;
	firstInstance_ = firstInstance;
}

void GpuTask::removeBuffer(Buffer* buffer) {
	if (isBuilt_) {
		std::cerr << "Cannot remove buffer from built GPU task. Call destroy() first." << std::endl;
		return;
	}
	for (size_t i = 0; i < buffers_.size(); ++i) {
		if (buffers_[i] == buffer) {
			buffers_.erase(buffers_.begin() + i);
			bufferStages_.erase(bufferStages_.begin() + i);
			break;
		}
	}
}

void GpuTask::clearBuffers() {
	if (isBuilt_) {
		std::cerr << "Cannot clear buffers from built GPU task. Call destroy() first." << std::endl;
		return;
	}
	buffers_.clear();
	bufferStages_.clear();
}

void GpuTask::pushConstants(VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* data) {
	PushConstantData pcData;
	pcData.stageFlags = stageFlags;
	pcData.offset	  = offset;
	pcData.size		  = size;
	pcData.data.resize(size);
	memcpy(pcData.data.data(), data, size);

	pushConstants_.clear();
	pushConstants_.push_back(pcData);
}

void GpuTask::addRecordingCallback(RecordingCallback callback) { recordingCallbacks_.push_back(callback); }

void GpuTask::clearRecordingCallbacks() { recordingCallbacks_.clear(); }

void GpuTask::beginDefaultRenderPass(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
	if (graphicsPipelines_.empty()) {
		std::cerr << "GpuTask: No graphics pipeline available for render pass" << std::endl;
		return;
	}

	std::vector<VkClearValue> clearValues(2);
	clearValues[0].color		= {{0.2f, 0.2f, 0.2f, 1.0f}};
	clearValues[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType			 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass		 = graphicsPipelines_[0]->getRenderPass();
	renderPassInfo.framebuffer		 = graphicsPipelines_[0]->getSwapchainFramebuffer(imageIndex);
	renderPassInfo.renderArea.offset = {0, 0};
	renderPassInfo.renderArea.extent = {graphicsPipelines_[0]->getWidth(), graphicsPipelines_[0]->getHeight()};
	renderPassInfo.clearValueCount	 = static_cast<uint32_t>(clearValues.size());
	renderPassInfo.pClearValues		 = clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void GpuTask::endDefaultRenderPass(VkCommandBuffer commandBuffer) { vkCmdEndRenderPass(commandBuffer); }

descriptor::DescriptorSetManager* GpuTask::getDescriptorManager() {
	if (!descriptorManager_) {
		descriptorManager_ = std::make_unique<descriptor::DescriptorSetManager>();
	}
	return descriptorManager_.get();
}

void GpuTask::enableDescriptorManager(bool enable) { useDescriptorManager_ = enable; }

query::QueryPool* GpuTask::createQueryPool(uint32_t queryCount) {
	if (!queryPool_) {
		queryPool_ = std::make_unique<query::QueryPool>();
		if (!queryPool_->create(gpu_, query::QueryType::TIMESTAMP, queryCount)) {
			std::cerr << "Failed to create query pool" << std::endl;
			queryPool_.reset();
			return nullptr;
		}
	}
	return queryPool_.get();
}

ComputePipeline* GpuTask::createComputePipeline(const std::string& name) {
	auto  pipeline = std::make_unique<ComputePipeline>(gpu_, name);
	auto* ptr	   = pipeline.get();
	pipelines_.push_back(std::move(pipeline));
	return ptr;
}

GraphicsPipeline* GpuTask::createGraphicsPipeline(const std::string& name) {
	auto  pipeline = std::make_unique<GraphicsPipeline>(gpu_, name);
	auto* ptr	   = pipeline.get();
	graphicsPipelines_.push_back(std::move(pipeline));
	return ptr;
}

void GpuTask::wait() {
	if (gpu_ && gpu_->device && fence_ != VK_NULL_HANDLE) {
		vkWaitForFences(gpu_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
	}
}

void GpuTask::registerWithGPU() {
	if (!gpu_) {
		std::cerr << "Cannot register GpuTask: GPU is null" << std::endl;
		return;
	}

	std::lock_guard<std::mutex> lock(gpu_->GpuTasksMutex);

	for (auto* task : gpu_->GpuTasks) {
		if (task == this) {
			return;
		}
	}

	gpu_->GpuTasks.push_back(this);
}

void GpuTask::unregisterFromGPU() {
	if (!gpu_) {
		return;
	}

	std::lock_guard<std::mutex> lock(gpu_->GpuTasksMutex);

	auto it = std::find(gpu_->GpuTasks.begin(), gpu_->GpuTasks.end(), this);
	if (it != gpu_->GpuTasks.end()) {
		gpu_->GpuTasks.erase(it);
	}
}
