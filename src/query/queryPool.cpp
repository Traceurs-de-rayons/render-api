#include "queryPool.hpp"

#include "renderDevice.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vulkan/vulkan_core.h>

using namespace renderApi::query;

QueryPool::QueryPool()
	: gpu_(nullptr), queryPool_(VK_NULL_HANDLE), queryCount_(0), type_(QueryType::TIMESTAMP), timestampPeriod_(1.0f), currentQueryIndex_(0) {}

QueryPool::~QueryPool() { destroy(); }

QueryPool::QueryPool(QueryPool&& other) noexcept
	: gpu_(other.gpu_), queryPool_(other.queryPool_), queryCount_(other.queryCount_), type_(other.type_),
	  timestampPeriod_(other.timestampPeriod_), currentQueryIndex_(other.currentQueryIndex_),
	  timestampNames_(std::move(other.timestampNames_)), timestampStartIndices_(std::move(other.timestampStartIndices_)) {
	other.queryPool_ = VK_NULL_HANDLE;
	other.queryCount_ = 0;
	other.currentQueryIndex_ = 0;
}

QueryPool& QueryPool::operator=(QueryPool&& other) noexcept {
	if (this != &other) {
		destroy();
		gpu_ = other.gpu_;
		queryPool_ = other.queryPool_;
		queryCount_ = other.queryCount_;
		type_ = other.type_;
		timestampPeriod_ = other.timestampPeriod_;
		currentQueryIndex_ = other.currentQueryIndex_;
		timestampNames_ = std::move(other.timestampNames_);
		timestampStartIndices_ = std::move(other.timestampStartIndices_);
		other.queryPool_ = VK_NULL_HANDLE;
		other.queryCount_ = 0;
		other.currentQueryIndex_ = 0;
	}
	return *this;
}

VkQueryType QueryPool::convertQueryType(QueryType type) const {
	switch (type) {
	case QueryType::TIMESTAMP:
		return VK_QUERY_TYPE_TIMESTAMP;
	case QueryType::OCCLUSION:
		return VK_QUERY_TYPE_OCCLUSION;
	case QueryType::PIPELINE_STATISTICS:
		return VK_QUERY_TYPE_PIPELINE_STATISTICS;
	}
	return VK_QUERY_TYPE_TIMESTAMP;
}

bool QueryPool::create(device::GPU* gpu, QueryType type, uint32_t queryCount) {
	destroy();

	gpu_ = gpu;
	queryCount_ = queryCount;
	type_ = type;
	currentQueryIndex_ = 0;

	if (!gpu_ || !gpu_->device) {
		std::cerr << "GPU not initialized" << std::endl;
		return false;
	}

	// Get timestamp period from physical device properties
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(gpu_->physicalDevice, &properties);
	timestampPeriod_ = properties.limits.timestampPeriod;

	VkQueryPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	poolInfo.queryType = convertQueryType(type_);
	poolInfo.queryCount = queryCount_;

	if (type_ == QueryType::PIPELINE_STATISTICS) {
		poolInfo.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
									  VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
									  VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
									  VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
									  VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
	}

	if (vkCreateQueryPool(gpu_->device, &poolInfo, nullptr, &queryPool_) != VK_SUCCESS) {
		std::cerr << "Failed to create query pool" << std::endl;
		return false;
	}

	timestampNames_.reserve(queryCount_);
	timestampStartIndices_.reserve(queryCount_ / 2);

	return true;
}

void QueryPool::destroy() {
	if (!gpu_ || !gpu_->device) return;

	if (queryPool_ != VK_NULL_HANDLE) {
		vkDestroyQueryPool(gpu_->device, queryPool_, nullptr);
		queryPool_ = VK_NULL_HANDLE;
	}

	queryCount_ = 0;
	currentQueryIndex_ = 0;
	timestampNames_.clear();
	timestampStartIndices_.clear();
}

void QueryPool::writeTimestamp(VkCommandBuffer cmd, uint32_t queryIndex, VkPipelineStageFlagBits pipelineStage) {
	if (!isValid() || type_ != QueryType::TIMESTAMP) {
		std::cerr << "Cannot write timestamp: invalid query pool or wrong type" << std::endl;
		return;
	}

	if (queryIndex >= queryCount_) {
		std::cerr << "Query index out of bounds: " << queryIndex << " >= " << queryCount_ << std::endl;
		return;
	}

	vkCmdWriteTimestamp(cmd, pipelineStage, queryPool_, queryIndex);
}

void QueryPool::beginTimestamp(VkCommandBuffer cmd, const std::string& name) {
	if (!isValid() || type_ != QueryType::TIMESTAMP) {
		std::cerr << "Cannot begin timestamp: invalid query pool or wrong type" << std::endl;
		return;
	}

	if (currentQueryIndex_ >= queryCount_) {
		std::cerr << "Query pool full: cannot begin timestamp '" << name << "'" << std::endl;
		return;
	}

	timestampStartIndices_.push_back(currentQueryIndex_);
	timestampNames_.push_back(name);

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, currentQueryIndex_);
	currentQueryIndex_++;
}

void QueryPool::endTimestamp(VkCommandBuffer cmd) {
	if (!isValid() || type_ != QueryType::TIMESTAMP) {
		std::cerr << "Cannot end timestamp: invalid query pool or wrong type" << std::endl;
		return;
	}

	if (timestampStartIndices_.empty()) {
		std::cerr << "No timestamp to end: beginTimestamp() not called" << std::endl;
		return;
	}

	if (currentQueryIndex_ >= queryCount_) {
		std::cerr << "Query pool full: cannot end timestamp" << std::endl;
		return;
	}

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, currentQueryIndex_);
	currentQueryIndex_++;
}

void QueryPool::reset(VkCommandBuffer cmd) {
	if (!isValid()) return;

	vkCmdResetQueryPool(cmd, queryPool_, 0, queryCount_);
	currentQueryIndex_ = 0;
	timestampNames_.clear();
	timestampStartIndices_.clear();
}

void QueryPool::resetRange(VkCommandBuffer cmd, uint32_t firstQuery, uint32_t queryCount) {
	if (!isValid()) return;

	if (firstQuery + queryCount > queryCount_) {
		std::cerr << "Query range out of bounds" << std::endl;
		return;
	}

	vkCmdResetQueryPool(cmd, queryPool_, firstQuery, queryCount);
}

bool QueryPool::getResults(std::vector<uint64_t>& results, bool wait) {
	if (!isValid()) return false;

	results.resize(queryCount_);

	VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT;
	if (wait) {
		flags |= VK_QUERY_RESULT_WAIT_BIT;
	}

	VkResult result = vkGetQueryPoolResults(gpu_->device,
											queryPool_,
											0,
											queryCount_,
											results.size() * sizeof(uint64_t),
											results.data(),
											sizeof(uint64_t),
											flags);

	if (result != VK_SUCCESS) {
		if (result == VK_NOT_READY) {
			return false;
		}
		std::cerr << "Failed to get query pool results" << std::endl;
		return false;
	}

	return true;
}

bool QueryPool::getTimestampResults(std::vector<TimestampResult>& results) {
	if (!isValid() || type_ != QueryType::TIMESTAMP) {
		return false;
	}

	if (timestampStartIndices_.empty()) {
		return true; // No timestamps recorded
	}

	std::vector<uint64_t> rawResults;
	if (!getResults(rawResults, true)) {
		return false;
	}

	results.clear();
	results.reserve(timestampStartIndices_.size());

	for (size_t i = 0; i < timestampStartIndices_.size(); ++i) {
		uint32_t startIdx = timestampStartIndices_[i];
		uint32_t endIdx = startIdx + 1;

		if (endIdx >= rawResults.size()) {
			break;
		}

		TimestampResult result;
		result.name = timestampNames_[i];
		result.startTime = rawResults[startIdx];
		result.endTime = rawResults[endIdx];

		// Convert to milliseconds
		uint64_t duration = result.endTime - result.startTime;
		result.durationMs = (double)duration * (double)timestampPeriod_ / 1000000.0;

		results.push_back(result);
	}

	return true;
}

double QueryPool::getTimingMs(uint32_t startIndex, uint32_t endIndex) {
	if (!isValid() || type_ != QueryType::TIMESTAMP) {
		return 0.0;
	}

	if (startIndex >= queryCount_ || endIndex >= queryCount_) {
		std::cerr << "Query indices out of bounds" << std::endl;
		return 0.0;
	}

	std::vector<uint64_t> results;
	if (!getResults(results, true)) {
		return 0.0;
	}

	uint64_t duration = results[endIndex] - results[startIndex];
	return (double)duration * (double)timestampPeriod_ / 1000000.0;
}