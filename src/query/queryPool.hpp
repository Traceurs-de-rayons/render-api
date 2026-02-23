#ifndef QUERY_POOL_HPP
#define QUERY_POOL_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi::device {
	struct GPU;
}

namespace renderApi::query {

	enum class QueryType {
		TIMESTAMP,
		OCCLUSION,
		PIPELINE_STATISTICS
	};

	struct TimestampResult {
		std::string name;
		uint64_t	startTime;
		uint64_t	endTime;
		double		durationMs;
	};

	class QueryPool {
	  public:
		QueryPool();
		~QueryPool();

		QueryPool(const QueryPool&)			   = delete;
		QueryPool& operator=(const QueryPool&) = delete;
		QueryPool(QueryPool&& other) noexcept;
		QueryPool& operator=(QueryPool&& other) noexcept;

		bool create(device::GPU* gpu, QueryType type, uint32_t queryCount);
		void destroy();

		// Timestamp queries
		void writeTimestamp(VkCommandBuffer cmd, uint32_t queryIndex, VkPipelineStageFlagBits pipelineStage);
		void beginTimestamp(VkCommandBuffer cmd, const std::string& name);
		void endTimestamp(VkCommandBuffer cmd);

		// Reset queries
		void reset(VkCommandBuffer cmd);
		void resetRange(VkCommandBuffer cmd, uint32_t firstQuery, uint32_t queryCount);

		// Retrieve results
		bool getResults(std::vector<uint64_t>& results, bool wait = true);
		bool getTimestampResults(std::vector<TimestampResult>& results);

		// Get timing in milliseconds between two query indices
		double getTimingMs(uint32_t startIndex, uint32_t endIndex);

		VkQueryPool getHandle() const { return queryPool_; }
		uint32_t	getQueryCount() const { return queryCount_; }
		QueryType	getType() const { return type_; }
		bool		isValid() const { return queryPool_ != VK_NULL_HANDLE; }

		// Get timestamp period (nanoseconds per tick)
		float getTimestampPeriod() const { return timestampPeriod_; }

	  private:
		device::GPU* gpu_;
		VkQueryPool	 queryPool_;
		uint32_t	 queryCount_;
		QueryType	 type_;
		float		 timestampPeriod_;

		// Timestamp tracking
		uint32_t					  currentQueryIndex_;
		std::vector<std::string>	  timestampNames_;
		std::vector<uint32_t>		  timestampStartIndices_;

		VkQueryType convertQueryType(QueryType type) const;
	};

	// Helper function to create a timestamp query pool
	inline QueryPool createTimestampPool(device::GPU* gpu, uint32_t maxTimestamps = 64) {
		QueryPool pool;
		pool.create(gpu, QueryType::TIMESTAMP, maxTimestamps);
		return pool;
	}

} // namespace renderApi::query

#endif