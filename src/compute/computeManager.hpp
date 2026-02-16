#ifndef COMPUTE_MANAGER_HPP
#define COMPUTE_MANAGER_HPP

#include "computeTask.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace renderApi {

	class GPUContext;

	class ComputeManager {
		private:
			GPUContext*								  context_;
			std::vector<std::unique_ptr<ComputeTask>> tasks_;
			std::mutex								  tasksMutex_;

		public:
			ComputeManager();
			~ComputeManager();

			ComputeManager(const ComputeManager&)			 = delete;
			ComputeManager& operator=(const ComputeManager&) = delete;

			bool initialize(GPUContext& context);
			void shutdown();
			ComputeTask* createTask(const std::vector<uint32_t>& spirvCode, const std::string& name = "");
			void addTask(std::unique_ptr<ComputeTask> task);
			void removeTask(const std::string& name);
			ComputeTask* getTask(const std::string& name);
			void executeAll();
			void waitIdle() const;
	};

}

#endif
