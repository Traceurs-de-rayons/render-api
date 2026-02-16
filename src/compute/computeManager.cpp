#include "computeManager.hpp"

#include "computeTask.hpp"
#include "gpuContext.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

using namespace renderApi;

ComputeManager::ComputeManager() : context_(nullptr) {}

ComputeManager::~ComputeManager() { shutdown(); }

bool ComputeManager::initialize(GPUContext& context) {
	context_ = &context;
	std::cout << "[ComputeManager] Initialized" << std::endl;
	return true;
}

void ComputeManager::shutdown() {

	std::lock_guard<std::mutex> lock(tasksMutex_);
	tasks_.clear();
	context_ = nullptr;

	std::cout << "[ComputeManager] Shutdown" << std::endl;
}

ComputeTask* ComputeManager::createTask(const std::vector<uint32_t>& spirvCode, const std::string& name) {
	if (!context_) {
		std::cerr << "[ComputeManager] Not initialized!" << std::endl;
		return nullptr;
	}

	auto task = std::make_unique<ComputeTask>();
	if (!task->create(*context_, name)) {
		return nullptr;
	}

	task->setShader(spirvCode, name);

	ComputeTask* ptr = task.get();

	std::lock_guard<std::mutex> lock(tasksMutex_);
	tasks_.push_back(std::move(task));

	return ptr;
}

void ComputeManager::addTask(std::unique_ptr<ComputeTask> task) {
	if (!task || !task->isValid()) {
		std::cerr << "[ComputeManager] Invalid task!" << std::endl;
		return;
	}

	std::lock_guard<std::mutex> lock(tasksMutex_);
	tasks_.push_back(std::move(task));
}

void ComputeManager::removeTask(const std::string& name) {
	std::lock_guard<std::mutex> lock(tasksMutex_);

	auto it = std::remove_if(tasks_.begin(), tasks_.end(), [&name](const std::unique_ptr<ComputeTask>& task) { return task->getName() == name; });

	tasks_.erase(it, tasks_.end());
}

ComputeTask* ComputeManager::getTask(const std::string& name) {
	std::lock_guard<std::mutex> lock(tasksMutex_);

	for (auto& task : tasks_) {
		if (task->getName() == name) {
			return task.get();
		}
	}

	return nullptr;
}

void ComputeManager::executeAll() {
	if (!context_)
		return;

	VkCommandBuffer cmd = context_->beginOneTimeCommands();

	{
		std::lock_guard<std::mutex> lock(tasksMutex_);
		for (auto& task : tasks_) {
			task->execute(cmd);
		}
	}

	context_->endOneTimeCommands(cmd);
}

void ComputeManager::waitIdle() const {
	if (context_) {
		context_->waitIdle();
	}
}
