#include "graphicsManager.hpp"
#include "gpuContext.hpp"
#include "graphicsTask.hpp"
#include "renderWindow.hpp"

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

GraphicsManager::GraphicsManager() : context_(nullptr), window_(nullptr) {}

GraphicsManager::~GraphicsManager() { shutdown(); }

bool GraphicsManager::initialize(GPUContext& context, RenderWindow& window) {
	if (!window.isValid()) {
		std::cerr << "[GraphicsManager] Window is not valid!" << std::endl;
		return false;
	}

	context_ = &context;
	window_	 = &window;

	std::cout << "[GraphicsManager] Initialized" << std::endl;
	return true;
}

void GraphicsManager::shutdown() {

	std::lock_guard<std::mutex> lock(tasksMutex_);
	tasks_.clear();
	context_ = nullptr;
	window_	 = nullptr;

	std::cout << "[GraphicsManager] Shutdown" << std::endl;
}

GraphicsTask* GraphicsManager::createTask(const std::vector<uint32_t>& vertSpirv,
										   const std::vector<uint32_t>& fragSpirv,
										   const std::string&			 name) {
	if (!context_ || !window_) {
		std::cerr << "[GraphicsManager] Not initialized!" << std::endl;
		return nullptr;
	}

	auto task = std::make_unique<GraphicsTask>();
	if (!task->create(*context_, *window_, name)) {
		return nullptr;
	}

	task->addShader(renderApi::ShaderStage::Vertex, vertSpirv, "vertex")
		 .addShader(renderApi::ShaderStage::Fragment, fragSpirv, "fragment");

	GraphicsTask* ptr = task.get();

	std::lock_guard<std::mutex> lock(tasksMutex_);
	tasks_.push_back(std::move(task));

	return ptr;
}

void GraphicsManager::addTask(std::unique_ptr<GraphicsTask> task) {
	if (!task || !task->isValid()) {
		std::cerr << "[GraphicsManager] Invalid task!" << std::endl;
		return;
	}

	std::lock_guard<std::mutex> lock(tasksMutex_);
	tasks_.push_back(std::move(task));
}

void GraphicsManager::removeTask(const std::string& name) {
	std::lock_guard<std::mutex> lock(tasksMutex_);

	auto it = std::remove_if(tasks_.begin(), tasks_.end(), [&name](const std::unique_ptr<GraphicsTask>& task) { return task->getName() == name; });

	tasks_.erase(it, tasks_.end());
}

GraphicsTask* GraphicsManager::getTask(const std::string& name) {
	std::lock_guard<std::mutex> lock(tasksMutex_);

	for (auto& task : tasks_) {
		if (task->getName() == name) {
			return task.get();
		}
	}

	return nullptr;
}

bool GraphicsManager::renderFrame() {
	if (!window_ || !context_)
		return false;
	if (!window_->acquireNextImage())
		return false;
	if (!render())
		return false;
	if (!present())
		return false;
	return true;
}

bool GraphicsManager::render() {
	if (!window_ || !context_)
		return false;

	VkFence fence = window_->getInFlightFence();
	vkWaitForFences(context_->getDevice(), 1, &fence, VK_TRUE, UINT64_MAX);
	vkResetFences(context_->getDevice(), 1, &fence);

	VkCommandBuffer cmd = context_->beginOneTimeCommands();
	VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};
	window_->beginRenderPass(cmd, clearColor);


	std::lock_guard<std::mutex> lock(tasksMutex_);
	VkFramebuffer				framebuffer = window_->getCurrentFramebuffer();
	VkRenderPass				renderPass	= window_->getRenderPass();
	VkExtent2D					extent		= window_->getExtent();

	for (auto& task : tasks_)
		if (task && task->isEnabled())
			task->bind(cmd, framebuffer, renderPass, extent);

	window_->endRenderPass(cmd);
	std::vector<VkSemaphore> waitSemaphores	  = {window_->getImageAvailableSemaphore()};
	std::vector<VkSemaphore> signalSemaphores = {window_->getRenderFinishedSemaphore()};

	if (!context_->submitGraphics(cmd, waitSemaphores, signalSemaphores, fence)) {
		std::cerr << "[GraphicsManager] Failed to submit graphics command!" << std::endl;
		return false;
	}

	return true;
}

bool GraphicsManager::present() {
	if (!window_)
		return false;
	return window_->present();
}

void GraphicsManager::waitIdle() const {
	if (context_)
		context_->waitIdle();
}
