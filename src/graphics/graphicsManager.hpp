#ifndef GRAPHICS_MANAGER_HPP
#define GRAPHICS_MANAGER_HPP

#include "gpuContext.hpp"
#include "graphicsTask.hpp"
#include "renderWindow.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace renderApi;

class GraphicsManager {
	private:
		GPUContext*									context_;
		RenderWindow*								window_;
		std::vector<std::unique_ptr<GraphicsTask>>	tasks_;
		std::mutex									tasksMutex_;

	public:
		GraphicsManager();
		~GraphicsManager();

		GraphicsManager(const GraphicsManager&)			   = delete;
		GraphicsManager& operator=(const GraphicsManager&) = delete;

		bool initialize(GPUContext& context, RenderWindow& window);
		void shutdown();
		GraphicsTask* createTask(const std::vector<uint32_t>& vertSpirv, const std::vector<uint32_t>& fragSpirv, const std::string& name = "");
		void addTask(std::unique_ptr<GraphicsTask> task);
		void removeTask(const std::string& name);
		GraphicsTask* getTask(const std::string& name);
		bool renderFrame();
		bool render();
		bool present();
		void waitIdle() const;
		RenderWindow* getWindow() const { return window_; }
};


#endif
