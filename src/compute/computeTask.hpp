#ifndef COMPUTE_TASK_HPP
#define COMPUTE_TASK_HPP

#include "buffer.hpp"
#include "descriptors.hpp"

#include <atomic>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	class GPUContext;

	// Shader module structure for compute
	struct ComputeShaderModule {
		std::string				  name;
		std::vector<uint32_t>	  spirvCode;
		VkShaderModule			  module;
		std::string				  entryPoint;

		ComputeShaderModule()
			: module(VK_NULL_HANDLE), entryPoint("main") {}

		ComputeShaderModule(const std::vector<uint32_t>& code, const std::string& n = "",
							const std::string& entry = "main")
			: name(n), spirvCode(code), module(VK_NULL_HANDLE), entryPoint(entry) {}
	};

	// Buffer binding
	struct BufferBinding {
		uint32_t binding;
		Buffer*	 buffer;
	};

	class ComputeTask {
	  private:
		GPUContext* context_;
		std::string name_;

		// Shader support (compute pipeline only uses 1 shader, but architecture allows future flexibility)
		ComputeShaderModule shader_;
		bool				hasShader_;

		// Pipeline
		VkPipeline		 pipeline_;
		VkPipelineLayout pipelineLayout_;

		// Descriptors
		DescriptorLayout			descriptorLayout_;
		DescriptorSet				descriptorSet_;
		std::vector<BufferBinding>	bufferBindings_;

		// State
		bool built_;

		// Dispatch size
		uint32_t groupsX_;
		uint32_t groupsY_;
		uint32_t groupsZ_;

		std::atomic<bool> enabled_;

		// Internal methods
		bool createShaderModule();
		void destroyShaderModule();
		bool createPipeline();
		bool createDescriptors();

	  public:
		ComputeTask();
		~ComputeTask();

		ComputeTask(const ComputeTask&)			   = delete;
		ComputeTask& operator=(const ComputeTask&) = delete;

		ComputeTask(ComputeTask&& other) noexcept;
		ComputeTask& operator=(ComputeTask&& other) noexcept;

		// Initialization
		bool create(GPUContext& context, const std::string& name = "");
		void destroy();

		// Shader management - dynamic add/update
		ComputeTask& setShader(const std::vector<uint32_t>& spirvCode, 
							   const std::string& name = "", 
							   const std::string& entryPoint = "main");
		ComputeTask& updateShader(const std::vector<uint32_t>& spirvCode);
		bool hasShader() const { return hasShader_; }
		void clearShader();

		// Buffer bindings
		ComputeTask& bindBuffer(uint32_t binding, Buffer& buffer);

		// Build/rebuild pipeline
		bool build();
		bool rebuild();

		// Dispatch size
		void setDispatchSize(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1);

		// State management
		void setEnabled(bool enabled) { enabled_.store(enabled); }
		bool isEnabled() const { return enabled_.load(); }
		const std::string& getName() const { return name_; }
		bool isValid() const { return pipeline_ != VK_NULL_HANDLE; }
		bool isBuilt() const { return built_; }

		// Execution
		void execute(VkCommandBuffer cmd);
	};

}

#endif