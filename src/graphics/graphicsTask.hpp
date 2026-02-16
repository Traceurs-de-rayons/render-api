#ifndef GRAPHICS_TASK_HPP
#define GRAPHICS_TASK_HPP

#include "buffer.hpp"
#include "descriptors.hpp"
#include "renderWindow.hpp"

#include <atomic>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace renderApi {

	class GPUContext;

	enum class ShaderStage {
		Vertex,
		Fragment,
		Geometry,
		TessellationControl,
		TessellationEvaluation
	};

	// Shader module structure
	struct ShaderModule {
		ShaderStage				  stage;
		std::string				  name;
		std::vector<uint32_t>	  spirvCode;
		VkShaderModule			  module;
		std::string				  entryPoint;

		ShaderModule()
			: stage(ShaderStage::Vertex), module(VK_NULL_HANDLE), entryPoint("main") {}

		ShaderModule(ShaderStage s, const std::vector<uint32_t>& code, const std::string& n = "",
					 const std::string& entry = "main")
			: stage(s), name(n), spirvCode(code), module(VK_NULL_HANDLE), entryPoint(entry) {}
	};

	// Vertex buffer binding
	struct VertexBufferBinding {
		uint32_t binding;
		Buffer*	 buffer;
		uint32_t stride;
	};

	// Uniform buffer binding
	struct UniformBufferBinding {
		uint32_t binding;
		Buffer*	 buffer;
	};

	class GraphicsTask {
	  private:
		GPUContext*	  context_;
		RenderWindow* window_;
		std::string	  name_;

		// Multiple shaders support
		std::map<ShaderStage, ShaderModule> shaders_;
		std::map<ShaderStage, bool> shadersEnabled_;  // Stage -> enabled/disabled

		// Pipeline
		VkPipeline		 pipeline_;
		VkPipelineLayout pipelineLayout_;

		// Descriptors
		DescriptorLayout descriptorLayout_;
		DescriptorSet	 descriptorSet_;

		// Bindings
		std::vector<VertexBufferBinding>   vertexBindings_;
		Buffer*							   indexBuffer_;
		VkIndexType						   indexType_;
		std::vector<UniformBufferBinding>  uniformBindings_;

		// State
		bool built_;

		// Viewport & Scissor
		float	 viewportX_, viewportY_, viewportW_, viewportH_;
		int32_t	 scissorX_, scissorY_;
		uint32_t scissorW_, scissorH_;
		bool	 customViewport_;
		bool	 customScissor_;

		std::atomic<bool> enabled_;

		// Internal methods
		bool createShaderModule(ShaderModule& shader);
		void destroyShaderModule(ShaderModule& shader);
		bool createShaderModules();
		void destroyShaderModules();
		bool createPipeline(VkRenderPass renderPass);
		bool createDescriptors();

	  public:
		GraphicsTask();
		~GraphicsTask();

		GraphicsTask(const GraphicsTask&)			 = delete;
		GraphicsTask& operator=(const GraphicsTask&) = delete;

		GraphicsTask(GraphicsTask&& other) noexcept;
		GraphicsTask& operator=(GraphicsTask&& other) noexcept;

		// Initialization
		bool create(GPUContext& context, RenderWindow& window, const std::string& name = "");
		void destroy();

		// Shader management - dynamic add/remove/update
		GraphicsTask& addShader(ShaderStage stage, const std::vector<uint32_t>& spirvCode,
								const std::string& name = "", const std::string& entryPoint = "main");
		GraphicsTask& removeShader(ShaderStage stage);
		GraphicsTask& updateShader(ShaderStage stage, const std::vector<uint32_t>& spirvCode);
		bool hasShader(ShaderStage stage) const;
		void clearShaders();

		// Enable/disable specific shaders (requires rebuild)
		GraphicsTask& enableShader(ShaderStage stage);
		GraphicsTask& disableShader(ShaderStage stage);
		bool isShaderEnabled(ShaderStage stage) const;

		// Buffer bindings
		GraphicsTask& bindVertexBuffer(uint32_t binding, Buffer& buffer, uint32_t stride);
		GraphicsTask& bindIndexBuffer(Buffer& buffer, VkIndexType indexType = VK_INDEX_TYPE_UINT32);
		GraphicsTask& bindUniformBuffer(uint32_t binding, Buffer& buffer);

		// Build/rebuild pipeline
		bool build();
		bool rebuild();

		// Viewport & Scissor
		void setViewport(float x, float y, float width, float height);
		void setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height);
		void resetViewport() { customViewport_ = false; }
		void resetScissor() { customScissor_ = false; }

		// State management
		void setEnabled(bool enabled) { enabled_.store(enabled); }
		bool isEnabled() const { return enabled_.load(); }
		const std::string& getName() const { return name_; }
		bool isValid() const { return pipeline_ != VK_NULL_HANDLE; }
		bool isBuilt() const { return built_; }

		// Rendering
		void bind(VkCommandBuffer cmd, VkFramebuffer framebuffer, VkRenderPass renderPass, VkExtent2D extent);
	};

}

#endif
